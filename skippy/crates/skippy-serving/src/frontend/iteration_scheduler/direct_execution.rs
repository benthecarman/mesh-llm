//! Execute admitted direct iterations under one native runtime lease.

use super::*;
use skippy_runtime::IterationBatchOutput;

type Runnable = (DirectIteration, Option<RuntimeSessionAlignStats>);

/// The session position a direct iteration runs against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SessionTarget {
    /// Trim a session that is ahead of this token count.
    TrimAhead(u64),
    /// Trim a session that is ahead, then require it to stand at this token
    /// count: a peer stage's decode names the position it must run at.
    Exact(u64),
}

impl SessionTarget {
    fn token_count(self) -> u64 {
        match self {
            Self::TrimAhead(token_count) | Self::Exact(token_count) => token_count,
        }
    }
}

struct BatchTiming {
    wait: Vec<f64>,
    lock_wait_ms: f64,
    lock_hold_ms: f64,
}

impl SchedulerWorker {
    /// Align and execute a direct batch, then deliver each request's own result.
    pub(super) fn run_direct_iteration_batch(&mut self) {
        let batch = take_direct_iteration_batch(
            &mut self.direct_iterations,
            self.direct_group_batch_size,
            self.max_direct_iteration_tokens,
        );
        debug_assert!(!batch.is_empty(), "validated direct queue must yield work");
        let active = retain_active(batch);
        if active.is_empty() {
            return;
        }
        let lock_started = Instant::now();
        let mut runtime = lock_runtime(&self.runtime);
        let lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
        let hold_started = Instant::now();
        let runnable = align_requests(&mut runtime, active);
        if runnable.is_empty() {
            return;
        }
        let requests = runnable
            .iter()
            .map(|(request, _)| RuntimeIterationBatchRequest {
                session_id: &request.session_id,
                token_ids: &request.token_ids,
                positions: &request.positions,
                sampling: request.sampling.as_ref(),
                input: request.input.as_ref(),
                sample_last: request.sample_last,
                phase: request.phase,
            })
            .collect::<Vec<_>>();
        #[cfg(test)]
        if requests
            .iter()
            .any(|request| request.session_id == tests::PANICKING_DIRECT_SESSION)
        {
            panic!("direct iteration batch panicked");
        }
        let selections = runnable
            .iter()
            .map(|(request, _)| request.selected_tokens.as_deref().unwrap_or_default())
            .collect::<Vec<_>>();
        let wait = runnable
            .iter()
            .map(|(request, _)| request.enqueued_at.elapsed().as_secs_f64() * 1_000.0)
            .collect();
        let result = runtime
            .iteration_batch_scored(&requests, &selections)
            .map_err(openai_backend_error);
        let hold = hold_started.elapsed();
        drop(runtime);
        self.record_compute(hold);
        let timing = BatchTiming {
            wait,
            lock_wait_ms,
            lock_hold_ms: hold.as_secs_f64() * 1_000.0,
        };
        self.record_direct_batch(&runnable, &timing);
        match result {
            Ok((output, scores)) => send_outputs(runnable, output, scores, timing),
            Err(error) => fail_requests(runnable, error),
        }
    }

    /// Record execution and timing evidence for one native direct batch.
    fn record_direct_batch(&self, requests: &[Runnable], timing: &BatchTiming) {
        let decode_steps = requests
            .iter()
            .filter(|(request, _)| request.phase == IterationBatchPhase::Decode)
            .count();
        self.compute_meter.record_decode_tokens(decode_steps as u64);
        if let Some(telemetry) = self.telemetry.as_ref() {
            telemetry.emit_debug(
                "stage.scheduler_feature_iteration",
                BTreeMap::from([
                    (
                        "skippy.scheduler.batch_size".to_string(),
                        json!(requests.len()),
                    ),
                    (
                        "skippy.scheduler.token_count".to_string(),
                        json!(
                            requests
                                .iter()
                                .map(|(r, _)| r.token_ids.len())
                                .sum::<usize>()
                        ),
                    ),
                    (
                        "skippy.scheduler.batch_wait_max_ms".to_string(),
                        json!(timing.wait.iter().copied().fold(0.0_f64, f64::max)),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_wait_ms".to_string(),
                        json!(timing.lock_wait_ms),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_hold_ms".to_string(),
                        json!(timing.lock_hold_ms),
                    ),
                ]),
            );
        }
    }
}

/// Remove cancelled requests before a direct batch enters native execution.
fn retain_active(batch: Vec<DirectIteration>) -> Vec<DirectIteration> {
    batch
        .into_iter()
        .filter_map(|request| match request.ensure_active() {
            Ok(()) => Some(request),
            Err(error) => {
                request.reply.send(Err(error));
                None
            }
        })
        .collect()
}

/// Align every retained request with its session's current token position.
fn align_requests(runtime: &mut RuntimeState, batch: Vec<DirectIteration>) -> Vec<Runnable> {
    batch
        .into_iter()
        .filter_map(|request| {
            let alignment = request.ensure_active().and_then(|()| match request.target {
                Some(target) => align_request(runtime, &request, target),
                None => Ok(None),
            });
            match alignment {
                Ok(alignment) => Some((request, alignment)),
                Err(error) => {
                    request.reply.send(Err(error));
                    None
                }
            }
        })
        .collect()
}

/// Align one request or report its error without corrupting the remaining batch.
fn align_request(
    runtime: &mut RuntimeState,
    request: &DirectIteration,
    target: SessionTarget,
) -> InferenceResult<Option<RuntimeSessionAlignStats>> {
    if request.selected_tokens.is_some() && !runtime.model.supports_causal_readout() {
        return Err(InferenceError::unsupported(
            "loaded stage does not support causal readouts",
        ));
    }
    let alignment = align_session_to_target(runtime, &request.session_id, target)?;
    let target = target.token_count();
    if request.selected_tokens.is_some()
        && runtime
            .session_token_count(&request.session_id)
            .unwrap_or(0)
            != target
    {
        return Err(InferenceError::invalid_request(
            "decision readout position does not match session prefix",
        ));
    }
    Ok(alignment)
}

/// Trim a session that is ahead of `target`; an exact target then also
/// requires the session to stand there, without admitting a lane for it.
fn align_session_to_target(
    runtime: &mut RuntimeState,
    session_id: &str,
    target: SessionTarget,
) -> InferenceResult<Option<RuntimeSessionAlignStats>> {
    let alignment = runtime
        .align_session_to_token_count_if_ahead(session_id, target.token_count())
        .map_err(openai_backend_error)?;
    if let SessionTarget::Exact(position) = target {
        runtime
            .ensure_session_at_position(session_id, position)
            .map_err(|error| InferenceError::invalid_request(format!("{error:#}")))?;
    }
    Ok(alignment)
}

/// Deliver the same terminal failure to every affected request.
fn fail_requests(requests: Vec<Runnable>, error: InferenceError) {
    for (request, _) in requests {
        request.reply.send(Err(error.clone()));
    }
}

/// Map sparse native predictions back to their original request slots.
fn prediction_slots(
    output: &IterationBatchOutput,
    count: usize,
    score_count: usize,
) -> InferenceResult<Vec<Option<i32>>> {
    if output.request_outputs.len() != count || score_count != count {
        return Err(InferenceError::backend(format!(
            "scheduler returned {} outputs and {score_count} readouts for {count} requests",
            output.request_outputs.len()
        )));
    }
    let mut predicted = vec![None; count];
    for sample in &output.samples {
        let Some(slot) = predicted.get_mut(sample.request_index) else {
            return Err(InferenceError::backend(format!(
                "scheduler sample references request {}, but only {count} requests ran",
                sample.request_index
            )));
        };
        if slot.replace(sample.predicted_token).is_some() {
            return Err(InferenceError::backend(format!(
                "scheduler returned duplicate sample for request {}",
                sample.request_index
            )));
        }
    }
    Ok(predicted)
}

/// Deliver frames, predictions, scores and timing to their matching request owners.
fn send_outputs(
    requests: Vec<Runnable>,
    output: IterationBatchOutput,
    scores: Vec<Vec<f32>>,
    timing: BatchTiming,
) {
    let predicted = match prediction_slots(&output, requests.len(), scores.len()) {
        Ok(predicted) => predicted,
        Err(error) => {
            fail_requests(requests, error);
            return;
        }
    };
    let batch_size = requests.len();
    for (((((request, session_alignment), output), predicted), batch_wait_ms), log_probabilities) in
        requests
            .into_iter()
            .zip(output.request_outputs)
            .zip(predicted)
            .zip(timing.wait)
            .zip(scores)
    {
        request.reply.send(Ok(SchedulerIterationOutcome {
            predicted: predicted.unwrap_or(-1),
            log_probabilities,
            output,
            batch_size,
            batch_wait_ms,
            runtime_lock_wait_ms: timing.lock_wait_ms,
            runtime_lock_hold_ms: timing.lock_hold_ms,
            session_alignment,
        }));
    }
}

#[cfg(test)]
mod session_target_tests {
    use super::*;

    fn decode_iteration(
        session_id: &str,
        target: SessionTarget,
    ) -> (
        DirectIteration,
        std_mpsc::Receiver<InferenceResult<SchedulerIterationOutcome>>,
    ) {
        let (reply, result) = std_mpsc::sync_channel(1);
        let request = DirectIteration {
            session_id: session_id.to_string(),
            target: Some(target),
            token_ids: vec![7],
            positions: Vec::new(),
            sampling: None,
            input: None,
            sample_last: true,
            selected_tokens: None,
            phase: IterationBatchPhase::Decode,
            deadline: None,
            cancellation: None,
            enqueued_at: Instant::now(),
            reply: reply.into(),
        };
        (request, result)
    }

    /// Submits one frame iteration through `submit` and returns the target
    /// the scheduler worker would see.
    fn submitted_target(
        submit: impl FnOnce(&IterationScheduler) -> InferenceResult<SchedulerIterationOutcome>
        + Send
        + 'static,
    ) -> Option<SessionTarget> {
        let (commands, receiver) = std_mpsc::sync_channel(1);
        let scheduler = IterationScheduler {
            shared: Arc::new(IterationSchedulerShared {
                commands,
                max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
                owner_count: AtomicUsize::new(1),
                worker: Mutex::new(None),
            }),
        };
        let caller = thread::spawn(move || submit(&scheduler));
        let SchedulerCommand::ExecuteIteration(request) = receiver.recv().unwrap() else {
            panic!("expected direct iteration command");
        };
        let target = request.target;
        request
            .reply
            .send(Err(InferenceError::backend("not executed")));
        assert!(caller.join().unwrap().is_err());
        target
    }

    #[test]
    fn peer_decode_frames_require_their_exact_position() {
        let target = submitted_target(|scheduler| {
            scheduler.execute_peer_decode_frame_iteration("s", 5, &[7], &[], None, None, true)
        });
        assert_eq!(target, Some(SessionTarget::Exact(5)));
        let target = submitted_target(|scheduler| {
            scheduler.execute_frame_iteration("s", 5, &[7], &[], None, None, true)
        });
        assert_eq!(target, Some(SessionTarget::TrimAhead(5)));
    }

    #[test]
    fn the_worker_fails_a_frame_at_the_wrong_position_before_native_execution() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("behind", 3);
        let (behind, behind_result) = decode_iteration("behind", SessionTarget::Exact(5));
        let (unknown, unknown_result) = decode_iteration("unknown", SessionTarget::Exact(5));
        // Nothing is left to run, so the modelless runtime is never reached.
        assert!(align_requests(&mut runtime, vec![behind, unknown]).is_empty());
        for (result, expected) in [
            (
                behind_result,
                "is at position 3, but the decode is at position 5",
            ),
            (unknown_result, "is not active"),
        ] {
            let error = result.recv().unwrap().unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
        assert_eq!(runtime.session_stats().active_sessions, 0);
    }

    #[test]
    fn an_exact_target_rejects_a_session_behind_it_or_unknown() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("behind", 3);
        for session_id in ["behind", "unknown"] {
            let error = align_session_to_target(&mut runtime, session_id, SessionTarget::Exact(5))
                .unwrap_err();
            assert_eq!(
                error.status(),
                InferenceError::invalid_request("").status(),
                "{session_id}: {error}"
            );
        }
        assert_eq!(runtime.session_token_count("unknown"), None);
        assert_eq!(runtime.session_stats().active_sessions, 0);
    }

    #[test]
    fn an_exact_target_admits_its_own_position_and_a_fresh_session_at_zero() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("at", 5);
        assert!(align_session_to_target(&mut runtime, "at", SessionTarget::Exact(5)).is_ok());
        assert!(align_session_to_target(&mut runtime, "fresh", SessionTarget::Exact(0)).is_ok());
    }

    #[test]
    fn a_trim_target_still_runs_a_session_behind_it() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("behind", 3);
        assert!(
            align_session_to_target(&mut runtime, "behind", SessionTarget::TrimAhead(5)).is_ok()
        );
    }
}
