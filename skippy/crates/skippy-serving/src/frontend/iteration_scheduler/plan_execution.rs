//! Running a scheduler plan as one native batch, and containing its failure.

use super::*;
use skippy_runtime::IterationSample;

impl SchedulerWorker {
    pub(super) fn execute_plan(
        &self,
        plan: &skippy_scheduler::IterationPlan,
    ) -> InferenceResult<Vec<IterationPrediction>> {
        let mut runtime = lock_runtime(&self.runtime);
        let hold_started = Instant::now();
        let requests = plan
            .work
            .iter()
            .map(|work| RuntimeIterationBatchRequest {
                session_id: &work.sequence_id,
                token_ids: &work.tokens,
                // Text-only serving leaves positions to the native runtime so
                // model-specific widths (for example Qwen mRoPE's four
                // position dimensions) are expanded correctly.
                positions: &[],
                sampling: work.sampling.as_ref(),
                input: None,
                sample_last: work.sample_last,
                phase: match work.phase {
                    IterationPhase::Decode => IterationBatchPhase::Decode,
                    IterationPhase::Prefill | IterationPhase::Recompute => {
                        IterationBatchPhase::Prefill
                    }
                },
            })
            .collect::<Vec<_>>();
        let result = runtime.iteration_batch_sampled(&requests);
        drop(runtime);
        self.record_compute(hold_started.elapsed());
        let decode_steps = plan
            .work
            .iter()
            .filter(|work| work.phase == IterationPhase::Decode)
            .count();
        self.compute_meter.record_decode_tokens(decode_steps as u64);
        let outputs = result.map_err(openai_backend_error)?;
        plan_predictions(plan.work.len(), outputs.samples)
    }

    /// Fail every request with work in `plan`, and only those.
    ///
    /// A failed native batch says nothing about requests outside it: queued
    /// requests, other planned requests and direct iterations keep running.
    pub(super) fn fail_plan_requests(
        &mut self,
        plan: &skippy_scheduler::IterationPlan,
        error: &InferenceError,
    ) {
        let ids = plan
            .work
            .iter()
            .map(|work| work.sequence_id.clone())
            .collect::<BTreeSet<_>>();
        for id in ids {
            self.fail_request(&id, error.clone());
        }
    }

    /// Fail the requests the scheduler stopped because their next decode had
    /// no KV room.
    pub(super) fn fail_out_of_decode_room(&mut self, plan: &skippy_scheduler::IterationPlan) {
        for id in &plan.out_of_decode_room {
            self.fail_request(
                id,
                InferenceError::backend(format!(
                    "the KV cache has no room for request {id} to decode another token"
                )),
            );
        }
    }
}

/// Map native samples onto plan work, rejecting any the plan cannot own.
///
/// A sample for a missing row, or a second sample for one row, means the
/// native batch's results cannot be attributed safely. The scheduler would
/// fail a duplicated row while the worker streamed one of its tokens, so the
/// whole batch fails instead.
fn plan_predictions(
    work_count: usize,
    samples: Vec<IterationSample>,
) -> InferenceResult<Vec<IterationPrediction>> {
    let mut sampled = vec![false; work_count];
    samples
        .into_iter()
        .map(|sample| {
            let index = sample.request_index;
            let Some(seen) = sampled.get_mut(index) else {
                return Err(InferenceError::backend(format!(
                    "native iteration sampled request {index}, but the plan has {work_count} rows"
                )));
            };
            if std::mem::replace(seen, true) {
                return Err(InferenceError::backend(format!(
                    "native iteration sampled request {index} more than once"
                )));
            }
            Ok(IterationPrediction {
                work_index: index,
                token: sample.predicted_token,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker over a runtime with no lanes: any native batch it runs fails
    /// before reaching native code, as a real batch error would.
    fn worker_with_failing_batches(lanes: usize, kv_tokens: usize) -> SchedulerWorker {
        let (_commands, receiver) = std_mpsc::channel();
        SchedulerWorker {
            compute_meter: Arc::default(),
            runtime: Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(0))),
            scheduler: Scheduler::new(build_scheduler_config(
                lanes,
                kv_tokens,
                0,
                Some(8),
                Some(8),
                8,
                false,
            )),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: kv_tokens,
            max_direct_batch_size: lanes,
            direct_group_batch_size: lanes,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
    }

    fn submit(
        worker: &mut SchedulerWorker,
        id: &str,
        prompt_tokens: Vec<i32>,
    ) -> std_mpsc::Receiver<SchedulerEvent> {
        let (reply, events) = std_mpsc::channel();
        worker.submit(ScheduledRequest {
            id: id.into(),
            prompt_tokens,
            max_tokens: 8,
            sampling: None,
            chat_sampling_metadata: None,
            generated_tokens: Vec::new(),
            runtime_configured: true,
            retain_runtime: false,
            reply,
        });
        events
    }

    fn queue_direct_iteration(worker: &mut SchedulerWorker, session_id: &str) {
        let (reply, _result) = std_mpsc::sync_channel(1);
        worker.direct_iterations.push_back(DirectIteration {
            session_id: session_id.into(),
            target_token_count: None,
            token_ids: vec![1],
            positions: Vec::new(),
            sampling: None,
            input: None,
            sample_last: true,
            selected_tokens: None,
            phase: IterationBatchPhase::Prefill,
            deadline: None,
            cancellation: None,
            enqueued_at: Instant::now(),
            reply: reply.into(),
        });
    }

    fn expect_error(events: &std_mpsc::Receiver<SchedulerEvent>) -> InferenceError {
        loop {
            match events.try_recv() {
                Ok(SchedulerEvent::Error(error)) => return error,
                Ok(SchedulerEvent::Token { .. }) => continue,
                Ok(SchedulerEvent::Complete) => panic!("request completed instead of failing"),
                Err(error) => panic!("request was not failed: {error}"),
            }
        }
    }

    #[test]
    fn a_failed_batch_fails_only_its_own_requests() {
        let mut worker = worker_with_failing_batches(1, 64);
        let batched = submit(&mut worker, "batched", vec![1, 2]);
        // One lane: this request waits behind the batched one.
        let queued = submit(&mut worker, "queued", vec![3, 4]);
        queue_direct_iteration(&mut worker, "direct");
        // Serve planned work this turn rather than the direct queue.
        worker.last_served_direct = true;

        worker.run_iteration();

        let error = expect_error(&batched);
        assert!(error.message().contains("execution lanes"), "{error}");
        assert!(!worker.requests.contains_key("batched"));
        assert!(worker.requests.contains_key("queued"));
        assert!(worker.scheduler.sequence("queued").is_some());
        assert!(matches!(
            queued.try_recv(),
            Err(std_mpsc::TryRecvError::Empty)
        ));
        assert_eq!(worker.direct_iterations.len(), 1);
    }

    #[test]
    fn a_request_without_decode_room_fails_alone() {
        // The KV pool holds exactly the first request's 5-token prompt.
        let mut worker = worker_with_failing_batches(2, 5);
        let full = submit(&mut worker, "full", vec![1, 2, 3, 4, 5]);
        let waiting = submit(&mut worker, "waiting", vec![6]);
        let plan = worker.scheduler.plan_iteration();
        assert_eq!(plan.work.len(), 1);
        let predictions = [IterationPrediction {
            work_index: 0,
            token: 10,
        }];
        worker.scheduler.complete_iteration(&plan, &predictions);
        worker.finish_iteration(&plan, &predictions);

        // Its first decode needs a sixth cell and nothing else can run to free
        // one, so it alone is stopped before any native batch runs.
        worker.run_iteration();

        let error = expect_error(&full);
        assert!(error.message().contains("no room"), "{error}");
        assert!(worker.requests.contains_key("waiting"));
        assert!(worker.scheduler.sequence("waiting").is_some());
        assert!(matches!(
            waiting.try_recv(),
            Err(std_mpsc::TryRecvError::Empty)
        ));
    }

    fn sample(request_index: usize, predicted_token: i32) -> IterationSample {
        IterationSample {
            request_index,
            predicted_token,
        }
    }

    #[test]
    fn plan_predictions_keep_each_row_sample() {
        let predictions = plan_predictions(3, vec![sample(0, 10), sample(2, 30)]).unwrap();
        assert_eq!(
            predictions,
            [
                IterationPrediction {
                    work_index: 0,
                    token: 10,
                },
                IterationPrediction {
                    work_index: 2,
                    token: 30,
                },
            ]
        );
    }

    #[test]
    fn plan_predictions_reject_a_row_sampled_twice() {
        let error = plan_predictions(2, vec![sample(1, 10), sample(1, 11)]).unwrap_err();
        assert!(error.message().contains("more than once"), "{error}");
    }

    #[test]
    fn plan_predictions_reject_a_row_outside_the_plan() {
        let error = plan_predictions(2, vec![sample(0, 10), sample(2, 11)]).unwrap_err();
        assert!(error.message().contains("plan has 2 rows"), "{error}");
    }
}
