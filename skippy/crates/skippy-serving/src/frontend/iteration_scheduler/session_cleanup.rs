//! Session cleanup that cannot be shed by a full command queue.
//!
//! Ordinary runtime work fails fast when the scheduler's bounded command
//! queue is full: the request gets a retryable rate-limit error and nothing
//! has changed. A session drop is different. Its request is already over, so
//! no one retries it, and a drop rejected under load left the session holding
//! its lane and KV cells until the process restarted. Cleanup therefore waits
//! for room in the queue instead.
//!
//! Waiting cannot deadlock the worker: callers are request or connection
//! threads that already block on the operation's result, and the worker
//! drains the queue every turn without waiting on those threads.

use super::*;

impl IterationScheduler {
    /// Runs session cleanup on the scheduler worker, waiting for queue room
    /// rather than failing when the command queue is full.
    pub(crate) fn execute_session_cleanup<T>(
        &self,
        label: &'static str,
        operation: impl FnOnce(&mut RuntimeState) -> InferenceResult<T> + Send + 'static,
    ) -> InferenceResult<T>
    where
        T: Send + 'static,
    {
        self.execute_session_cleanup_timed(label, operation)
            .map(|outcome| outcome.value)
    }

    /// [`Self::execute_session_cleanup`] with the queue and lock timings.
    pub(crate) fn execute_session_cleanup_timed<T>(
        &self,
        label: &'static str,
        operation: impl FnOnce(&mut RuntimeState) -> InferenceResult<T> + Send + 'static,
    ) -> InferenceResult<SchedulerRuntimeOutcome<T>>
    where
        T: Send + 'static,
    {
        let (operation, result) = runtime_operation(label, operation);
        self.shared
            .commands
            .send(SchedulerCommand::ExecuteRuntime(operation))
            .map_err(|_| InferenceError::backend("iteration scheduler stopped"))?;
        result.recv().map_err(|error| {
            InferenceError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::TelemetryLevel;

    /// Upper bound on commands needed to fill the queue; the real capacity
    /// is `MIN_COMMAND_QUEUE_CAPACITY` for this configuration.
    const QUEUE_FILL_LIMIT: usize = 1024;

    /// How long a cleanup gets to show it was shed. Only a regression can
    /// finish inside it, so a slow machine can hide a regression but never
    /// fail a correct build.
    const SHED_CLEANUP_GRACE: Duration = Duration::from_millis(250);

    fn scheduler_over(runtime: &Arc<Mutex<RuntimeState>>) -> IterationScheduler {
        let config = StageConfig {
            run_id: "session-cleanup-test".to_string(),
            topology_id: "session-cleanup-test".to_string(),
            model_id: "session-cleanup-test".to_string(),
            ..StageConfig::default()
        };
        let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
        IterationScheduler::new(Arc::clone(runtime), &config, 1, true, None, telemetry).unwrap()
    }

    /// Parks the worker inside a runtime operation until the returned
    /// sender fires, then fills the command queue behind it.
    fn block_worker_with_full_queue(scheduler: &IterationScheduler) -> std_mpsc::Sender<()> {
        let (started, worker_started) = std_mpsc::channel();
        let (release, released) = std_mpsc::channel::<()>();
        scheduler
            .execute_runtime_detached("blocked-worker", move |_| {
                started.send(()).unwrap();
                let _ = released.recv();
            })
            .unwrap();
        worker_started.recv().unwrap();
        let mut queued = 0;
        while scheduler
            .execute_runtime_detached("queue-filler", |_| {})
            .is_ok()
        {
            queued += 1;
            assert!(queued < QUEUE_FILL_LIMIT, "command queue never filled");
        }
        release
    }

    #[test]
    fn session_cleanup_waits_for_room_in_a_full_queue() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        lock_runtime(&runtime).track_session_tokens_for_test("finished", 4);
        let scheduler = scheduler_over(&runtime);
        let release = block_worker_with_full_queue(&scheduler);

        // Ordinary work is shed while the queue is full.
        let shed = scheduler
            .execute_runtime("ordinary-work", |_| Ok(()))
            .unwrap_err();
        assert!(shed.to_string().contains("queue is full"), "{shed}");

        let cleanup_scheduler = scheduler.clone();
        let (finished, cleanup_finished) = std_mpsc::channel();
        thread::spawn(move || {
            let outcome =
                cleanup_scheduler.execute_session_cleanup("local-session-drop", |runtime| {
                    runtime
                        .drop_session_timed("finished")
                        .map(|_| ())
                        .map_err(openai_backend_error)
                });
            let _ = finished.send(outcome);
        });
        // The cleanup cannot finish while the worker is parked, so this wait
        // always times out with the fix in place. A cleanup that was shed by
        // the full queue would instead finish at once with an error.
        match cleanup_finished.recv_timeout(SHED_CLEANUP_GRACE) {
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            other => panic!("cleanup finished while the queue was full: {other:?}"),
        }
        release.send(()).unwrap();
        cleanup_finished
            .recv()
            .expect("cleanup thread reports its outcome")
            .unwrap();

        assert_eq!(
            lock_runtime(&runtime).session_stats().tracked_token_counts,
            0,
            "the session was dropped even though its cleanup met a full queue"
        );
    }

    #[test]
    fn session_cleanup_reports_a_stopped_scheduler() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        let scheduler = scheduler_over(&runtime);
        scheduler
            .shared
            .commands
            .send(SchedulerCommand::Shutdown)
            .unwrap();
        // The worker exits on shutdown, so the command is never answered.
        let error = scheduler
            .execute_session_cleanup("local-session-drop", |_| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("stopped"), "{error}");
    }
}
