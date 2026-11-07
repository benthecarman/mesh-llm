mod aggregator;
mod parser_policy;

#[test]
fn native_error_line_travels_with_the_open_error() {
    super::clear_last_native_error();
    super::record_native_error_line(3, b"llama_model_loader: loaded meta data\n");
    super::record_native_error_line(
        super::NATIVE_LOG_LEVEL_ERROR,
        b"llama_model_load: error loading model: tensor data is not within the file bounds\n",
    );
    super::record_native_error_line(
        super::NATIVE_LOG_LEVEL_ERROR,
        b"llama_model_load_from_file_impl: failed to load model\n",
    );

    let error =
        super::with_last_native_error(anyhow::anyhow!("ModelError: failed to load llama model"));

    let text = format!("{error:#}");
    assert!(text.contains("failed to load llama model"), "{text}");
    assert!(text.contains("not within the file bounds"), "{text}");
    let again = super::with_last_native_error(anyhow::anyhow!("later failure"));
    assert_eq!(format!("{again:#}"), "later failure");
}

#[test]
fn native_error_line_during_thread_teardown_is_dropped() {
    // Native threads can log from thread-exit hooks after this thread's error
    // lines are destroyed. A regression panics inside a thread-local
    // destructor, which aborts the whole test binary with SIGABRT.
    static RAN_AFTER_ERROR_LINES_DESTROYED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    struct LogsOnDrop;
    impl Drop for LogsOnDrop {
        fn drop(&mut self) {
            let destroyed = super::NATIVE_ERROR_LINES.try_with(|_| ()).is_err();
            RAN_AFTER_ERROR_LINES_DESTROYED.store(destroyed, std::sync::atomic::Ordering::SeqCst);
            super::record_native_error_line(
                super::NATIVE_LOG_LEVEL_ERROR,
                b"ggml: error logged during thread teardown\n",
            );
        }
    }
    thread_local! {
        static LATE_LOGGER: LogsOnDrop = const { LogsOnDrop };
    }

    std::thread::spawn(|| {
        // Thread-local destructors run in reverse registration order, so
        // touching the error lines second destroys them before LATE_LOGGER.
        LATE_LOGGER.with(|_| {});
        super::clear_last_native_error();
    })
    .join()
    .expect("teardown thread should exit cleanly");
    assert!(
        RAN_AFTER_ERROR_LINES_DESTROYED.load(std::sync::atomic::Ordering::SeqCst),
        "the late log did not run after the error lines were destroyed; \
         this test no longer exercises thread teardown"
    );
}
