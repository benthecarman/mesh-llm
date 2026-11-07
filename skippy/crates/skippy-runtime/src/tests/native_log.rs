use super::*;

struct ResetNativeLogForwarding;

impl Drop for ResetNativeLogForwarding {
    fn drop(&mut self) {
        unregister_filtered_native_logs();
        set_filtered_native_logs_enabled(false);
    }
}

#[test]
fn write_native_log_respects_forwarding_flag() {
    let _native_log_guard = native_log_test_guard();
    let _reset = ResetNativeLogForwarding;
    unregister_filtered_native_logs();
    let mut rx = register_filtered_native_logs();

    let line =
        CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");

    set_filtered_native_logs_enabled(false);
    unsafe { write_native_log(log_level(0), line.as_ptr(), ptr::null_mut()) };
    assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));

    set_filtered_native_logs_enabled(true);
    unsafe { write_native_log(log_level(0), line.as_ptr(), ptr::null_mut()) };
    assert_eq!(
        rx.blocking_recv(),
        Some(NativeLogEvent {
            message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
            category: "tokenizer",
            params: Vec::new(),
        })
    );
}

#[test]
fn native_log_note_forwards_when_forwarding_enabled() {
    let _native_log_guard = native_log_test_guard();
    let _reset = ResetNativeLogForwarding;
    unregister_filtered_native_logs();
    let mut rx = register_filtered_native_logs();
    set_filtered_native_logs_enabled(true);

    write_native_log_note("skippy_model_open begin\nwith context");

    let expected = NativeLogEvent {
        message: "mesh-llm: skippy_model_open begin with context".to_string(),
        category: "model",
        params: Vec::new(),
    };
    let mut events = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(event) if event == expected => break,
            Ok(event) => events.push(event),
            Err(TryRecvError::Empty) => {
                panic!("expected forwarded native log note, got {events:?}")
            }
            Err(error) => panic!("expected forwarded native log note, receiver errored: {error}"),
        }
    }
}

#[test]
fn register_filtered_native_logs_replaces_receiver_cleanly() {
    let _native_log_guard = native_log_test_guard();
    let _reset = ResetNativeLogForwarding;
    unregister_filtered_native_logs();
    set_filtered_native_logs_enabled(true);

    let first = register_filtered_native_logs();
    drop(first);
    let mut second = register_filtered_native_logs();

    let line =
        CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");
    unsafe { write_native_log(log_level(0), line.as_ptr(), ptr::null_mut()) };

    assert_eq!(
        second.blocking_recv(),
        Some(NativeLogEvent {
            message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
            category: "tokenizer",
            params: Vec::new(),
        })
    );
}

#[test]
fn forwarding_recovers_after_a_contained_panic_poisons_its_locks() {
    let _native_log_guard = native_log_test_guard();
    let _reset = ResetNativeLogForwarding;
    // The native log callback contains panics, so one raised while it holds
    // the aggregator or the sender lock leaves that lock poisoned.
    let poison = |mutex_holder: fn()| {
        let _ = std::panic::catch_unwind(mutex_holder);
    };
    poison(|| {
        let _aggregator = native_log_aggregator().lock();
        panic!("contained native log panic");
    });
    poison(|| {
        let _sender = native_log_filtered_tx().lock();
        panic!("contained native log panic");
    });
    assert!(native_log_aggregator().is_poisoned());
    assert!(native_log_filtered_tx().is_poisoned());

    let mut rx = register_filtered_native_logs();
    set_filtered_native_logs_enabled(true);
    let line =
        CString::new("init_tokenizer: initializing tokenizer for type 2\n").expect("cstring");
    // SAFETY: The CString stays alive for the synchronous callback invocation.
    unsafe { write_native_log(log_level(0), line.as_ptr(), ptr::null_mut()) };

    assert_eq!(
        rx.try_recv(),
        Ok(NativeLogEvent {
            message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
            category: "tokenizer",
            params: Vec::new(),
        })
    );
    assert!(measured_native_buffers().is_some());
    assert!(!native_log_aggregator().is_poisoned());
    assert!(!native_log_filtered_tx().is_poisoned());
}

#[cfg(test)]
fn log_level(level: i32) -> skippy_ffi::bindings::llama::ggml_log_level {
    // Test helper: the generated enum mirrors the C values verbatim.
    unsafe { std::mem::transmute(level) }
}
