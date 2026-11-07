//! Retirement of the host's per-generation commit tracking on every path that
//! ends a generation, including terminal commands the queue refuses.

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use skippy_serving::frontend::{
    GenerationAbort, GenerationCommit, GenerationLifecycleIngress, GenerationLifecycleObservation,
    GenerationReceipt, GenerationStart, GenerationTermination, LinearProposalDiscardReason,
};

use super::ON_TIME;
use crate::{
    NativeLifecycleIngress,
    plugin_dispatch::{
        PLUGIN_RETIRE_RESERVE, PluginCommand, PluginCommandQueue, PluginCommandQueueError,
        PluginDriver, finish_generation,
    },
    test_support::{CallbackGate, fake_active_with_commit_gate, fake_active_with_events},
};

fn begin_generation(request_id: u64, session_id: u64) -> GenerationLifecycleObservation {
    GenerationLifecycleObservation::Started(GenerationStart {
        request_id,
        session_id,
        agent_session_id: None,
        prompt_token_ids: Arc::from([3]),
        frontend_request_id: None,
    })
}

/// Fills `queue`'s ordinary capacity with aborts for a generation no test
/// begins, so they cannot change tracked state.
fn fill_queue(queue: &PluginCommandQueue) {
    let mut queued = 0_usize;
    while queue
        .try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 9,
            session_id: 9,
        }))
        .is_ok()
    {
        queued += 1;
        assert!(queued < 10_000, "queue never reported full");
    }
}

fn wait_until_untracked(driver: &PluginDriver) {
    let deadline = Instant::now() + ON_TIME;
    while driver.active.tracked_generation_count() != 0 {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the generation state to be retired"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_finish_refused_by_a_full_queue_still_retires_its_generation() {
    let gate = CallbackGate::new();
    let release = gate.release_on_drop();
    let (active, _) = fake_active_with_commit_gate(Arc::clone(&gate));
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress.try_submit(begin_generation(1, 2)).unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    // The worker holds the commit callback, so ordinary traffic fills the
    // queue behind it.
    gate.wait_until_entered();
    fill_queue(&driver.queue);

    let refused = ingress.try_submit(GenerationLifecycleObservation::Completed(
        GenerationReceipt::test_fixture(1, 2, GenerationTermination::MaxTokens),
    ));
    assert!(refused.is_err(), "a full queue must refuse the finish");

    drop(release);
    wait_until_untracked(&driver);
}

#[test]
fn an_abort_refused_by_a_full_queue_still_retires_its_generation() {
    let gate = CallbackGate::new();
    let release = gate.release_on_drop();
    let (active, _) = fake_active_with_commit_gate(Arc::clone(&gate));
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress.try_submit(begin_generation(1, 2)).unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    gate.wait_until_entered();
    fill_queue(&driver.queue);

    ingress.receipt_unavailable(&GenerationAbort {
        request_id: 1,
        session_id: 2,
    });

    drop(release);
    wait_until_untracked(&driver);
}

#[test]
fn a_failed_finish_fence_still_retires_its_generation() {
    let (active, _) = fake_active_with_events(Duration::ZERO);
    active
        .begin(&GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        })
        .unwrap();
    let stopped_passive_queue = PluginCommandQueue::new();
    stopped_passive_queue.close();

    let error = finish_generation(
        &active,
        &stopped_passive_queue,
        &GenerationReceipt::test_fixture(1, 2, GenerationTermination::MaxTokens),
    )
    .unwrap_err();

    assert!(error.to_string().contains("passive worker stopped"));
    assert_eq!(active.tracked_generation_count(), 0);
}

#[test]
fn a_retirement_runs_after_the_generation_begin_queued_before_it() {
    let gate = CallbackGate::new();
    let release = gate.release_on_drop();
    let (active, _) = fake_active_with_commit_gate(Arc::clone(&gate));
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    // Generation A holds the worker in its commit callback.
    ingress.try_submit(begin_generation(1, 2)).unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    gate.wait_until_entered();
    // Generation B's begin is still queued when its finish is refused, so
    // the retirement must run after that begin rather than before it.
    ingress.try_submit(begin_generation(3, 4)).unwrap();
    fill_queue(&driver.queue);
    let refused = ingress.try_submit(GenerationLifecycleObservation::Completed(
        GenerationReceipt::test_fixture(3, 4, GenerationTermination::MaxTokens),
    ));
    assert!(refused.is_err(), "a full queue must refuse the finish");

    drop(release);
    // Generation C is queued after B's retirement, so once C is tracked the
    // retirement has run.
    let deadline = Instant::now() + ON_TIME;
    while ingress.try_submit(begin_generation(5, 6)).is_err() {
        assert!(Instant::now() < deadline, "the queue never drained");
        thread::sleep(Duration::from_millis(1));
    }
    while !driver.active.tracks_generation(5, 6) {
        assert!(Instant::now() < deadline, "generation C never began");
        thread::sleep(Duration::from_millis(1));
    }

    assert!(!driver.active.tracks_generation(3, 4));
    assert!(driver.active.tracks_generation(1, 2));
}

#[test]
fn queued_retirements_leave_ordinary_and_terminal_capacity_alone() {
    let queue = PluginCommandQueue::new();
    fill_queue(&queue);
    let mut retirements = 0_u64;
    while queue.try_enqueue_retirement(1, retirements).is_ok() {
        retirements += 1;
        assert!(retirements < 10_000, "retirements were never bounded");
    }
    assert_eq!(retirements, PLUGIN_RETIRE_RESERVE as u64);

    assert!(matches!(
        queue.try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 9,
            session_id: 9,
        })),
        Err(PluginCommandQueueError::Full)
    ));
    assert!(
        queue
            .try_enqueue_terminal(PluginCommand::Discard(
                vec![1],
                LinearProposalDiscardReason::PositionMismatch,
            ))
            .is_ok(),
        "retirements must not consume the terminal reserve"
    );

    // Draining a retirement frees retirement room only.
    while let Some(queued) = queue.next() {
        if matches!(queued.command, PluginCommand::Retire { .. }) {
            break;
        }
    }
    assert!(queue.try_enqueue_retirement(2, 0).is_ok());
}
