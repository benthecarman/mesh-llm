//! Admission of local requests against the instance lifecycle owning a port.

use std::sync::Arc;

use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{Node, NodeRole};
use crate::runtime::{InstanceAdmissionError, InstanceLifecycleRecord, InstanceLifecycleState};

/// Give every local target in `targets` a serving lifecycle, as a managed
/// runtime instance has before it publishes its port. For fixtures whose
/// local targets are stand-in upstreams.
pub(crate) fn serve_local_targets(node: &Node, targets: &ModelTargets) {
    for target in targets.targets.values().flatten() {
        if let InferenceTarget::Local(port) = target {
            node.register_serving_local_port(*port);
        }
    }
}

fn serving_lifecycle() -> Arc<tokio::sync::Mutex<InstanceLifecycleRecord>> {
    Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        4,
    )))
}

#[tokio::test]
async fn a_local_port_without_a_lifecycle_is_not_accepting() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let port = 4100;
    assert_eq!(
        node.begin_runtime_instance_request(port).await.err(),
        Some(InstanceAdmissionError::Unregistered { port }),
        "an unowned port must not run untracked work"
    );

    let lifecycle = serving_lifecycle();
    node.register_runtime_instance_lifecycle(port, lifecycle.clone());
    let admitted = node
        .begin_runtime_instance_request(port)
        .await
        .expect("a serving lifecycle admits work");
    assert_eq!(lifecycle.lock().await.in_flight_count(), 1);
    drop(admitted);
    assert_eq!(lifecycle.lock().await.in_flight_count(), 0);

    // A target can outlive its instance's registration; unload must not leak
    // requests past drain through that stale target.
    node.unregister_runtime_instance_lifecycle(port);
    assert_eq!(
        node.begin_runtime_instance_request(port).await.err(),
        Some(InstanceAdmissionError::Unregistered { port })
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_draining_lifecycle_still_rejects_new_work() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let port = 4101;
    let lifecycle = serving_lifecycle();
    lifecycle
        .lock()
        .await
        .transition_to(InstanceLifecycleState::Draining)
        .expect("serving drains");
    node.register_runtime_instance_lifecycle(port, lifecycle);
    assert_eq!(
        node.begin_runtime_instance_request(port).await.err(),
        Some(InstanceAdmissionError::NotAccepting {
            state: InstanceLifecycleState::Draining
        })
    );
    node.endpoint.close().await;
}
