//! Undo the node state a runtime load publishes before its launch succeeds.
//!
//! A load publishes its serving assignment and source policy before the native
//! launch, so peers and stage requests see them while a large model opens. When
//! the launch fails, that state must go back to what it was before the load,
//! without touching state another live instance of the same model still owns.
use super::*;
use crate::inference::skippy;

/// Source policies a runtime load has published.
///
/// Dropping this without `commit` puts back the policy entry each published
/// `(model, profile)` pair had before the load. This covers every failure path,
/// including early returns and a dropped load future. Pairs owned by a live
/// runtime or managed model are never rolled back.
pub(super) struct LoadSourcePolicyRollback {
    profile: String,
    /// Model names with a live owner for `profile` when the load began. The
    /// control loop holds the lifecycle maps exclusively for the whole load,
    /// so this snapshot stays current until the load returns.
    owned_by_live_models: Vec<String>,
    /// Published model IDs with the entry found before the first publish.
    published: Vec<(String, Option<bool>)>,
    committed: bool,
}

impl LoadSourcePolicyRollback {
    pub(super) fn for_load(ctx: &RunAutoRuntimeLoopContext<'_>, profile: &str) -> Self {
        let runtime_owners = ctx
            .runtime_models
            .values()
            .map(|entry| (entry.model_name.as_str(), entry.profile.as_str()));
        let managed_owners = ctx
            .managed_models
            .values()
            .map(|controller| (controller.model_name.as_str(), controller.profile.as_str()));
        Self::new(profile, runtime_owners.chain(managed_owners))
    }

    fn new<'a>(profile: &str, live_owners: impl Iterator<Item = (&'a str, &'a str)>) -> Self {
        Self {
            profile: profile.to_string(),
            owned_by_live_models: live_owners
                .filter(|(_, owner_profile)| *owner_profile == profile)
                .map(|(model, _)| model.to_string())
                .collect(),
            published: Vec::new(),
            committed: false,
        }
    }

    /// Register `model_id`'s policy for this load's profile.
    pub(super) fn publish(&mut self, model_id: &str, local_source_required: bool) {
        let tracked = self.published.iter().any(|(id, _)| id == model_id)
            || self.owned_by_live_models.iter().any(|id| id == model_id);
        if !tracked && !model_id.is_empty() {
            let prior = skippy::registered_local_source_policy(model_id, &self.profile);
            self.published.push((model_id.to_string(), prior));
        }
        skippy::register_local_source_policy(model_id, &self.profile, local_source_required);
    }

    /// Keep the published policies: the load now owns them.
    pub(super) fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for LoadSourcePolicyRollback {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for (model_id, prior) in self.published.drain(..).rev() {
            match prior {
                Some(required) => {
                    skippy::register_local_source_policy(&model_id, &self.profile, required);
                }
                None => skippy::unregister_local_source_policy(&model_id, &self.profile),
            }
        }
    }
}

/// Withdraw a failed load's serving assignment only when no live instance of
/// the model remains. A second load of a model that is already serving must
/// not stop the node advertising the first copy.
pub(super) async fn withdraw_failed_load_serving(
    ctx: &RunAutoRuntimeLoopContext<'_>,
    model_name: &str,
) {
    withdraw_failed_load_serving_assignment(
        ctx.node,
        ctx.runtime_instance_registry,
        ctx.managed_models
            .values()
            .map(|controller| (controller.model_name.as_str(), &controller.lifecycle)),
        model_name,
    )
    .await;
}

type ManagedLifecycle = Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>;

async fn withdraw_failed_load_serving_assignment<'a>(
    node: &mesh::Node,
    runtime_instance_registry: &RuntimeInstanceRegistry,
    managed_models: impl Iterator<Item = (&'a str, &'a ManagedLifecycle)>,
    model_name: &str,
) {
    if live_managed_model(managed_models, model_name).await
        || runtime_registry_has_model(runtime_instance_registry, model_name).await
    {
        return;
    }
    remove_serving_assignment(node, model_name).await;
}

/// A managed model that is still warming owns its assignment before it
/// appears in the instance registry. One that failed or stopped stays in the
/// managed map until unloaded, but no longer serves anything.
async fn live_managed_model<'a>(
    managed_models: impl Iterator<Item = (&'a str, &'a ManagedLifecycle)>,
    model_name: &str,
) -> bool {
    for (_, lifecycle) in managed_models.filter(|(managed, _)| *managed == model_name) {
        if !lifecycle.lock().await.state().is_terminal() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy_test_model(name: &str) -> String {
        format!("load-rollback-{name}-{}", std::process::id())
    }

    #[test]
    fn failed_load_withdraws_policies_it_introduced() {
        let alias = policy_test_model("alias");
        let config_id = policy_test_model("config-id");

        let mut rollback = LoadSourcePolicyRollback::new("strict", std::iter::empty());
        rollback.publish(&alias, true);
        rollback.publish(&config_id, true);
        assert!(skippy::local_source_required_for_model(&alias, None));
        assert!(skippy::local_source_required_for_model(&config_id, None));
        drop(rollback);

        assert_eq!(
            skippy::registered_local_source_policy(&alias, "strict"),
            None
        );
        assert_eq!(
            skippy::registered_local_source_policy(&config_id, "strict"),
            None
        );
        assert!(!skippy::local_source_required_for_model(&config_id, None));
    }

    #[test]
    fn failed_load_restores_a_policy_registered_before_it() {
        let model = policy_test_model("preregistered");
        skippy::register_local_source_policy(&model, "strict", true);

        let mut rollback = LoadSourcePolicyRollback::new("strict", std::iter::empty());
        rollback.publish(&model, false);
        drop(rollback);

        assert_eq!(
            skippy::registered_local_source_policy(&model, "strict"),
            Some(true)
        );
        skippy::unregister_local_source_policy(&model, "strict");
    }

    #[test]
    fn failed_load_keeps_a_policy_a_live_instance_owns() {
        let model = policy_test_model("live-owner");

        let mut rollback =
            LoadSourcePolicyRollback::new("strict", [(model.as_str(), "strict")].into_iter());
        rollback.publish(&model, true);
        drop(rollback);

        assert_eq!(
            skippy::registered_local_source_policy(&model, "strict"),
            Some(true)
        );
        skippy::unregister_local_source_policy(&model, "strict");
    }

    #[test]
    fn live_owner_of_another_profile_does_not_block_rollback() {
        let model = policy_test_model("other-profile");

        let mut rollback =
            LoadSourcePolicyRollback::new("strict", [(model.as_str(), "fallback")].into_iter());
        rollback.publish(&model, true);
        drop(rollback);

        assert_eq!(
            skippy::registered_local_source_policy(&model, "strict"),
            None
        );
    }

    #[test]
    fn committed_load_keeps_its_policies() {
        let model = policy_test_model("committed");

        let mut rollback = LoadSourcePolicyRollback::new("strict", std::iter::empty());
        rollback.publish(&model, true);
        rollback.commit();

        assert_eq!(
            skippy::registered_local_source_policy(&model, "strict"),
            Some(true)
        );
        skippy::unregister_local_source_policy(&model, "strict");
    }

    fn new_registry() -> RuntimeInstanceRegistry {
        Arc::new(tokio::sync::Mutex::new(HashMap::new()))
    }

    async fn register_serving_instance(
        registry: &RuntimeInstanceRegistry,
        node: &mesh::Node,
        model: &str,
        instance_id: &str,
    ) {
        register_runtime_instance(
            registry,
            node,
            RuntimeModelRegistration {
                primary_model_name: "Primary",
                model_name: model,
                instance_id,
                context_length: Some(4096),
                capabilities: models::ModelCapabilities::default(),
                workload_class: mesh::ModelWorkloadClass::CausalGeneration,
            },
        )
        .await;
    }

    async fn serves(node: &mesh::Node, model: &str) -> bool {
        node.serving_models().await.iter().any(|m| m == model)
    }

    fn managed_lifecycle(state: InstanceLifecycleState) -> ManagedLifecycle {
        Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            state, 50,
        )))
    }

    #[tokio::test]
    async fn failed_duplicate_load_keeps_serving_the_live_instance() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let registry = new_registry();
        register_serving_instance(&registry, &node, "Duplicate", "runtime-1").await;
        // The second load re-asserts the assignment before its launch.
        add_serving_assignment(&node, "Primary", "Duplicate").await;

        withdraw_failed_load_serving_assignment(&node, &registry, std::iter::empty(), "Duplicate")
            .await;

        assert!(serves(&node, "Duplicate").await);
    }

    #[tokio::test]
    async fn failed_load_keeps_serving_a_warming_managed_model() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let registry = new_registry();
        add_serving_assignment(&node, "Primary", "Managed").await;
        let warming = managed_lifecycle(InstanceLifecycleState::Warming);

        withdraw_failed_load_serving_assignment(
            &node,
            &registry,
            [("Managed", &warming)].into_iter(),
            "Managed",
        )
        .await;

        assert!(serves(&node, "Managed").await);
    }

    #[tokio::test]
    async fn failed_managed_model_does_not_keep_a_failed_load_advertised() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let registry = new_registry();
        add_serving_assignment(&node, "Primary", "Managed").await;
        let failed = managed_lifecycle(InstanceLifecycleState::Failed);
        let stopped = managed_lifecycle(InstanceLifecycleState::Stopped);

        withdraw_failed_load_serving_assignment(
            &node,
            &registry,
            [("Managed", &failed), ("Managed", &stopped)].into_iter(),
            "Managed",
        )
        .await;

        assert!(!serves(&node, "Managed").await);
    }

    #[tokio::test]
    async fn failed_only_load_withdraws_its_serving_assignment() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node should initialize");
        let registry = new_registry();
        register_serving_instance(&registry, &node, "Other", "runtime-1").await;
        add_serving_assignment(&node, "Primary", "Lonely").await;
        let other = managed_lifecycle(InstanceLifecycleState::Serving);

        withdraw_failed_load_serving_assignment(
            &node,
            &registry,
            [("Other", &other)].into_iter(),
            "Lonely",
        )
        .await;

        assert!(!serves(&node, "Lonely").await);
        assert!(serves(&node, "Other").await);
    }
}
