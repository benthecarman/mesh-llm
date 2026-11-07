//! Admission of a direct local load's memory plan. A requested lane count is
//! bounded, an explicit context or lane count must fit the device, and the
//! capacity ledger holds the plan's resident, KV, draft and compute bytes, not
//! only the weights the load was admitted with, before any native allocation.

use super::capacity::RuntimeCapacityReservation;
use super::context_planning::{RuntimeResourcePlan, RuntimeResourcePlanBreakdown};
use super::split_planning::format_gb;
use anyhow::{Context, Result};

/// llama.cpp's sequence-id ceiling (`LLAMA_MAX_SEQ`).
const LLAMA_MAX_SEQ: usize = 256;
/// Sequence ids one lane occupies: the lane itself, its auxiliary
/// recurrent/speculative sequence, and room for a resident prefix, which the
/// native cache layout places after `lane_count * 2`.
const SEQUENCE_IDS_PER_LANE: usize = 3;
/// Most lanes a local load accepts, the same ceiling the split topology
/// planner applies to an explicit lane count.
pub(super) const MAX_PARALLEL_LANES: usize = LLAMA_MAX_SEQ / SEQUENCE_IDS_PER_LANE;

/// Rejects a requested lane count above [`MAX_PARALLEL_LANES`].
pub(super) fn ensure_parallel_within_cap(parallel_override: Option<usize>) -> Result<()> {
    if let Some(parallel) = parallel_override {
        anyhow::ensure!(
            parallel <= MAX_PARALLEL_LANES,
            "parallel = {parallel} exceeds the maximum of {MAX_PARALLEL_LANES} lanes per model"
        );
    }
    Ok(())
}

/// Checks the plan fits, then grows `reservation` to the plan's footprint so
/// later loads see this load's KV cache, draft and compute buffers.
pub(super) fn admit_local_plan(
    plan: &RuntimeResourcePlan,
    reservation: Option<&RuntimeCapacityReservation>,
) -> Result<()> {
    ensure_plan_fits(plan)?;
    if let (Some(reservation), Some(footprint)) = (reservation, planned_footprint_bytes(plan)) {
        reservation.grow_to(footprint).with_context(|| {
            plan.breakdown.map_or_else(String::new, |breakdown| {
                format!(
                    "planned context {} x {} lanes does not fit beside the models already loaded",
                    breakdown.context_length, breakdown.slots
                )
            })
        })?;
    }
    Ok(())
}

/// A measured footprint that cannot fit the minimum context always fails. An
/// explicit context size or lane count is not shrunk to fit, so it must fit as
/// requested.
fn ensure_plan_fits(plan: &RuntimeResourcePlan) -> Result<()> {
    let Some(breakdown) = plan.breakdown else {
        return Ok(());
    };
    anyhow::ensure!(
        breakdown.measured_fit != Some(false),
        "measured native buffers leave no capacity for the minimum context under the current model configuration"
    );
    let explicit = !breakdown.context_auto || !breakdown.slots_auto;
    anyhow::ensure!(
        !explicit || plan.fits(),
        "{}",
        explicit_plan_overrun_message(&breakdown)
    );
    Ok(())
}

fn explicit_plan_overrun_message(breakdown: &RuntimeResourcePlanBreakdown) -> String {
    format!(
        "requested context {} x {} lanes needs {} of KV cache and {} resident, but this node has {} with {} available for KV; lower ctx_size or parallel",
        breakdown.context_length,
        breakdown.slots,
        format_gb(breakdown.planned_kv_bytes),
        format_gb(resident_bytes(breakdown)),
        format_gb(breakdown.vram_bytes),
        format_gb(breakdown.kv_budget_bytes),
    )
}

/// Weights, projector and draft, which stay resident beside the KV cache.
fn resident_bytes(breakdown: &RuntimeResourcePlanBreakdown) -> u64 {
    breakdown
        .model_bytes
        .saturating_add(breakdown.projector_bytes)
        .saturating_add(breakdown.speculative_bytes)
}

/// Bytes the plan commits: resident bytes, the planned KV pool (which includes
/// a draft context's per-token KV), and the compute buffers the planner holds
/// outside the KV budget. A plan that fits never exceeds its device.
fn planned_footprint_bytes(plan: &RuntimeResourcePlan) -> Option<u64> {
    plan.breakdown.map(|breakdown| {
        resident_bytes(&breakdown)
            .saturating_add(breakdown.planned_kv_bytes)
            .saturating_add(breakdown.compute_charge_bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::capacity::{
        RuntimeCapacityLedger, RuntimeCapacityPool, RuntimeCapacityRequest,
    };
    use crate::runtime::context_planning::{
        RuntimeResourcePlanInput, RuntimeResourcePlanningProfile, plan_runtime_resources,
    };

    const GB: u64 = 1_000_000_000;

    fn llama_8b_meta() -> crate::models::gguf::GgufCompactMeta {
        crate::models::gguf::GgufCompactMeta {
            architecture: "llama".to_string(),
            context_length: 131_072,
            embedding_size: 4096,
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        }
    }

    fn plan(
        meta: &crate::models::gguf::GgufCompactMeta,
        vram_bytes: u64,
        ctx_size_override: Option<u32>,
        parallel_override: Option<usize>,
    ) -> RuntimeResourcePlan {
        plan_runtime_resources(RuntimeResourcePlanInput {
            ctx_size_override,
            parallel_override,
            model_bytes: 5 * GB,
            projector_bytes: 0,
            vram_bytes,
            metadata: Some(meta),
            kv_cache_quant: crate::models::gguf::GgufKvCacheQuant::F16,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        })
    }

    fn reserve(ledger: &RuntimeCapacityLedger, id: &str, bytes: u64) -> RuntimeCapacityReservation {
        reserve_in(ledger, id, 24 * GB, bytes)
    }

    fn reserve_in(
        ledger: &RuntimeCapacityLedger,
        id: &str,
        capacity_bytes: u64,
        bytes: u64,
    ) -> RuntimeCapacityReservation {
        ledger
            .reserve(RuntimeCapacityRequest {
                instance_id: id.to_string(),
                model_name: id.to_string(),
                pool: RuntimeCapacityPool::Node,
                capacity_bytes,
                required_bytes: bytes,
            })
            .expect("admission should fit")
    }

    #[test]
    fn parallel_above_the_lane_cap_is_rejected() {
        ensure_parallel_within_cap(None).expect("auto lanes are always allowed");
        ensure_parallel_within_cap(Some(MAX_PARALLEL_LANES)).expect("the cap itself is allowed");
        let error = ensure_parallel_within_cap(Some(MAX_PARALLEL_LANES + 1))
            .expect_err("a lane count past the sequence-id ceiling must fail");
        assert!(error.to_string().contains("parallel"));
    }

    #[test]
    fn an_explicit_context_that_overruns_the_device_is_rejected() {
        let meta = llama_8b_meta();
        // 128k x 4 lanes of F16 KV is 64 GiB, far past a 24 GB device.
        let oversized = plan(&meta, 24 * GB, Some(131_072), None);
        assert!(!oversized.fits());
        let error = admit_local_plan(&oversized, None)
            .expect_err("an explicit context must fit as requested");
        assert!(error.to_string().contains("lower ctx_size or parallel"));

        let fitting = plan(&meta, 24 * GB, Some(8192), None);
        admit_local_plan(&fitting, None).expect("a context that fits is admitted");
    }

    #[test]
    fn an_explicit_lane_count_whose_minimum_context_overruns_is_rejected() {
        let meta = llama_8b_meta();
        let oversized = plan(&meta, 6 * GB, None, Some(MAX_PARALLEL_LANES));
        assert!(!oversized.fits());
        admit_local_plan(&oversized, None).expect_err("an explicit lane count must fit");
    }

    #[test]
    fn an_auto_plan_at_its_minimum_context_is_not_rejected_by_the_override_check() {
        let meta = llama_8b_meta();
        // 0.3 GB free leaves 0.255 GB for KV; 512 x 4 lanes of F16 needs 0.27 GB.
        let tight = plan(&meta, 5_300_000_000, None, None);
        assert!(!tight.fits());
        admit_local_plan(&tight, None).expect("auto plans keep their existing admission");
    }

    #[test]
    fn an_auto_plan_that_overruns_its_device_fails_closed_at_the_ledger() {
        let meta = llama_8b_meta();
        let ledger = RuntimeCapacityLedger::default();
        let reservation = reserve_in(&ledger, "runtime-1", 5_300_000_000, 5 * GB);
        // Weights, 0.27 GB of KV at the minimum context and 0.045 GB of
        // compute exceed the 5.3 GB device.
        let tight = plan(&meta, reservation.capacity_budget_bytes(), None, None);
        let error = admit_local_plan(&tight, Some(&reservation))
            .expect_err("the ledger must not hold more than the device");
        assert!(format!("{error:#}").contains("planned context 512 x 4 lanes"));
        assert_eq!(ledger.used_bytes(&RuntimeCapacityPool::Node), 5 * GB);
    }

    #[test]
    fn admission_grows_the_ledger_to_the_planned_kv_cache() {
        let meta = llama_8b_meta();
        let ledger = RuntimeCapacityLedger::default();
        let first = reserve(&ledger, "runtime-1", 5_500_000_000);
        let first_plan = plan(&meta, first.capacity_budget_bytes(), None, None);
        let first_footprint = planned_footprint_bytes(&first_plan).expect("breakdown");
        let breakdown = first_plan.breakdown.expect("breakdown");
        assert!(breakdown.planned_kv_bytes > 0);
        assert_eq!(
            first_footprint,
            breakdown.model_bytes + breakdown.planned_kv_bytes + breakdown.compute_charge_bytes,
            "the footprint holds the weights, the KV pool and the compute buffers"
        );

        admit_local_plan(&first_plan, Some(&first)).expect("the plan fits its budget");

        assert_eq!(
            ledger.used_bytes(&RuntimeCapacityPool::Node),
            first_footprint
        );
        let second = reserve(&ledger, "runtime-2", 5_500_000_000);
        assert_eq!(
            second.capacity_budget_bytes(),
            24 * GB - first_footprint,
            "the next load plans beside the first load's KV cache and compute, not just its weights"
        );
    }
}
