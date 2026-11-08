//! Memory-component reservations and the KV room check before each decode.
//!
//! Admission reserves `admission_tokens` per sequence, which serving bounds to
//! the prompt plus a fixed decode headroom. A sequence that decodes past that
//! headroom grows its reservation so later admissions see the room it holds.
//! Every decode is also checked against what active sequences really occupy.
//! A decode that does not fit waits for room instead of reaching a native
//! batch that would fail for every request in it; only when no work in a plan
//! can progress is one waiting sequence failed to free its room.

use std::cmp::Reverse;

use super::Scheduler;
use crate::{IterationPlan, MemoryComponent, Sequence, SequenceStatus};

impl Scheduler {
    pub(super) fn can_reserve_memory(&self, sequence: &Sequence) -> bool {
        self.config
            .memory_components
            .iter()
            .zip(self.component_used_bytes.iter().copied())
            .all(|(component, used)| {
                used.saturating_add(component.reservation_bytes(sequence.admission_tokens))
                    <= component.available_bytes()
            })
    }

    /// Reserve a newly admitted sequence's admission tokens.
    pub(super) fn reserve_memory(&mut self, sequence: &mut Sequence) {
        sequence.reserved_tokens = sequence.admission_tokens;
        for (index, component) in self.config.memory_components.iter().enumerate() {
            self.component_used_bytes[index] = self.component_used_bytes[index]
                .saturating_add(component.reservation_bytes(sequence.reserved_tokens));
        }
    }

    pub(super) fn release_memory(&mut self, sequence: &Sequence) {
        for (index, component) in self.config.memory_components.iter().enumerate() {
            self.component_used_bytes[index] = self.component_used_bytes[index]
                .saturating_sub(component.reservation_bytes(sequence.reserved_tokens));
        }
    }

    pub(super) fn remove_active(&mut self, id: &str) {
        if let Some(sequence) = self.active.remove(id) {
            self.release_memory(&sequence);
        }
    }

    pub(super) fn component_over_capacity(&self) -> bool {
        self.config
            .memory_components
            .iter()
            .zip(self.component_used_bytes.iter().copied())
            .any(|(component, used)| used > component.available_bytes())
    }

    /// What active sequences occupy now, for checking this plan's decodes.
    pub(super) fn decode_room(&self) -> DecodeRoom {
        let occupied = self
            .config
            .memory_components
            .iter()
            .map(|component| {
                self.active
                    .values()
                    .map(|sequence| component.reservation_bytes(sequence.occupied_tokens()))
                    .fold(0u64, u64::saturating_add)
            })
            .collect();
        DecodeRoom { occupied }
    }

    /// Make room for `id` to hold `context_tokens` after its next decode.
    ///
    /// Returns `false` when the tokens every active sequence occupies, with
    /// this sequence grown to `context_tokens`, would not fit a component.
    /// Otherwise any growth past the sequence's reservation is reserved.
    pub(super) fn reserve_decode_room(
        &mut self,
        room: &mut DecodeRoom,
        id: &str,
        context_tokens: usize,
    ) -> bool {
        let Some(sequence) = self.active.get_mut(id) else {
            return false;
        };
        let occupied_tokens = sequence.occupied_tokens();
        let components = &self.config.memory_components;
        let occupied_after = |(component, occupied): (&MemoryComponent, &u64)| {
            occupied
                .saturating_sub(component.reservation_bytes(occupied_tokens))
                .saturating_add(component.reservation_bytes(context_tokens))
        };
        let fits = components
            .iter()
            .zip(&room.occupied)
            .all(|entry| occupied_after(entry) <= entry.0.available_bytes());
        if !fits {
            return false;
        }
        for (component, occupied) in components.iter().zip(room.occupied.iter_mut()) {
            *occupied = occupied_after((component, &*occupied));
        }
        let reserved_tokens = sequence.reserved_tokens;
        if context_tokens > reserved_tokens {
            sequence.reserved_tokens = context_tokens;
            for (index, component) in components.iter().enumerate() {
                let growth = component
                    .reservation_bytes(context_tokens)
                    .saturating_sub(component.reservation_bytes(reserved_tokens));
                self.component_used_bytes[index] =
                    self.component_used_bytes[index].saturating_add(growth);
            }
        }
        true
    }

    /// Fail one of the sequences waiting for decode room when this plan has
    /// no work at all, so the pool cannot stay full with nothing progressing.
    ///
    /// While any other work runs, waiting decodes keep their place: a finished
    /// sequence frees room for them. The sequence failed is the one a
    /// preemption would pick: lowest priority, then the most recently admitted.
    pub(super) fn fail_stalled_decode(&mut self, waiting: &[String], plan: &mut IterationPlan) {
        if !plan.work.is_empty() {
            return;
        }
        let Some(victim) = waiting
            .iter()
            .filter_map(|id| self.active.get(id))
            .min_by_key(|sequence| {
                (
                    sequence.priority,
                    Reverse(sequence.admitted_at),
                    Reverse(sequence.enqueue_order),
                )
            })
            .map(|sequence| sequence.id.clone())
        else {
            return;
        };
        if let Some(sequence) = self.active.get_mut(&victim) {
            sequence.status = SequenceStatus::Failed;
        }
        self.remove_active(&victim);
        self.metrics.failed = self.metrics.failed.saturating_add(1);
        self.refresh_counts();
        plan.out_of_decode_room.push(victim);
    }
}

/// Bytes active sequences occupy per memory component while one plan is
/// built, kept current as its decodes are checked.
pub(super) struct DecodeRoom {
    occupied: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use crate::{
        IterationPhase, IterationPrediction, MemoryComponent, Scheduler, SchedulerConfig, Sequence,
    };

    fn kv_config(capacity_tokens: u64) -> SchedulerConfig {
        SchedulerConfig {
            max_active_sequences: 4,
            max_tokens_per_iteration: 64,
            prefill_chunk_tokens: 64,
            memory_components: vec![MemoryComponent {
                name: "kv-cells".into(),
                capacity_bytes: capacity_tokens,
                resident_bytes: 0,
                bytes_per_token: 1,
                bytes_per_sequence: 0,
            }],
            ..SchedulerConfig::default()
        }
    }

    fn kv_scheduler(capacity_tokens: u64) -> Scheduler {
        Scheduler::new(kv_config(capacity_tokens))
    }

    fn sequence(id: &str, prompt_len: usize, max_tokens: u32, admission_tokens: usize) -> Sequence {
        Sequence::new(
            id.into(),
            (0..prompt_len).map(|token| token as i32).collect(),
            max_tokens,
            None,
            1,
        )
        .with_admission_tokens(admission_tokens)
    }

    /// Complete `plan`, sampling token 7 for every row that samples.
    fn complete(scheduler: &mut Scheduler, plan: &crate::IterationPlan) {
        let predictions = plan
            .work
            .iter()
            .enumerate()
            .filter(|(_, work)| work.sample_last)
            .map(|(work_index, _)| IterationPrediction {
                work_index,
                token: 7,
            })
            .collect::<Vec<_>>();
        scheduler.complete_iteration(plan, &predictions);
    }

    /// Plan and complete one iteration.
    fn step(scheduler: &mut Scheduler) {
        let plan = scheduler.plan_iteration();
        complete(scheduler, &plan);
    }

    fn used_kv(scheduler: &Scheduler) -> u64 {
        scheduler.snapshot().component_used_bytes[0].1
    }

    #[test]
    fn decode_past_the_admission_headroom_grows_the_reservation() {
        let mut scheduler = kv_scheduler(16);
        scheduler.submit(sequence("a", 2, 8, 3)).unwrap();
        step(&mut scheduler);
        assert_eq!(used_kv(&scheduler), 3);

        // Prompt (2) plus the first sampled token fills the reservation.
        let plan = scheduler.plan_iteration();
        assert_eq!(plan.work[0].phase, IterationPhase::Decode);
        assert_eq!(used_kv(&scheduler), 3);
        complete(&mut scheduler, &plan);

        let plan = scheduler.plan_iteration();
        assert_eq!(plan.work.len(), 1);
        assert!(plan.out_of_decode_room.is_empty());
        assert_eq!(used_kv(&scheduler), 4);
    }

    fn planned(plan: &crate::IterationPlan) -> Vec<&str> {
        plan.work
            .iter()
            .map(|work| work.sequence_id.as_str())
            .collect()
    }

    #[test]
    fn a_decode_without_room_waits_for_a_finishing_sequence() {
        // Each sequence reserves its 2-token prompt plus one decode token, so
        // both are admitted, but the pool cannot hold both past that.
        let mut scheduler = kv_scheduler(7);
        scheduler.submit(sequence("a", 2, 3, 3)).unwrap();
        scheduler.submit(sequence("b", 2, 8, 3)).unwrap();
        step(&mut scheduler);

        // The first decode keeps both inside their reservations (3 + 3).
        let plan = scheduler.plan_iteration();
        assert_eq!(planned(&plan), ["a", "b"]);
        complete(&mut scheduler, &plan);

        // "a" is planned first and grows to 4 tokens; "b" would need a fourth
        // token too with only 3 left, so it waits rather than failing.
        let plan = scheduler.plan_iteration();
        assert_eq!(planned(&plan), ["a"]);
        assert!(plan.out_of_decode_room.is_empty());
        assert!(scheduler.sequence("b").is_some());
        assert_eq!(used_kv(&scheduler), 4 + 3);
        // That decode was "a"'s last token, which frees its room for "b".
        complete(&mut scheduler, &plan);
        assert!(scheduler.sequence("a").is_none());

        let plan = scheduler.plan_iteration();
        assert_eq!(planned(&plan), ["b"]);
        assert!(plan.out_of_decode_room.is_empty());
        assert_eq!(scheduler.metrics().failed, 0);
    }

    #[test]
    fn a_stalled_pool_fails_only_the_newest_waiting_sequence() {
        let mut scheduler = kv_scheduler(7);
        scheduler.submit(sequence("a", 2, 8, 3)).unwrap();
        scheduler.submit(sequence("b", 2, 8, 3)).unwrap();
        step(&mut scheduler);
        step(&mut scheduler);
        // "a" grows to 4 tokens while "b" waits at 3.
        let plan = scheduler.plan_iteration();
        assert_eq!(planned(&plan), ["a"]);
        complete(&mut scheduler, &plan);

        // Neither can grow now (5 + 3 and 4 + 4 exceed 7), so nothing would
        // progress: the most recently admitted waiting sequence is failed.
        let plan = scheduler.plan_iteration();
        assert!(plan.work.is_empty());
        assert_eq!(plan.out_of_decode_room, ["b"]);
        assert!(scheduler.sequence("b").is_none());
        assert_eq!(scheduler.metrics().failed, 1);
        assert_eq!(used_kv(&scheduler), 4);

        let plan = scheduler.plan_iteration();
        assert_eq!(planned(&plan), ["a"]);
        assert!(plan.out_of_decode_room.is_empty());
    }

    #[test]
    fn decode_room_counts_occupied_tokens_not_unused_reservations() {
        // "b" reserves far more than it occupies; "a" may decode into that
        // unused headroom rather than fail early.
        let mut scheduler = kv_scheduler(12);
        scheduler.submit(sequence("a", 2, 8, 3)).unwrap();
        scheduler.submit(sequence("b", 2, 8, 8)).unwrap();
        step(&mut scheduler);
        for _ in 0..3 {
            let plan = scheduler.plan_iteration();
            assert!(plan.out_of_decode_room.is_empty());
            assert_eq!(plan.work.len(), 2);
            complete(&mut scheduler, &plan);
        }
        // Reservations now exceed the pool, so nothing new is admitted.
        assert_eq!(used_kv(&scheduler), 5 + 8);
        scheduler.submit(sequence("c", 1, 1, 1)).unwrap();
        let plan = scheduler.plan_iteration();
        assert_eq!(plan.admitted, 0);
    }

    #[test]
    fn mixed_iterations_check_decode_room() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            mixed_prefill_decode: true,
            ..kv_config(3)
        });
        scheduler.submit(sequence("a", 2, 8, 3)).unwrap();
        let plan = scheduler.plan_iteration();
        complete(&mut scheduler, &plan);
        // Calibrate the one-row decode cost so the next plan is mixed.
        let plan = scheduler.plan_iteration();
        scheduler.observe_iteration_duration(&plan, std::time::Duration::from_millis(5));
        complete(&mut scheduler, &plan);

        let plan = scheduler.plan_iteration();
        assert!(plan.work.is_empty());
        assert_eq!(plan.out_of_decode_room, ["a"]);
        assert!(scheduler.sequence("a").is_none());
    }
}
