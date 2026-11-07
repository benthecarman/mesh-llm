use crate::models::{ModelCapabilities, ModelTopology};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelDemand {
    pub last_active: u64,
    pub request_count: u64,
}

pub const DEMAND_TTL_SECS: u64 = 86400;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Catalog,
    HuggingFace,
    LocalGguf,
    DirectUrl,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ServedModelIdentity {
    pub model_name: String,
    pub is_primary: bool,
    pub source_kind: ModelSourceKind,
    pub canonical_ref: Option<String>,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub artifact: Option<String>,
    pub local_file_name: Option<String>,
    /// Hash of a reference STRING identifying the model source (repo,
    /// revision, file) -- `None` for a bare local path, where there is no
    /// such string. NOT a hash of the served weight bytes; see
    /// [`Self::weights_digest`] for that fact.
    pub identity_hash: Option<String>,
    /// SHA-256 of the served GGUF's file BYTES, lowercase hex -- computed at
    /// load time from the file actually opened for serving. `None` when the
    /// file could not be read, or when no load-time hash has been computed
    /// for this model yet -- an absent fact, never a fabricated digest.
    /// Distinct from `identity_hash`: a name/reference-string hash and a
    /// content hash are different facts about the model, and this field
    /// never replaces the other.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServedModelDescriptor {
    pub identity: ServedModelIdentity,
    /// Every required plugin mesh channel must be available at the client.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<ChannelRequirement>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub capabilities_known: bool,
    pub capabilities: ModelCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<ModelTopology>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ServedModelMetadata>,
}

/// Opaque protocol terms belonging to a plugin mesh channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ChannelRequirement {
    pub channel: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub terms: Vec<u8>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// The primary inference contract exposed by a loaded model runtime.
///
/// This is deliberately separate from multimodal capabilities: a causal model
/// can accept image or audio inputs, while an embedding or reranking model has
/// a different output contract even when its input is text-only.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelWorkloadClass {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
    SpeechSynthesis,
    /// Typed System One decisions (`POST /systemone`) with no text generation,
    /// such as a Laya encoder + decision head.
    Decision,
    /// Speech-to-text (`/v1/audio/transcriptions` and `/translations`) with no
    /// chat generation, such as a Whisper model.
    SpeechRecognition,
    /// Document OCR with page text, layout, and reading order.
    DocumentOcr,
    /// Joint image/text encoder similarity and label ranking.
    ImageTextSimilarity,
    /// Dense relative depth estimation from an image.
    DepthEstimation,
    /// Explicit metadata from a newer peer that this node cannot interpret.
    /// Unlike absent legacy metadata, this never authorizes inference.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServedModelMetadata {
    /// `None` is the mixed-version representation for nodes that predate
    /// workload advertisement. Callers must not treat it as evidence that a
    /// non-chat endpoint is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload_class: Option<ModelWorkloadClass>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_count_b: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_expert_count: Option<u32>,
}

impl ServedModelMetadata {
    /// Report whether all optional metadata is absent; a workload-only record is nonempty.
    pub fn is_empty(&self) -> bool {
        self.workload_class.is_none()
            && self.architecture.is_none()
            && self.parameter_size.is_none()
            && self.parameter_count_b.is_none()
            && self.quant.is_none()
            && self.native_context_length.is_none()
            && self.tokenizer.is_none()
            && self.layer_count.is_none()
            && self.embedding_size.is_none()
            && self.head_count.is_none()
            && self.kv_head_count.is_none()
            && self.expert_count.is_none()
            && self.active_expert_count.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelRuntimeDescriptor {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    pub ready: bool,
}

impl ModelRuntimeDescriptor {
    pub fn advertised_context_length(&self) -> Option<u32> {
        self.ready.then_some(self.context_length).flatten()
    }
}

/// Itemized view of the capacity a node advertises. The announcement's
/// `vram_bytes` stays the placement budget; this block explains how that
/// number was derived. Invariant: `total_bytes == reserved_bytes +
/// platform_reserve_bytes + configured_reserve_bytes + usable_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdvertisedMemory {
    /// Enumerated accelerator memory: the sum of device VRAM, or the unified
    /// working set on SoCs.
    pub total_bytes: u64,
    /// Driver/runtime reserved or unavailable bytes when the platform reports
    /// a true value.
    pub reserved_bytes: u64,
    /// Withheld by platform policy before the owner configures anything: the
    /// share of unified memory the platform keeps for the system (the Tegra
    /// collector budgets 90% of physical RAM). Zero for discrete GPUs, whose
    /// budget is the device memory minus the driver reserve, and zero on
    /// Metal while the survey reports the working set as the device memory.
    pub platform_reserve_bytes: u64,
    /// Withheld by the node owner: the effective safety margin plus whatever
    /// a `max_vram_gb` cap leaves out.
    pub configured_reserve_bytes: u64,
    /// What remains for mesh placement after both reserves.
    pub usable_bytes: u64,
    /// Total system RAM when the platform reports it.
    pub system_ram_bytes: Option<u64>,
    /// Portion of the local fit budget backed by system RAM, after any
    /// `max_vram_gb` cap. Never advertised as accelerator capacity.
    pub ram_offload_bytes: u64,
}

/// Most demand entries a merge from gossip will grow the local map to.
/// Demand is re-gossiped mesh-wide, so without a ceiling one peer could plant
/// any number of model names in every node's map.
pub const MAX_DEMAND_ENTRIES: usize = 512;

/// Longest model reference accepted from a peer's demand map, in bytes.
pub const MAX_DEMAND_MODEL_NAME_BYTES: usize = 512;

/// How far ahead of the local clock a peer's `last_active` may be. Later
/// timestamps are clamped to this bound; unclamped, they would keep the entry
/// inside [`DEMAND_TTL_SECS`] indefinitely.
pub const MAX_DEMAND_CLOCK_SKEW_SECS: u64 = 300;

/// Merge a peer's demand map into ours at the current wall-clock time, with
/// no entry protected from eviction.
pub fn merge_demand(
    ours: &mut HashMap<String, ModelDemand>,
    theirs: &HashMap<String, ModelDemand>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    merge_demand_at(ours, theirs, now, |_| false);
}

/// Merge a peer's demand map into ours, treating `now` as the current Unix
/// time in seconds.
///
/// Entries with an empty or oversized model name are dropped, and a
/// `last_active` further ahead than [`MAX_DEMAND_CLOCK_SKEW_SECS`] is clamped
/// to that bound. Known models are always updated. New models are added
/// freshest first; once the map holds [`MAX_DEMAND_ENTRIES`], a new model
/// replaces the stalest entry that `is_protected` does not hold, and only if
/// the new model is fresher. A full map therefore recovers as soon as planted
/// entries stop being refreshed, instead of after [`DEMAND_TTL_SECS`].
pub fn merge_demand_at(
    ours: &mut HashMap<String, ModelDemand>,
    theirs: &HashMap<String, ModelDemand>,
    now: u64,
    is_protected: impl Fn(&str) -> bool,
) {
    let latest_allowed = now.saturating_add(MAX_DEMAND_CLOCK_SKEW_SECS);
    let mut new_models = Vec::new();
    for (model, their_demand) in theirs {
        if model.is_empty() || model.len() > MAX_DEMAND_MODEL_NAME_BYTES {
            continue;
        }
        let last_active = their_demand.last_active.min(latest_allowed);
        match ours.get_mut(model) {
            Some(entry) => {
                entry.last_active = entry.last_active.max(last_active);
                entry.request_count = entry.request_count.max(their_demand.request_count);
            }
            None => new_models.push((
                model,
                ModelDemand {
                    last_active,
                    request_count: their_demand.request_count,
                },
            )),
        }
    }
    admit_new_demand(ours, new_models, is_protected);
}

/// Insert models not yet in `ours`, freshest first, within
/// [`MAX_DEMAND_ENTRIES`]; see [`merge_demand_at`].
fn admit_new_demand(
    ours: &mut HashMap<String, ModelDemand>,
    mut new_models: Vec<(&String, ModelDemand)>,
    is_protected: impl Fn(&str) -> bool,
) {
    let mut room = MAX_DEMAND_ENTRIES.saturating_sub(ours.len());
    let mut stalest_first = Vec::new();
    if new_models.len() > room {
        new_models.sort_unstable_by(|(a_model, a), (b_model, b)| {
            b.last_active
                .cmp(&a.last_active)
                .then_with(|| a_model.cmp(b_model))
        });
        stalest_first = ours
            .iter()
            .filter(|(model, _)| !is_protected(model))
            .map(|(model, demand)| (model.clone(), demand.last_active))
            .collect();
        stalest_first.sort_unstable_by(|(a_model, a), (b_model, b)| {
            a.cmp(b).then_with(|| a_model.cmp(b_model))
        });
    }
    let mut evictable = stalest_first.into_iter().peekable();
    for (model, demand) in new_models {
        if room > 0 {
            room -= 1;
        } else if let Some((evicted, _)) =
            evictable.next_if(|(_, last_active)| *last_active < demand.last_active)
        {
            ours.remove(&evicted);
        } else {
            // New models are sorted freshest first: no later one can evict.
            break;
        }
        ours.insert(model.clone(), demand);
    }
}

pub fn infer_served_model_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
    primary_model_path: Option<&std::path::Path>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source
        .and_then(identity_from_model_source)
        .or_else(|| {
            primary_model_path.and_then(|path| identity_from_local_path(primary_model_name, path))
        });
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if idx == 0 || model_name == primary_model_name {
                let mut id = primary.clone().unwrap_or_default();
                id.model_name = model_name.clone();
                id.is_primary = true;
                if id.local_file_name.is_none() {
                    id.local_file_name = Some(format!("{model_name}.gguf"));
                }
                id
            } else {
                ServedModelIdentity {
                    model_name: model_name.clone(),
                    is_primary: false,
                    source_kind: ModelSourceKind::Unknown,
                    local_file_name: Some(format!("{model_name}.gguf")),
                    ..Default::default()
                }
            };
            ServedModelDescriptor {
                requires: Vec::new(),
                identity,
                capabilities_known: false,
                capabilities: ModelCapabilities::default(),
                topology: None,
                metadata: None,
            }
        })
        .collect()
}

pub fn infer_available_model_descriptors(
    _available_models: &[String],
) -> Vec<ServedModelDescriptor> {
    Vec::new()
}

pub fn infer_local_served_model_descriptor(
    _model_name: &str,
    _is_primary: bool,
) -> Option<ServedModelDescriptor> {
    None
}

fn identity_from_local_path(
    model_name: &str,
    path: &std::path::Path,
) -> Option<ServedModelIdentity> {
    let local_file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .or_else(|| Some(format!("{model_name}.gguf")));
    Some(ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        local_file_name,
        ..Default::default()
    })
}

fn identity_from_model_source(source: &str) -> Option<ServedModelIdentity> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((repo_id, revision, selector)) = parse_model_ref_source(trimmed) {
        let canonical_ref = format_model_ref(&repo_id, revision.as_deref(), selector.as_deref());
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: selector,
            local_file_name: None,
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if is_explicit_local_path(trimmed) {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    if let Some((repo_id, revision, file)) = parse_hf_resolve_url_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if let Some((repo_id, revision, file)) = parse_hf_ref_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::DirectUrl,
            canonical_ref: Some(trimmed.to_string()),
            repository: None,
            revision: None,
            artifact: None,
            local_file_name: trimmed.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(trimmed)),
            weights_digest: None,
        });
    }

    if trimmed.ends_with(".gguf")
        || (trimmed.contains('/') && !trimmed.ends_with('/') && trimmed.split('/').count() != 2)
    {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    Some(ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::Catalog,
        canonical_ref: Some(trimmed.to_string()),
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: None,
        identity_hash: Some(identity_hash_for(&format!("catalog:{trimmed}"))),
        weights_digest: None,
    })
}

fn parse_model_ref_source(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    if input.starts_with("http://")
        || input.starts_with("https://")
        || is_explicit_local_path(input)
    {
        return None;
    }
    let (org, tail) = input.split_once('/')?;
    if org.is_empty() || tail.is_empty() || tail.contains('/') || org.contains(':') {
        return None;
    }
    let (repo, revision, selector) = parse_repo_tail_selector_and_revision(tail)?;
    Some((format!("{org}/{repo}"), revision, selector))
}

fn parse_repo_tail_selector_and_revision(
    tail: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let at_pos = tail.find('@');
    let colon_pos = tail.find(':');
    match (at_pos, colon_pos) {
        (Some(at), Some(colon)) if at < colon => nonempty_model_ref_parts(
            &tail[..at],
            Some(&tail[at + 1..colon]),
            Some(&tail[colon + 1..]),
        ),
        (Some(at), Some(colon)) if colon < at => nonempty_model_ref_parts(
            &tail[..colon],
            Some(&tail[at + 1..]),
            Some(&tail[colon + 1..at]),
        ),
        (Some(at), None) => nonempty_model_ref_parts(&tail[..at], Some(&tail[at + 1..]), None),
        (None, Some(colon)) => {
            nonempty_model_ref_parts(&tail[..colon], None, Some(&tail[colon + 1..]))
        }
        (None, None) => nonempty_model_ref_parts(tail, None, None),
        _ => None,
    }
}

fn nonempty_model_ref_parts(
    repo: &str,
    revision: Option<&str>,
    selector: Option<&str>,
) -> Option<(String, Option<String>, Option<String>)> {
    if repo.is_empty() || revision.is_some_and(str::is_empty) || selector.is_some_and(str::is_empty)
    {
        return None;
    }
    Some((
        repo.to_string(),
        revision.map(str::to_string),
        selector.map(str::to_string),
    ))
}

fn format_model_ref(repo: &str, revision: Option<&str>, selector: Option<&str>) -> String {
    match (revision, selector) {
        (Some(revision), Some(selector)) => format!("{repo}@{revision}:{selector}"),
        (Some(revision), None) => format!("{repo}@{revision}"),
        (None, Some(selector)) => format!("{repo}:{selector}"),
        (None, None) => repo.to_string(),
    }
}

fn is_explicit_local_path(source: &str) -> bool {
    source.starts_with('/') || source.starts_with("./") || source.starts_with("../")
}

fn local_gguf_identity_from_source(source: &str) -> ServedModelIdentity {
    let local_file_name = std::path::Path::new(source)
        .file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string);
    ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name,
        identity_hash: None,
        weights_digest: None,
    }
}

fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    if is_explicit_local_path(input) {
        return None;
    }
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, rev)) => (repo, Some(rev.to_string())),
        None => (parts[1], None),
    };
    if parts[0].is_empty() || repo_tail.is_empty() || parts[2].is_empty() {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

fn parse_hf_resolve_url_parts(url: &str) -> Option<(String, Option<String>, String)> {
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = path.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    let canonical = format!("{repo}@{revision}/{file}");
    parse_hf_ref_parts(&canonical)
}

fn format_hf_canonical_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(rev) => format!("{repo}@{rev}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod demand_tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;

    fn demand(last_active: u64, request_count: u64) -> ModelDemand {
        ModelDemand {
            last_active,
            request_count,
        }
    }

    fn merge(ours: &mut HashMap<String, ModelDemand>, theirs: &HashMap<String, ModelDemand>) {
        merge_demand_at(ours, theirs, NOW, |_| false);
    }

    fn full_map(last_active: u64) -> HashMap<String, ModelDemand> {
        (0..MAX_DEMAND_ENTRIES)
            .map(|i| (format!("model-{i}"), demand(last_active, 1)))
            .collect()
    }

    #[test]
    fn merge_clamps_demand_timestamped_in_the_future() {
        let latest_allowed = NOW + MAX_DEMAND_CLOCK_SKEW_SECS;
        let mut ours = HashMap::from([("known".to_string(), demand(NOW - 10, 1))]);
        let theirs = HashMap::from([
            ("known".to_string(), demand(u64::MAX, u64::MAX)),
            ("planted".to_string(), demand(u64::MAX, 1)),
            ("skewed".to_string(), demand(NOW + 60, 2)),
        ]);

        merge(&mut ours, &theirs);

        assert_eq!(ours["known"].last_active, latest_allowed);
        assert_eq!(ours["known"].request_count, u64::MAX);
        assert_eq!(ours["planted"].last_active, latest_allowed);
        assert_eq!(ours["skewed"].last_active, NOW + 60);
        // Clamped entries still expire on the normal TTL.
        assert!(latest_allowed.saturating_sub(NOW) < DEMAND_TTL_SECS);
    }

    #[test]
    fn merge_drops_empty_and_oversized_model_names() {
        let mut ours = HashMap::new();
        let theirs = HashMap::from([
            (String::new(), demand(NOW, 1)),
            ("x".repeat(MAX_DEMAND_MODEL_NAME_BYTES + 1), demand(NOW, 1)),
            ("x".repeat(MAX_DEMAND_MODEL_NAME_BYTES), demand(NOW, 1)),
        ]);

        merge(&mut ours, &theirs);

        assert_eq!(ours.len(), 1);
        assert!(ours.contains_key(&"x".repeat(MAX_DEMAND_MODEL_NAME_BYTES)));
    }

    #[test]
    fn merge_caps_new_entries_and_keeps_the_freshest() {
        let mut ours = HashMap::from([("ours".to_string(), demand(NOW - 1000, 1))]);
        let theirs: HashMap<_, _> = (0..MAX_DEMAND_ENTRIES as u64 * 2)
            .map(|i| (format!("new-{i}"), demand(NOW - i, 1)))
            .collect();

        merge(&mut ours, &theirs);

        // Every newcomer down to `new-{MAX-1}` is fresher than `ours`, so the
        // map ends up holding exactly the freshest MAX_DEMAND_ENTRIES.
        assert_eq!(ours.len(), MAX_DEMAND_ENTRIES);
        assert!(!ours.contains_key("ours"));
        assert!(ours.contains_key("new-0"));
        assert!(ours.contains_key(&format!("new-{}", MAX_DEMAND_ENTRIES - 1)));
        assert!(!ours.contains_key(&format!("new-{MAX_DEMAND_ENTRIES}")));
    }

    #[test]
    fn full_map_updates_known_models_and_refuses_staler_newcomers() {
        let mut ours = full_map(NOW - 500);
        let theirs = HashMap::from([
            ("model-0".to_string(), demand(NOW, 7)),
            ("stale-newcomer".to_string(), demand(NOW - 600, 1)),
        ]);

        merge(&mut ours, &theirs);

        assert_eq!(ours.len(), MAX_DEMAND_ENTRIES);
        assert_eq!(ours["model-0"].last_active, NOW);
        assert_eq!(ours["model-0"].request_count, 7);
        assert!(!ours.contains_key("stale-newcomer"));
    }

    #[test]
    fn full_map_evicts_stalest_unprotected_entry_for_fresher_demand() {
        let mut ours = full_map(NOW - 100);
        ours.insert("model-0".to_string(), demand(NOW - 900, 1));
        ours.insert("model-1".to_string(), demand(NOW - 800, 1));
        let theirs = HashMap::from([("fresh".to_string(), demand(NOW, 1))]);

        merge_demand_at(&mut ours, &theirs, NOW, |model| model == "model-0");

        assert_eq!(ours.len(), MAX_DEMAND_ENTRIES);
        assert!(ours.contains_key("fresh"));
        assert!(ours.contains_key("model-0"), "protected entry is kept");
        assert!(
            !ours.contains_key("model-1"),
            "stalest unprotected entry goes"
        );
    }

    #[test]
    fn full_map_of_protected_entries_admits_nothing() {
        let mut ours = full_map(NOW - 900);
        let theirs = HashMap::from([("fresh".to_string(), demand(NOW, 1))]);

        merge_demand_at(&mut ours, &theirs, NOW, |_| true);

        assert_eq!(ours.len(), MAX_DEMAND_ENTRIES);
        assert!(!ours.contains_key("fresh"));
    }
}
