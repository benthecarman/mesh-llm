//! Finding a stage load's model source on this node by name, when the load
//! carries no local path that exists here.

use std::path::{Component, Path};

use crate::inference::skippy::{StageLoadRequest, StageSourceFetch};

/// Who asked this node to load a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageLoadOrigin {
    /// This node's own runtime, over the in-process control channel.
    Local,
    /// Another mesh node, over stage control.
    Peer,
}

/// Set `load.model_path` from the load's model names through the model store,
/// the catalog, or a download that `source_fetch` allows, when a runtime-slice
/// load has no path that exists here.
pub(crate) async fn resolve_stage_source_by_name(
    load: &mut StageLoadRequest,
    origin: StageLoadOrigin,
    source_fetch: StageSourceFetch,
) {
    if load.load_mode != skippy_protocol::LoadMode::RuntimeSlice
        || load
            .model_path
            .as_deref()
            .is_some_and(|path| Path::new(path).exists())
    {
        return;
    }
    let candidates = source_name_candidates(&load.model_id, &load.package_ref, origin)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    for candidate in candidates {
        if let Some(path) = source_fetch.resolve_model_candidate(&candidate).await {
            load.model_path = Some(path.to_string_lossy().to_string());
            return;
        }
    }
}

fn source_name_candidates<'a>(
    model_id: &'a str,
    package_ref: &'a str,
    origin: StageLoadOrigin,
) -> Vec<&'a str> {
    [
        model_id,
        package_ref.strip_prefix("gguf://").unwrap_or_default(),
    ]
    .into_iter()
    .filter(|candidate| !candidate.is_empty())
    .filter(|candidate| origin == StageLoadOrigin::Local || !names_a_local_path(candidate))
    .collect()
}

/// A peer names a model, never a file on this node. The resolver opens any
/// name that is an existing path as given, and joins other names onto the
/// model store, so a peer's name must be neither a path that exists here nor
/// one that can leave the store.
fn names_a_local_path(candidate: &str) -> bool {
    let path = Path::new(candidate);
    path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_source_names_cannot_be_local_paths() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("model.gguf");
        std::fs::write(&existing, b"GGUF").unwrap();
        let existing = existing.to_string_lossy().into_owned();

        let cases = [
            (
                existing.clone(),
                "gguf:///no/such/model.gguf".to_string(),
                Vec::<&str>::new(),
            ),
            (
                "org/repo:Q4_K_M".to_string(),
                format!("gguf://{existing}"),
                vec!["org/repo:Q4_K_M"],
            ),
            (
                "../../outside/model".to_string(),
                "gguf://./model.gguf".to_string(),
                Vec::new(),
            ),
            (
                "/no/such/model.gguf".to_string(),
                "gguf://org/../../model.gguf".to_string(),
                Vec::new(),
            ),
        ];
        for (model_id, package_ref, expected) in &cases {
            assert_eq!(
                &source_name_candidates(model_id, package_ref, StageLoadOrigin::Peer),
                expected,
                "peer names ({model_id}, {package_ref})"
            );
            assert_eq!(
                source_name_candidates(model_id, package_ref, StageLoadOrigin::Local).len(),
                2,
                "local loads keep every name"
            );
        }
    }

    #[test]
    fn peer_source_names_keep_model_references() {
        assert_eq!(
            source_name_candidates(
                "unsloth/Qwen3-8B-GGUF:Q4_K_M",
                "hf://meshllm/demo",
                StageLoadOrigin::Peer
            ),
            vec!["unsloth/Qwen3-8B-GGUF:Q4_K_M"]
        );
        assert_eq!(
            source_name_candidates(
                "Qwen3-8B-Q4_K_M",
                "gguf://Qwen3-8B-Q4_K_M.gguf",
                StageLoadOrigin::Peer
            ),
            vec!["Qwen3-8B-Q4_K_M", "Qwen3-8B-Q4_K_M.gguf"]
        );
    }
}
