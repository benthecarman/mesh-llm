//! A running host must not overwrite edits made to its config file by the
//! CLI or an operator while it runs.

use super::*;

fn config_with_model(model: &str) -> MeshConfig {
    let mut config = minimal_valid_config();
    config.models.push(crate::plugin::ModelConfigEntry {
        model: model.to_string(),
        ..Default::default()
    });
    config
}

fn configured_models(config: &MeshConfig) -> Vec<&str> {
    config
        .models
        .iter()
        .map(|model| model.model.as_str())
        .collect()
}

fn models_on_disk(config_path: &Path) -> Vec<String> {
    ConfigStore::open(config_path)
        .model_refs()
        .expect("read config models")
}

#[test]
fn apply_after_a_cli_edit_reports_a_conflict_and_keeps_the_edit() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    assert!(matches!(
        state.apply(config_with_model("host-a.gguf"), 0),
        ApplyResult::Applied { revision: 1, .. }
    ));

    // The CLI edits the file while the host keeps revision 1 in memory.
    ConfigStore::open(&config_path)
        .add_model_ref("cli-b.gguf")
        .expect("cli edit");

    let result = state.apply(config_with_model("host-c.gguf"), 1);

    assert!(
        matches!(
            result,
            ApplyResult::RevisionConflict {
                current_revision: 2
            }
        ),
        "an apply based on the config before the edit must conflict, got {result:?}"
    );
    assert_eq!(
        models_on_disk(&config_path),
        ["host-a.gguf", "cli-b.gguf"],
        "the CLI edit must survive"
    );
    assert_eq!(
        configured_models(state.config()),
        ["host-a.gguf", "cli-b.gguf"],
        "the host takes the edit in so the next apply builds on it"
    );
    assert_eq!(state.revision(), 2);
    assert_eq!(read_revision(&revision_sidecar_path(&config_path)), 2);

    let mut rebased = state.config().clone();
    rebased.models.push(crate::plugin::ModelConfigEntry {
        model: "host-c.gguf".to_string(),
        ..Default::default()
    });
    assert!(matches!(
        state.apply(rebased, 2),
        ApplyResult::Applied { revision: 3, .. }
    ));
    assert_eq!(
        models_on_disk(&config_path),
        ["host-a.gguf", "cli-b.gguf", "host-c.gguf"]
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_config_file_created_after_startup_is_not_overwritten() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load without a file");

    ConfigStore::open(&config_path)
        .add_model_ref("cli-only.gguf")
        .expect("cli creates the file");

    let result = state.apply(config_with_model("host-only.gguf"), 0);

    assert!(
        matches!(
            result,
            ApplyResult::RevisionConflict {
                current_revision: 1
            }
        ),
        "got {result:?}"
    );
    assert_eq!(models_on_disk(&config_path), ["cli-only.gguf"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_edit_the_host_cannot_load_is_left_alone() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    assert!(matches!(
        state.apply(config_with_model("host-a.gguf"), 0),
        ApplyResult::Applied { .. }
    ));
    let broken = "version = 1\n[[models]\n";
    std::fs::write(&config_path, broken).expect("operator edit");

    let result = state.apply(config_with_model("host-b.gguf"), 1);

    let ApplyResult::PersistError(error) = result else {
        panic!("expected the apply to be refused, got {result:?}");
    };
    assert!(
        error.contains("edited while the host was running"),
        "{error}"
    );
    assert_eq!(std::fs::read_to_string(&config_path).unwrap(), broken);
    assert_eq!(state.revision(), 1, "nothing was taken in");
    assert_eq!(configured_models(state.config()), ["host-a.gguf"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn consecutive_host_applies_do_not_conflict_with_their_own_writes() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    std::fs::write(&config_path, "version = 1\n# hand-written\n").expect("seed config");
    let mut state = ConfigState::load(&config_path).expect("load");

    for (revision, model) in ["one.gguf", "two.gguf", "three.gguf"].iter().enumerate() {
        let result = state.apply(config_with_model(model), revision as u64);
        assert!(
            matches!(result, ApplyResult::Applied { .. }),
            "apply {revision} got {result:?}"
        );
    }
    assert_eq!(models_on_disk(&config_path), ["three.gguf"]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_edit_of_the_current_config_is_rebased_on_a_disk_edit() {
    let dir = test_dir();
    let config_path = dir.join("config.toml");
    let mut state = ConfigState::load(&config_path).expect("load");
    assert!(matches!(
        state.apply(config_with_model("host-a.gguf"), 0),
        ApplyResult::Applied { revision: 1, .. }
    ));
    ConfigStore::open(&config_path)
        .add_model_ref("cli-b.gguf")
        .expect("cli edit");

    let (result, edits) = state.apply_edit(|config| {
        config.models.push(crate::plugin::ModelConfigEntry {
            model: "console-c.gguf".to_string(),
            ..Default::default()
        });
        config.models.len()
    });

    assert!(
        matches!(result, ApplyResult::Applied { revision: 3, .. }),
        "the edit is applied on top of the disk edit, got {result:?}"
    );
    assert_eq!(
        edits, 3,
        "the caller sees the result of the edit that was saved"
    );
    assert_eq!(
        models_on_disk(&config_path),
        ["host-a.gguf", "cli-b.gguf", "console-c.gguf"]
    );

    std::fs::remove_dir_all(&dir).ok();
}
