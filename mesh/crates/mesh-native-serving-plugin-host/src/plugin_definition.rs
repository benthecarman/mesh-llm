//! Loading and validating one native serving plugin's function table.
//!
//! The table comes from an independently compiled library, so every entry is
//! untrusted until checked. Validation copies the callbacks out of the table
//! once; later host calls go through [`PluginCallbacks`] and can never reach a
//! NULL entry.

use std::{ffi::c_char, path::Path, ptr::NonNull};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use mesh_native_serving_plugin_api as abi;

const ERROR_BUFFER_BYTES: usize = 2_048;

/// Every callback of a validated plugin table.
///
/// The ABI requires the table to stay immutable for the library's lifetime,
/// so copying the function pointers at load is equivalent to reading the
/// table on each call.
#[derive(Clone, Copy)]
pub(crate) struct PluginCallbacks {
    pub(crate) activate: abi::ActivatePlugin,
    pub(crate) shutdown: abi::ShutdownPlugin,
    pub(crate) begin_generation: abi::BeginGeneration,
    pub(crate) commit_generation: abi::CommitGeneration,
    pub(crate) abort_generation: abi::AbortGeneration,
    pub(crate) finish_generation: abi::FinishGeneration,
    pub(crate) start_proposal: abi::StartProposal,
    pub(crate) poll_proposal: abi::PollProposal,
    pub(crate) cancel_proposal: abi::CancelProposal,
    pub(crate) report_proposal: abi::ReportProposal,
    pub(crate) discard_proposal: abi::DiscardProposal,
    pub(crate) last_error: abi::LastError,
}

pub(crate) struct LoadedDefinition {
    pub(crate) _library: Option<Library>,
    pub(crate) callbacks: PluginCallbacks,
    pub(crate) name: String,
}

impl LoadedDefinition {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load native serving plugin {}", path.display()))?;
        let entry = unsafe {
            library.get::<abi::NativeServingPluginEntryV2>(abi::NATIVE_SERVING_PLUGIN_ENTRY_V2)
        }
        .with_context(|| {
            format!(
                "resolve native serving plugin entrypoint in {}",
                path.display()
            )
        })?;
        let api = NonNull::new(unsafe { entry() }.cast_mut())
            .context("native serving plugin entrypoint returned null")?;
        let (name, callbacks) = validate_table(unsafe { api.as_ref() })?;
        Ok(Self {
            _library: Some(library),
            callbacks,
            name,
        })
    }

    pub(crate) fn status_error(
        &self,
        instance: abi::PluginInstance,
        action: &str,
        status: abi::PluginStatus,
    ) -> anyhow::Error {
        let detail = self.last_error(instance);
        anyhow!("{action} `{}` failed with {status:?}: {detail}", self.name)
    }

    fn last_error(&self, instance: abi::PluginInstance) -> String {
        let mut buffer = [0_u8; ERROR_BUFFER_BYTES];
        let written = unsafe {
            (self.callbacks.last_error)(
                instance,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len(),
            )
        };
        let length = written.min(buffer.len());
        String::from_utf8_lossy(&buffer[..length]).into_owned()
    }
}

/// Checks a plugin table's version, layout, name, and callbacks, and returns
/// the plugin name with the resolved callbacks.
pub(crate) fn validate_table(
    table: &abi::NativeServingPluginV2,
) -> Result<(String, PluginCallbacks)> {
    if table.abi_version != abi::NATIVE_SERVING_PLUGIN_ABI_V2 {
        bail!(
            "native serving plugin ABI {} is incompatible with host ABI {}",
            table.abi_version,
            abi::NATIVE_SERVING_PLUGIN_ABI_V2
        );
    }
    if table.struct_size != size_of::<abi::NativeServingPluginV2>() {
        bail!(
            "native serving plugin table size {} does not match host size {}",
            table.struct_size,
            size_of::<abi::NativeServingPluginV2>()
        );
    }
    let name = unsafe { read_utf8(table.plugin_name, "plugin name") }?;
    if name.trim().is_empty() {
        bail!("native serving plugin name must not be empty");
    }
    let callbacks = PluginCallbacks {
        activate: required(table.activate, "activate")?,
        shutdown: required(table.shutdown, "shutdown")?,
        begin_generation: required(table.begin_generation, "begin_generation")?,
        commit_generation: required(table.commit_generation, "commit_generation")?,
        abort_generation: required(table.abort_generation, "abort_generation")?,
        finish_generation: required(table.finish_generation, "finish_generation")?,
        start_proposal: required(table.start_proposal, "start_proposal")?,
        poll_proposal: required(table.poll_proposal, "poll_proposal")?,
        cancel_proposal: required(table.cancel_proposal, "cancel_proposal")?,
        report_proposal: required(table.report_proposal, "report_proposal")?,
        discard_proposal: required(table.discard_proposal, "discard_proposal")?,
        last_error: required(table.last_error, "last_error")?,
    };
    Ok((name, callbacks))
}

fn required<T>(callback: Option<T>, field: &str) -> Result<T> {
    callback.with_context(|| format!("native serving plugin table has a null `{field}` callback"))
}

unsafe fn read_utf8(slice: abi::ByteSlice, label: &str) -> Result<String> {
    if slice.pointer.is_null() && slice.length != 0 {
        bail!("native serving plugin {label} has a null pointer");
    }
    let bytes = if slice.length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(slice.pointer, slice.length) }
    };
    std::str::from_utf8(bytes)
        .with_context(|| format!("native serving plugin {label} is not UTF-8"))
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fake_table;

    type ClearCallback = fn(&mut abi::NativeServingPluginV2);

    fn validation_error(table: &abi::NativeServingPluginV2) -> String {
        match validate_table(table) {
            Ok(_) => panic!("table validation unexpectedly succeeded"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn table_validation_rejects_version_and_layout_mismatches() {
        let mut table = fake_table();
        assert_eq!(validate_table(&table).unwrap().0, "test-serving-plugin");

        table.abi_version += 1;
        assert!(validation_error(&table).contains("incompatible"));
        table.abi_version = abi::NATIVE_SERVING_PLUGIN_ABI_V2;
        table.struct_size -= 1;
        assert!(validation_error(&table).contains("table size"));
    }

    #[test]
    fn table_validation_rejects_each_null_callback() {
        let clears: [(&str, ClearCallback); 12] = [
            ("activate", |table| table.activate = None),
            ("shutdown", |table| table.shutdown = None),
            ("begin_generation", |table| table.begin_generation = None),
            ("commit_generation", |table| table.commit_generation = None),
            ("abort_generation", |table| table.abort_generation = None),
            ("finish_generation", |table| table.finish_generation = None),
            ("start_proposal", |table| table.start_proposal = None),
            ("poll_proposal", |table| table.poll_proposal = None),
            ("cancel_proposal", |table| table.cancel_proposal = None),
            ("report_proposal", |table| table.report_proposal = None),
            ("discard_proposal", |table| table.discard_proposal = None),
            ("last_error", |table| table.last_error = None),
        ];
        for (field, clear) in clears {
            let mut table = fake_table();
            clear(&mut table);
            let error = validation_error(&table);
            assert!(
                error.contains(&format!("null `{field}` callback")),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn table_validation_rejects_a_null_entry_written_by_c() {
        // A C plugin leaves an entry zeroed rather than writing Rust's `None`.
        // The bytes must read back as `None`, not as a callable pointer.
        let mut table = fake_table();
        let entry = (&raw mut table.finish_generation).cast::<*const ()>();
        unsafe { entry.write(std::ptr::null()) };
        assert!(validation_error(&table).contains("null `finish_generation` callback"));
    }
}
