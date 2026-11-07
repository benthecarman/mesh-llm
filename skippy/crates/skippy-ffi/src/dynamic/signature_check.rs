//! Compile-time proof that the dynamic symbol signatures match the bindgen
//! declarations.
//!
//! The dynamic runtime resolves each native entry point by name and casts it
//! to a function-pointer type written by hand in the parent module. The name
//! lookup cannot see the signature, so a hand-written type that drifts from
//! the pinned headers would compile, load, and then miscall. The checks here
//! coerce every generated `extern "C"` declaration in [`crate::bindings`] to
//! a function pointer and require each parameter and the return type to
//! match the hand-written type, so drift fails the build instead.
//!
//! Every check is a statement in the body of an unnamed constant. Constant
//! bodies are type-checked and evaluated at compile time but never
//! code-generated, so the checks add no link-time reference to a native
//! symbol that a dynamic build does not link, even under `-C link-dead-code`.
//!
//! The parent module runs the checks where it names each symbol: the
//! required table, `optional_symbol!`, and `draft_api_symbol!`. A symbol
//! cannot be looked up through them without being checked.

use std::marker::PhantomData;

use crate::Opaque;
use crate::bindings::{llama, mtmd, skippy};
use crate::llama_draft::LlamaDraftApi;

/// `Self` crosses the C ABI as the generated native type `Native`.
///
/// Types must be identical, with three narrow exceptions listed below:
/// `*mut Opaque` and `*const Opaque` stand in for a pointer with the same
/// mutability to one of the listed opaque llama.cpp or mtmd handle types;
/// `i32` stands in for the `#[repr(i32)]` rope-type enum; and the llama copy
/// of a ggml typedef stands in for the mtmd copy of the same typedef.
///
/// Because the facade erases handles to [`Opaque`], a swap between two
/// listed handles of the same mutability (a context passed where a model is
/// expected) is not detectable here.
#[diagnostic::on_unimplemented(
    message = "facade type `{Self}` does not match the bindgen type `{Native}`",
    label = "the dynamic symbol table declares a signature the native header does not have"
)]
pub(super) trait AbiSame<Native> {}

impl<T> AbiSame<T> for T {}

macro_rules! opaque_handles {
    ($($handle:ty),+ $(,)?) => {
        $(
            impl AbiSame<*mut $handle> for *mut Opaque {}
            impl AbiSame<*const $handle> for *const Opaque {}
        )+
    };
}

// Bindgen emits each forward-declared handle once per generated module.
opaque_handles!(
    llama::llama_context,
    llama::llama_model,
    llama::llama_vocab,
    mtmd::llama_context,
    mtmd::llama_model,
    mtmd::mtmd_image_tokens,
    mtmd::mtmd_input_chunk,
    skippy::llama_context,
    skippy::llama_model,
    llama::llama_memory_i,
    llama::ggml_backend_device,
);

// The draft API reads the `#[repr(i32)]` rope-type enum as its integer
// representation so an unknown native value never becomes an invalid enum.
impl AbiSame<llama::llama_rope_type> for i32 {}

// The llama and mtmd modules each carry a generated copy of ggml's
// `ggml_log_callback` typedef; the facade passes the llama copy to both.
impl AbiSame<mtmd::ggml_log_callback> for llama::ggml_log_callback {}

macro_rules! same_log_levels {
    ($($level:ident),+) => {
        $(const _: () = assert!(
            llama::ggml_log_level::$level as u32 == mtmd::ggml_log_level::$level as u32
        );)+
    };
}

// Both copies come from one C enumeration; fail the build if they diverge.
const _: () = assert!(size_of::<llama::ggml_log_level>() == size_of::<mtmd::ggml_log_level>());
same_log_levels!(None, Debug, Info, Warn, Error, Cont);

/// The facade function-pointer type `Self` has the same arity as `Native`
/// and every parameter and the return type satisfy [`AbiSame`].
pub(super) trait SignatureMatches<Native> {}

macro_rules! signature_matches_impl {
    ($($facade:ident $native:ident),*) => {
        impl<FacadeRet, NativeRet, $($facade, $native),*>
            SignatureMatches<unsafe extern "C" fn($($native),*) -> NativeRet>
            for unsafe extern "C" fn($($facade),*) -> FacadeRet
        where
            FacadeRet: AbiSame<NativeRet>,
            $($facade: AbiSame<$native>,)*
        {
        }
    };
}

/// Implements [`SignatureMatches`] for every arity up to the given parameter
/// list by peeling one parameter pair per step.
macro_rules! signature_matches_arities {
    () => {
        signature_matches_impl!();
    };
    ($facade:ident $native:ident $(, $rest_facade:ident $rest_native:ident)*) => {
        signature_matches_impl!($facade $native $(, $rest_facade $rest_native)*);
        signature_matches_arities!($($rest_facade $rest_native),*);
    };
}

// The widest checked entry point, `skippy_apply_chat_template_json`, takes 22
// parameters; adding a wider symbol fails to compile until this list grows.
signature_matches_arities!(
    F0 N0, F1 N1, F2 N2, F3 N3, F4 N4, F5 N5, F6 N6, F7 N7, F8 N8, F9 N9, F10 N10, F11 N11,
    F12 N12, F13 N13, F14 N14, F15 N15, F16 N16, F17 N17, F18 N18, F19 N19, F20 N20, F21 N21
);

/// Accepts `native` only when the facade type `Facade` matches its signature.
///
/// Callers coerce the generated declaration to a function pointer whose
/// parameter types are inferred (`_`), which pins the native arity; the
/// bound then compares each type.
pub(super) const fn assert_signature<Facade, Native>(_facade: PhantomData<Facade>, _native: &Native)
where
    Facade: SignatureMatches<Native>,
{
}

/// Names the type of one [`LlamaDraftApi`] field through a projection.
#[cfg(not(windows))]
pub(super) const fn draft_api_field<Field>(
    _project: fn(&LlamaDraftApi) -> Field,
) -> PhantomData<Field> {
    PhantomData
}

/// Expands to the inferred type `_` for one parameter of a checked signature.
macro_rules! inferred_parameter {
    ($parameter:ty) => {
        _
    };
}
pub(super) use inferred_parameter;

/// Checks a named function-pointer type against the generated declaration
/// `$module::$name`. Each `_` stands for one native parameter; a wrong count
/// fails the coercion to the generated declaration.
macro_rules! check_signature {
    ($module:ident::$name:ident($($parameter:tt),*) as $facade:ty) => {
        const _: () = {
            let native: unsafe extern "C" fn($($parameter),*) -> _ =
                $crate::bindings::$module::$name;
            $crate::dynamic::signature_check::assert_signature(
                ::std::marker::PhantomData::<$facade>,
                &native,
            );
        };
    };
}
pub(super) use check_signature;

/// Checks the type of the [`LlamaDraftApi`] field `$field` against the
/// generated `llama.h` declaration `$name`. Windows runtimes have no draft
/// API, so there is nothing to check there.
#[cfg(not(windows))]
macro_rules! check_draft_api_field {
    ($field:ident: $name:ident($($parameter:tt),*)) => {
        const _: () = {
            let native: unsafe extern "C" fn($($parameter),*) -> _ = $crate::bindings::llama::$name;
            $crate::dynamic::signature_check::assert_signature(
                $crate::dynamic::signature_check::draft_api_field(|api| api.$field),
                &native,
            );
        };
    };
}
#[cfg(not(windows))]
pub(super) use check_draft_api_field;

#[cfg(test)]
mod tests {
    //! The production checks only prove that matching signatures compile. A
    //! mismatch is a compile error, so these tests probe the same trait
    //! bounds through autoref method resolution, which picks the matching
    //! impl when the bound holds and the fallback impl otherwise.

    use std::ffi::{c_char, c_void};
    use std::marker::PhantomData;

    use super::SignatureMatches;
    use crate::Opaque;
    use crate::bindings::{llama, mtmd};

    struct Probe<Facade, Native>(PhantomData<(Facade, Native)>);

    trait Accepted {
        fn accepted(&self) -> bool {
            true
        }
    }

    impl<Facade: SignatureMatches<Native>, Native> Accepted for Probe<Facade, Native> {}

    trait Rejected {
        fn accepted(&self) -> bool {
            false
        }
    }

    impl<Facade, Native> Rejected for &Probe<Facade, Native> {}

    macro_rules! accepted {
        ($facade:ty, $native:ty) => {
            (&Probe::<$facade, $native>(PhantomData)).accepted()
        };
    }

    #[test]
    fn identical_signatures_are_accepted() {
        assert!(accepted!(
            unsafe extern "C" fn(*mut crate::Session, u32) -> crate::Status,
            unsafe extern "C" fn(*mut crate::Session, u32) -> crate::Status
        ));
        assert!(accepted!(unsafe extern "C" fn(), unsafe extern "C" fn()));
    }

    #[test]
    fn opaque_pointers_stand_in_for_native_handles() {
        assert!(accepted!(
            unsafe extern "C" fn(*mut Opaque, i32) -> *mut f32,
            unsafe extern "C" fn(*mut llama::llama_context, i32) -> *mut f32
        ));
        assert!(accepted!(
            unsafe extern "C" fn(*const Opaque) -> *const Opaque,
            unsafe extern "C" fn(*const mtmd::mtmd_input_chunk) -> *const mtmd::mtmd_image_tokens
        ));
    }

    #[test]
    fn opaque_pointers_keep_their_mutability() {
        assert!(!accepted!(
            unsafe extern "C" fn(*const Opaque),
            unsafe extern "C" fn(*mut llama::llama_context)
        ));
        assert!(!accepted!(
            unsafe extern "C" fn() -> *mut Opaque,
            unsafe extern "C" fn() -> *const llama::llama_model
        ));
    }

    #[test]
    fn opaque_pointers_do_not_erase_non_handle_types() {
        assert!(!accepted!(
            unsafe extern "C" fn(*mut Opaque),
            unsafe extern "C" fn(*mut f32)
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(*const Opaque),
            unsafe extern "C" fn(*const crate::SamplingConfig)
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(*const c_void),
            unsafe extern "C" fn(*const c_char)
        ));
    }

    #[test]
    fn parameter_type_drift_is_rejected() {
        assert!(!accepted!(
            unsafe extern "C" fn(*mut crate::Session, u64) -> crate::Status,
            unsafe extern "C" fn(*mut crate::Session, u32) -> crate::Status
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(*const crate::RuntimeConfig),
            unsafe extern "C" fn(*mut crate::RuntimeConfig)
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(crate::SamplingConfig),
            unsafe extern "C" fn(*const crate::SamplingConfig)
        ));
    }

    #[test]
    fn return_type_drift_is_rejected() {
        assert!(!accepted!(
            unsafe extern "C" fn() -> u32,
            unsafe extern "C" fn() -> i32
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(),
            unsafe extern "C" fn() -> crate::Status
        ));
    }

    #[test]
    fn arity_drift_is_rejected() {
        assert!(!accepted!(
            unsafe extern "C" fn(i32),
            unsafe extern "C" fn(i32, i32)
        ));
        assert!(!accepted!(
            unsafe extern "C" fn(i32, i32),
            unsafe extern "C" fn(i32)
        ));
    }
}
