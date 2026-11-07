//! Ownership of the native half of a [`StageSession`](super::StageSession).

use std::ptr::{self, NonNull};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use skippy_ffi::Session as RawSession;

use crate::native::StageModelInner;

/// One native session and the model whose context it borrows.
///
/// Only the model's session constructors build one. Dropping it frees the
/// native session first and then releases the model, so the model outlives
/// the session whatever order callers drop their handles in.
pub(crate) struct NativeSession {
    raw: NonNull<RawSession>,
    _model: Arc<StageModelInner>,
}

impl NativeSession {
    /// Takes ownership of `raw`, a session that `model` created. `None` when
    /// the native call returned no session.
    pub(crate) fn adopt(raw: *mut RawSession, model: &Arc<StageModelInner>) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self {
            raw,
            _model: Arc::clone(model),
        })
    }
}

impl Drop for NativeSession {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live session this value owns, and `_model`
        // keeps the model whose context it borrows alive until after this.
        unsafe {
            let _ = skippy_ffi::skippy_session_free(self.raw.as_ptr(), ptr::null_mut());
        }
    }
}

/// The native side of a [`StageSession`](super::StageSession).
pub(crate) enum SessionHandle {
    Native(NativeSession),
    /// No native session, for tests of the bookkeeping above one. Native
    /// position and sequence reads report that there is none; any other
    /// native call on it is a test bug and panics.
    #[cfg(any(test, feature = "test-support"))]
    Detached,
}

impl SessionHandle {
    /// The live native session.
    pub(crate) fn native(&self) -> Option<&NativeSession> {
        match self {
            Self::Native(native) => Some(native),
            #[cfg(any(test, feature = "test-support"))]
            Self::Detached => None,
        }
    }

    /// The native session pointer for an FFI call.
    pub(crate) fn as_ptr(&self) -> *mut RawSession {
        match self.native() {
            Some(native) => native.raw.as_ptr(),
            None => panic!("a detached test session has no native session"),
        }
    }

    /// The native session pointer, or an error for a detached test session.
    pub(crate) fn try_ptr(&self) -> Result<*mut RawSession> {
        self.native()
            .map(|native| native.raw.as_ptr())
            .ok_or_else(|| anyhow!("skippy session has no native session"))
    }
}
