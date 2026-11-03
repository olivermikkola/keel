//! Ctrl-C handling for interactive CLI sessions.
//!
//! The first SIGINT flips a [`CancelToken`] so the agent loop stops cooperatively at the
//! next step or tool boundary. A second SIGINT before the token is reset terminates the
//! process immediately with the conventional exit status 130.

use crate::CancelToken;

#[cfg(unix)]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    use std::sync::Arc;

    static FLAG: AtomicPtr<AtomicBool> = AtomicPtr::new(std::ptr::null_mut());

    extern "C" fn on_sigint(_sig: libc::c_int) {
        let p = FLAG.load(Ordering::SeqCst);
        if p.is_null() {
            unsafe { libc::_exit(130) };
        }
        // SAFETY: the pointer comes from `Arc::into_raw` and is intentionally leaked,
        // so it stays valid for the life of the process.
        let flag = unsafe { &*p };
        if flag.swap(true, Ordering::SeqCst) {
            unsafe { libc::_exit(130) };
        }
    }

    pub fn install(flag: Arc<AtomicBool>) {
        FLAG.store(Arc::into_raw(flag) as *mut AtomicBool, Ordering::SeqCst);
        // SAFETY: zeroed sigaction is a valid starting value; we set handler and mask.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_sigint as extern "C" fn(libc::c_int) as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = 0;
            libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        }
    }
}

/// Route SIGINT to `token`. Call at most once per process; later calls are ignored.
pub fn install_sigint_cancel(token: &CancelToken) {
    static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if INSTALLED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    #[cfg(unix)]
    imp::install(token.0.clone());
    #[cfg(not(unix))]
    let _ = token;
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn sigint_sets_token_without_killing_process() {
        let token = CancelToken::new();
        install_sigint_cancel(&token);
        assert!(!token.is_cancelled());
        unsafe { libc::raise(libc::SIGINT) };
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }
}
