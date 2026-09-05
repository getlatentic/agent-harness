//! Set process environment variables for the length of one test.
//!
//! `std::env::set_var` is `unsafe` on Rust 2024 because a `getenv` running on
//! another thread can read memory it frees. Tests run on many threads, so every
//! mutation in this crate goes through the one lock here, and each is undone
//! when its guard drops — including on a panicking test.
//!
//! What the lock does not cover, stated once rather than at every call site: a
//! test that never touches the environment but spawns a process reads it at
//! `exec`, and can do so while a test here is mid-mutation. That is the race
//! `set_var` is unsafe for, and no lock in this module can close it; running
//! each test in its own process (`cargo nextest`) would. Until then the exposure
//! is bounded to the handful of tests below that spawn shells.

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, PoisonError};

static ENV: Mutex<()> = Mutex::new(());

/// Holds the environment lock, and puts back what it changed when dropped.
pub(crate) struct ScopedEnv {
    restore: Vec<(String, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

/// Take the lock without changing anything yet — for a test that reads the
/// environment through the code under test, or sets one variable several ways.
pub(crate) fn lock() -> ScopedEnv {
    let lock = ENV.lock().unwrap_or_else(PoisonError::into_inner);
    ScopedEnv { restore: Vec::new(), _lock: lock }
}

/// Set `key` to `value` until the guard drops.
pub(crate) fn set(key: &str, value: &str) -> ScopedEnv {
    scoped(&[(key, Some(value))])
}

/// Set or remove several variables at once until the guard drops.
pub(crate) fn scoped(vars: &[(&str, Option<&str>)]) -> ScopedEnv {
    let mut env = lock();
    for (key, value) in vars {
        env.set(key, *value);
    }
    env
}

/// Remove `key` until the guard drops.
pub(crate) fn unset(key: &str) -> ScopedEnv {
    scoped(&[(key, None)])
}

impl ScopedEnv {
    /// Set `key` (or remove it, with `None`). The value it had when this guard
    /// first touched it is what comes back on drop, however many times it is
    /// changed in between.
    pub(crate) fn set(&mut self, key: &str, value: Option<&str>) {
        if !self.restore.iter().any(|(k, _)| k == key) {
            self.restore.push((key.to_owned(), std::env::var_os(key)));
        }
        apply(key, value.map(OsString::from).as_deref());
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        // Fields drop after this body, so the lock is held through the restore.
        for (key, value) in self.restore.drain(..) {
            apply(&key, value.as_deref());
        }
    }
}

fn apply(key: &str, value: Option<&std::ffi::OsStr>) {
    // SAFETY: the caller holds `ENV`, so no other test is mutating the
    // environment or reading it through this module. The module doc says what
    // that does not cover.
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guard_restores_the_first_value_it_saw_however_often_it_changed() {
        // One guard at a time: the lock is not reentrant, so two live guards
        // on one thread would wait on each other forever.
        let key = "AGENT_HARNESS_TEST_ENV_GUARD";
        {
            let mut env = set(key, "one");
            env.set(key, Some("two"));
            env.set(key, None);
            assert_eq!(std::env::var_os(key), None);
        }
        assert_eq!(std::env::var_os(key), None, "unset before the guard, so unset after it");

        {
            let _env = set(key, "before");
            assert_eq!(std::env::var(key).as_deref(), Ok("before"));
        }
        {
            let _env = unset(key);
            assert_eq!(std::env::var_os(key), None);
        }
        assert_eq!(std::env::var_os(key), None, "each guard put back what it found");
    }
}
