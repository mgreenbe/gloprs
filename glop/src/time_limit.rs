//! Wall-clock and deterministic limits matching `ortools/util/time_limit.h`.

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::parameters::GlopParameters;

pub const SAFETY_BUFFER_SECONDS: f64 = 1e-4;
pub const HISTORY_SIZE: usize = 100;

#[derive(Debug)]
pub struct TimeLimit {
    start: Instant,
    last: Instant,
    wall_limit_seconds: f64,
    deterministic_limit: f64,
    elapsed_deterministic_time: f64,
    recent_intervals: VecDeque<f64>,
    wall_limit_reached: bool,
    external_limit: Option<Arc<AtomicBool>>,
    secondary_external_limit: Option<Arc<AtomicBool>>,
}

impl Default for TimeLimit {
    fn default() -> Self {
        Self::new(f64::INFINITY, f64::INFINITY)
    }
}

impl TimeLimit {
    #[must_use]
    pub fn new(wall_limit_seconds: f64, deterministic_limit: f64) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            wall_limit_seconds,
            deterministic_limit,
            elapsed_deterministic_time: 0.0,
            recent_intervals: VecDeque::with_capacity(HISTORY_SIZE),
            wall_limit_reached: false,
            external_limit: None,
            secondary_external_limit: None,
        }
    }

    #[must_use]
    pub fn from_parameters(parameters: &GlopParameters) -> Self {
        Self::new(
            parameters.max_time_in_seconds,
            parameters.max_deterministic_time,
        )
    }

    pub fn reset_from_parameters(&mut self, parameters: &GlopParameters) {
        self.reset(
            parameters.max_time_in_seconds,
            parameters.max_deterministic_time,
        );
    }

    pub fn reset(&mut self, wall_limit_seconds: f64, deterministic_limit: f64) {
        let now = Instant::now();
        self.start = now;
        self.last = now;
        self.wall_limit_seconds = wall_limit_seconds;
        self.deterministic_limit = deterministic_limit;
        self.elapsed_deterministic_time = 0.0;
        self.wall_limit_reached = false;
    }

    pub fn advance_deterministic_time(&mut self, duration: f64) {
        debug_assert!(duration >= 0.0);
        self.elapsed_deterministic_time += duration;
    }

    #[must_use]
    pub fn limit_reached(&mut self) -> bool {
        if self
            .external_limit
            .as_ref()
            .is_some_and(|limit| limit.load(Ordering::SeqCst))
            || self
                .secondary_external_limit
                .as_ref()
                .is_some_and(|limit| limit.load(Ordering::SeqCst))
            || self.deterministic_time_left() <= 0.0
        {
            return true;
        }
        if self.wall_limit_reached {
            return true;
        }
        let now = Instant::now();
        let interval = now
            .duration_since(self.last)
            .as_secs_f64()
            .max(SAFETY_BUFFER_SECONDS);
        self.last = now;
        if self.recent_intervals.len() == HISTORY_SIZE {
            self.recent_intervals.pop_front();
        }
        self.recent_intervals.push_back(interval);
        let running_max = self.recent_intervals.iter().copied().fold(0.0, f64::max);
        if self.elapsed_time() + running_max >= self.wall_limit_seconds {
            self.wall_limit_reached = true;
        }
        self.wall_limit_reached
    }

    #[must_use]
    pub fn time_left(&self) -> f64 {
        if self.wall_limit_reached {
            return 0.0;
        }
        if self.wall_limit_seconds == f64::INFINITY {
            return f64::INFINITY;
        }
        (self.wall_limit_seconds - self.elapsed_time()).max(0.0)
    }
    #[must_use]
    pub fn deterministic_time_left(&self) -> f64 {
        (self.deterministic_limit - self.elapsed_deterministic_time).max(0.0)
    }
    #[must_use]
    pub fn elapsed_time(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    #[must_use]
    pub const fn elapsed_deterministic_time(&self) -> f64 {
        self.elapsed_deterministic_time
    }
    #[must_use]
    pub const fn deterministic_limit(&self) -> f64 {
        self.deterministic_limit
    }
    pub const fn change_deterministic_limit(&mut self, limit: f64) {
        self.deterministic_limit = limit;
    }
    pub fn reset_history(&mut self) {
        self.recent_intervals.clear();
    }

    /// Restricts this limit to the remaining limits of `other`.
    ///
    /// Like GLOP, resetting preserves the running-max history and the
    /// secondary external limit. A primary external limit from `other`
    /// replaces this object's primary external limit when present.
    pub fn merge_with_global_time_limit(&mut self, other: Option<&Self>) {
        let Some(other) = other else {
            return;
        };
        let wall_limit = self.time_left().min(other.time_left());
        let deterministic_limit = self
            .deterministic_time_left()
            .min(other.deterministic_time_left());
        self.reset(wall_limit, deterministic_limit);
        if let Some(external_limit) = &other.external_limit {
            self.external_limit = Some(Arc::clone(external_limit));
        }
    }

    #[must_use]
    pub fn external_limit(&self) -> Option<&Arc<AtomicBool>> {
        self.external_limit.as_ref()
    }
    pub fn register_external_limit(&mut self, limit: Option<Arc<AtomicBool>>) {
        self.external_limit = limit;
    }
    pub fn register_secondary_external_limit(&mut self, limit: Option<Arc<AtomicBool>>) {
        self.secondary_external_limit = limit;
    }
}

#[cfg(test)]
mod tests {
    use super::TimeLimit;
    #[test]
    fn deterministic_limit_is_inclusive() {
        let mut limit = TimeLimit::new(f64::INFINITY, 1.0);
        limit.advance_deterministic_time(1.0);
        assert!(limit.limit_reached());
        limit.change_deterministic_limit(2.0);
        assert!(!limit.limit_reached());
    }
}
