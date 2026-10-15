//! Wall-clock and deterministic limits matching `ortools/util/time_limit.h`.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::parameters::GlopParameters;

pub const SAFETY_BUFFER_SECONDS: f64 = 1e-4;
pub const HISTORY_SIZE: usize = 100;

/// Fixed-window maximum with GLOP's `RunningMax` replacement order.
/// Recompute only when the evicted entry was the maximum and its replacement
/// is smaller; the common update and current-max query are constant time.
#[derive(Debug)]
struct RunningMax {
    values: Vec<f64>,
    last_index: usize,
    max_index: usize,
}

impl RunningMax {
    fn new() -> Self {
        Self {
            values: Vec::with_capacity(HISTORY_SIZE),
            last_index: 0,
            max_index: 0,
        }
    }

    fn add(&mut self, value: f64) {
        if self.values.len() < HISTORY_SIZE {
            if self.values.is_empty() || value >= self.current_max() {
                self.max_index = self.values.len();
            }
            self.values.push(value);
            return;
        }
        if value >= self.current_max() {
            self.max_index = self.last_index;
            self.values[self.last_index] = value;
        } else {
            self.values[self.last_index] = value;
            if self.last_index == self.max_index {
                self.max_index = 0;
                for index in 1..self.values.len() {
                    if self.values[index] > self.values[self.max_index] {
                        self.max_index = index;
                    }
                }
            }
        }
        self.last_index += 1;
        if self.last_index == HISTORY_SIZE {
            self.last_index = 0;
        }
    }

    fn current_max(&self) -> f64 {
        self.values[self.max_index]
    }

    fn clear(&mut self) {
        self.values.clear();
        self.last_index = 0;
        self.max_index = 0;
    }
}

#[derive(Debug)]
pub struct TimeLimit {
    start: Instant,
    last: Instant,
    wall_limit_seconds: f64,
    deterministic_limit: f64,
    elapsed_deterministic_time: f64,
    recent_intervals: RunningMax,
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
            recent_intervals: RunningMax::new(),
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
        self.recent_intervals.add(interval);
        // GLOP compares the same sampled clock value with its limit; do not
        // take a second clock reading through elapsed_time() on every pivot.
        if now.duration_since(self.start).as_secs_f64() + self.recent_intervals.current_max()
            >= self.wall_limit_seconds
        {
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
    #[inline]
    pub fn deterministic_time_left(&self) -> f64 {
        (self.deterministic_limit - self.elapsed_deterministic_time).max(0.0)
    }
    #[must_use]
    pub fn elapsed_time(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    #[must_use]
    #[inline]
    pub const fn elapsed_deterministic_time(&self) -> f64 {
        self.elapsed_deterministic_time
    }
    #[must_use]
    #[inline]
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
    use super::{HISTORY_SIZE, RunningMax, TimeLimit};

    #[test]
    #[allow(clippy::float_cmp)] // These are exactly represented small integers.
    fn running_max_matches_brute_force_through_wrap_and_repeated_peaks() {
        let mut running = RunningMax::new();
        let mut history = Vec::new();
        for index in 0..(HISTORY_SIZE * 6) {
            let value = if index % 7 == 0 {
                100.0
            } else {
                f64::from(u32::try_from(index % 37).unwrap())
            };
            running.add(value);
            history.push(value);
            if history.len() > HISTORY_SIZE {
                history.remove(0);
            }
            assert_eq!(
                running.current_max(),
                history.iter().copied().fold(0.0, f64::max)
            );
        }
        running.clear();
        running.add(3.0);
        assert_eq!(running.current_max(), 3.0);
    }
    #[test]
    fn deterministic_limit_is_inclusive() {
        let mut limit = TimeLimit::new(f64::INFINITY, 1.0);
        limit.advance_deterministic_time(1.0);
        assert!(limit.limit_reached());
        limit.change_deterministic_limit(2.0);
        assert!(!limit.limit_reached());
    }
}
