//! Dynamic top-k pricing candidates.
//!
//! Direct port of the algorithm in `ortools/glop/pricing.h`: updates are
//! constant-time, while `get_maximum()` usually hits a lazily maintained
//! top-31 heap and falls back to a dense candidate scan when necessary.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOP_K: usize = 31;

#[derive(Clone, Copy, Debug)]
struct HeapElement {
    index: usize,
    value: f64,
}

impl PartialEq for HeapElement {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.index == other.index
    }
}

impl Eq for HeapElement {}

// Reverse the value ordering: BinaryHeap's root is GLOP's minimum threshold.
impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .total_cmp(&self.value)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct DynamicMaximum {
    values: Vec<f64>,
    is_candidate: Vec<bool>,
    threshold: f64,
    tops: BinaryHeap<HeapElement>,
    equivalent_choices: Vec<usize>,
    random: StdRng,
}

impl Default for DynamicMaximum {
    fn default() -> Self {
        Self::new(1)
    }
}

impl DynamicMaximum {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            values: Vec::new(),
            is_candidate: Vec::new(),
            threshold: f64::NEG_INFINITY,
            tops: BinaryHeap::new(),
            equivalent_choices: Vec::new(),
            random: StdRng::seed_from_u64(seed),
        }
    }

    pub fn clear_and_resize(&mut self, size: usize) {
        self.tops.clear();
        self.threshold = f64::NEG_INFINITY;
        self.values.resize(size, 0.0);
        self.is_candidate.clear();
        self.is_candidate.resize(size, false);
    }

    pub fn clear(&mut self) {
        self.clear_and_resize(0);
    }

    #[must_use]
    pub const fn size(&self) -> usize {
        self.values.len()
    }

    pub fn remove(&mut self, position: usize) {
        self.is_candidate[position] = false;
    }

    pub fn start_dense_updates(&mut self) {
        self.tops.clear();
        self.threshold = f64::INFINITY;
    }

    pub fn dense_add_or_update(&mut self, position: usize, value: f64) {
        debug_assert!(!value.is_nan());
        debug_assert!(self.tops.is_empty());
        self.is_candidate[position] = true;
        self.values[position] = value;
    }

    pub fn add_or_update(&mut self, position: usize, value: f64) {
        debug_assert!(!value.is_nan());
        self.is_candidate[position] = true;
        self.values[position] = value;
        if value >= self.threshold {
            self.update_top_k(position, value);
        }
    }

    pub fn get_maximum(&mut self) -> Option<usize> {
        let mut best_value = f64::NEG_INFINITY;
        let mut best_position = None;
        self.equivalent_choices.clear();

        if !self.tops.is_empty() {
            let mut valid = BinaryHeap::new();
            for element in self.tops.drain() {
                if self.is_candidate[element.index] && self.values[element.index] == element.value {
                    if element.value >= best_value {
                        if element.value == best_value {
                            self.equivalent_choices.push(element.index);
                        } else {
                            self.equivalent_choices.clear();
                            best_value = element.value;
                            best_position = Some(element.index);
                        }
                    }
                    valid.push(element);
                }
            }
            self.tops = valid;
            if !self.tops.is_empty() {
                return self.randomize_if_many_choices(best_position);
            }
        }

        self.threshold = f64::NEG_INFINITY;
        for position in 0..self.values.len() {
            if !self.is_candidate[position] {
                continue;
            }
            let value = self.values[position];
            if value < self.threshold {
                continue;
            }
            self.update_top_k(position, value);
            if value >= best_value {
                if value == best_value {
                    self.equivalent_choices.push(position);
                } else {
                    self.equivalent_choices.clear();
                    best_value = value;
                    best_position = Some(position);
                }
            }
        }
        self.randomize_if_many_choices(best_position)
    }

    fn randomize_if_many_choices(&mut self, best: Option<usize>) -> Option<usize> {
        if self.equivalent_choices.is_empty() {
            return best;
        }
        if let Some(best) = best {
            self.equivalent_choices.push(best);
        }
        let choice = self.random.random_range(0..self.equivalent_choices.len());
        Some(self.equivalent_choices[choice])
    }

    fn update_top_k(&mut self, position: usize, value: f64) {
        debug_assert!(value >= self.threshold);
        if self.tops.len() < TOP_K {
            self.tops.push(HeapElement {
                index: position,
                value,
            });
            if self.tops.len() == TOP_K {
                self.threshold = self
                    .tops
                    .peek()
                    .map_or(f64::NEG_INFINITY, |entry| entry.value);
            }
            return;
        }
        if value == self.tops.peek().expect("full top-k heap").value {
            if self.random.random_bool(0.5) {
                self.tops.pop();
                self.tops.push(HeapElement {
                    index: position,
                    value,
                });
            }
            return;
        }
        self.tops.pop();
        self.tops.push(HeapElement {
            index: position,
            value,
        });
        self.threshold = self.tops.peek().expect("nonempty top-k heap").value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maintains_maximum_across_sparse_dense_and_stale_updates() {
        let mut prices = DynamicMaximum::new(7);
        prices.clear_and_resize(100);
        for index in 0..100 {
            prices.add_or_update(index, index as f64);
        }
        assert_eq!(prices.get_maximum(), Some(99));
        prices.remove(99);
        prices.add_or_update(98, -1.0);
        assert_eq!(prices.get_maximum(), Some(97));
        prices.start_dense_updates();
        for index in 0..100 {
            prices.dense_add_or_update(index, -(index as f64));
        }
        assert_eq!(prices.get_maximum(), Some(0));
    }
}
