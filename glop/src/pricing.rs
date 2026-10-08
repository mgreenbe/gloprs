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

use crate::random::SharedRandom;

const TOP_K: usize = 31;

#[derive(Clone, Copy, Debug)]
struct HeapElement {
    index: usize,
    value: f64,
}

#[derive(Clone, Debug)]
pub struct DynamicMaximum {
    values: Vec<f64>,
    is_candidate: Vec<bool>,
    threshold: f64,
    tops: Vec<HeapElement>,
    equivalent_choices: Vec<usize>,
    random: SharedRandom,
}

impl Default for DynamicMaximum {
    fn default() -> Self {
        Self::new(1)
    }
}

impl DynamicMaximum {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::new_with_random(SharedRandom::new(seed))
    }

    #[must_use]
    pub fn new_with_random(random: SharedRandom) -> Self {
        Self {
            values: Vec::new(),
            is_candidate: Vec::new(),
            threshold: f64::NEG_INFINITY,
            tops: Vec::new(),
            equivalent_choices: Vec::new(),
            random,
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
            let mut new_size = 0;
            for old_position in 0..self.tops.len() {
                let element = self.tops[old_position];
                if self.is_candidate[element.index] && self.values[element.index] == element.value {
                    self.tops[new_size] = element;
                    new_size += 1;
                    if element.value >= best_value {
                        if element.value == best_value {
                            self.equivalent_choices.push(element.index);
                        } else {
                            self.equivalent_choices.clear();
                            best_value = element.value;
                            best_position = Some(element.index);
                        }
                    }
                }
            }
            self.tops.truncate(new_size);
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
        let choice = self.random.uniform_index(self.equivalent_choices.len());
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
                self.make_heap();
                self.threshold = self.tops[0].value;
            }
            return;
        }
        if value == self.tops[0].value {
            if self.random.bernoulli_half() {
                self.tops[0].index = position;
            }
            return;
        }
        let mut heap_position = 0;
        while heap_position < TOP_K / 2 {
            let left = 2 * heap_position + 1;
            let right = left + 1;
            let next = if self.tops[left].value > self.tops[right].value {
                if value <= self.tops[right].value {
                    break;
                }
                right
            } else {
                if value <= self.tops[left].value {
                    break;
                }
                left
            };
            self.tops[heap_position] = self.tops[next];
            heap_position = next;
        }
        self.tops[heap_position] = HeapElement {
            index: position,
            value,
        };
        self.threshold = self.tops[0].value;
    }

    /// Reproduces libc++'s `std::make_heap()` ordering. Equal-valued elements
    /// are intentionally not ordered by index because the vector order affects
    /// both later tie draws and GLOP's shared random stream.
    fn make_heap(&mut self) {
        for start in (0..=(self.tops.len() - 2) / 2).rev() {
            let mut child = 2 * start + 1;
            if child + 1 < self.tops.len() && self.tops[child].value > self.tops[child + 1].value {
                child += 1;
            }
            if self.tops[child].value > self.tops[start].value {
                continue;
            }
            let top = self.tops[start];
            let mut hole = start;
            loop {
                self.tops[hole] = self.tops[child];
                hole = child;
                if (self.tops.len() - 2) / 2 < child {
                    break;
                }
                child = 2 * child + 1;
                if child + 1 < self.tops.len()
                    && self.tops[child].value > self.tops[child + 1].value
                {
                    child += 1;
                }
                if self.tops[child].value > top.value {
                    break;
                }
            }
            self.tops[hole] = top;
        }
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

    #[test]
    fn tied_heap_sequence_matches_glop() {
        let mut prices = DynamicMaximum::new(1);
        prices.clear_and_resize(100);
        for index in 0..100 {
            prices.add_or_update(index, (index % 7) as f64);
        }
        let expected = [27, 76, 90, 97, 41, 20, 69, 13, 62, 55];
        for index in expected {
            assert_eq!(prices.get_maximum(), Some(index));
            prices.remove(index);
        }
    }
}
