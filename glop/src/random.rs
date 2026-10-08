//! Deterministic random stream used by GLOP.
//!
//! Upstream uses one shared `std::mt19937_64` engine for all randomized
//! simplex decisions.  Sharing is behaviorally significant: a draw made by
//! pricing changes the next draw seen by a ratio test or cost perturbation.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::cell::RefCell;
use std::rc::Rc;

const STATE_SIZE: usize = 312;
const SHIFT_SIZE: usize = 156;
const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
const LOWER_MASK: u64 = (1_u64 << 31) - 1;
const UPPER_MASK: u64 = !LOWER_MASK;

#[derive(Clone, Debug)]
pub struct SharedRandom(Rc<RefCell<Mt19937_64>>);

impl SharedRandom {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(Rc::new(RefCell::new(Mt19937_64::new(seed))))
    }

    pub fn seed(&self, seed: u64) {
        *self.0.borrow_mut() = Mt19937_64::new(seed);
    }

    /// Matches libc++'s `uniform_int_distribution<int>(0, size - 1)` for a
    /// full-range 64-bit engine. It samples the required low bits and rejects
    /// values outside the requested range.
    ///
    /// # Panics
    ///
    /// Panics when `size` is zero or does not fit GLOP's 32-bit index domain.
    #[must_use]
    pub fn uniform_index(&self, size: usize) -> usize {
        assert!(size > 0);
        if size == 1 {
            return 0;
        }
        assert!(u32::try_from(size).is_ok());
        let range = u32::try_from(size).expect("checked above");
        let width = u32::BITS - (range - 1).leading_zeros();
        let mask = if width == u32::BITS {
            u32::MAX
        } else {
            (1_u32 << width) - 1
        };
        loop {
            let value = self.next_u64() as u32 & mask;
            if value < range {
                return value as usize;
            }
        }
    }

    /// Matches `absl::Bernoulli(random, 0.5)`. For `mt19937_64`, Abseil's
    /// `FastUniformBits<uint32_t>` consumes the low 32 bits of one output.
    #[must_use]
    pub fn bernoulli_half(&self) -> bool {
        (self.next_u64() as u32) < (1_u32 << 31)
    }

    /// Matches libc++'s default `uniform_real_distribution<double>` for
    /// `mt19937_64`: `generate_canonical` consumes one full-width output.
    #[must_use]
    pub fn uniform_unit_f64(&self) -> f64 {
        self.next_u64() as f64 / 18_446_744_073_709_551_616.0
    }

    fn next_u64(&self) -> u64 {
        self.0.borrow_mut().next_u64()
    }
}

#[derive(Debug)]
struct Mt19937_64 {
    state: [u64; STATE_SIZE],
    index: usize,
}

impl Mt19937_64 {
    fn new(seed: u64) -> Self {
        let mut state = [0; STATE_SIZE];
        state[0] = seed;
        for index in 1..STATE_SIZE {
            state[index] = 6_364_136_223_846_793_005_u64
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 62))
                .wrapping_add(index as u64);
        }
        Self {
            state,
            index: STATE_SIZE,
        }
    }

    fn next_u64(&mut self) -> u64 {
        if self.index == STATE_SIZE {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= (value >> 29) & 0x5555_5555_5555_5555;
        value ^= (value << 17) & 0x71D6_7FFF_EDA6_0000;
        value ^= (value << 37) & 0xFFF7_EEE0_0000_0000;
        value ^ (value >> 43)
    }

    fn twist(&mut self) {
        for index in 0..STATE_SIZE {
            let value = (self.state[index] & UPPER_MASK)
                | (self.state[(index + 1) % STATE_SIZE] & LOWER_MASK);
            self.state[index] = self.state[(index + SHIFT_SIZE) % STATE_SIZE]
                ^ (value >> 1)
                ^ if value & 1 == 0 { 0 } else { MATRIX_A };
        }
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_standard_mt19937_64_sequence() {
        let random = SharedRandom::new(5489);
        assert_eq!(random.next_u64(), 14_514_284_786_278_117_030);
        assert_eq!(random.next_u64(), 4_620_546_740_167_642_908);
        assert_eq!(random.next_u64(), 13_109_570_281_517_897_720);
    }

    #[test]
    fn clones_share_one_stream() {
        let first = SharedRandom::new(1);
        let second = first.clone();
        assert_eq!(first.next_u64(), 2_469_588_189_546_311_528);
        assert_eq!(second.next_u64(), 2_516_265_689_700_432_462);
    }

    #[test]
    fn matches_libcxx_and_abseil_distribution_sequence() {
        let random = SharedRandom::new(1);
        assert_eq!(random.uniform_index(3), 0);
        assert_eq!(random.uniform_index(31), 14);
        assert!(random.bernoulli_half());
        assert_eq!(
            random.uniform_unit_f64().to_bits(),
            0.021_024_228_416_727_027_f64.to_bits()
        );
        assert_eq!(random.uniform_index(3), 0);
    }
}
