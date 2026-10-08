//! Common scalar, index, status, and dense-vector types used by GLOP.
//!
//! This is the Rust counterpart of upstream `ortools/lp_data/lp_types.{h,cc}`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Scalar used for all numerical LP computations.
pub type Fractional = f64;
pub type GlopIndex = i32;

pub const RANGE_MAX: Fractional = Fractional::MAX;
pub const INFINITY: Fractional = Fractional::INFINITY;
pub const EPSILON: Fractional = Fractional::EPSILON;

#[must_use]
pub const fn to_double(value: Fractional) -> f64 {
    value
}

#[must_use]
pub fn is_finite(value: Fractional) -> bool {
    value.is_finite()
}

macro_rules! strong_index {
    ($name:ident, $storage:ty) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name($storage);

        impl $name {
            #[must_use]
            pub const fn new(value: $storage) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn value(self) -> $storage {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

strong_index!(ColIndex, i32);
strong_index!(RowIndex, i32);
strong_index!(EntryIndex, i64);

pub const INVALID_ROW: RowIndex = RowIndex::new(-1);
pub const INVALID_COL: ColIndex = ColIndex::new(-1);

#[must_use]
pub const fn row_to_col_index(row: RowIndex) -> ColIndex {
    ColIndex::new(row.value())
}

#[must_use]
pub const fn col_to_row_index(column: ColIndex) -> RowIndex {
    RowIndex::new(column.value())
}

#[must_use]
pub const fn col_to_int_index(column: ColIndex) -> GlopIndex {
    column.value()
}

#[must_use]
pub const fn row_to_int_index(row: RowIndex) -> GlopIndex {
    row.value()
}

/// A vector that can only be indexed with its logical LP index type.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TypedVec<I, T> {
    values: Vec<T>,
    index: PhantomData<fn(I) -> I>,
}

pub trait VectorIndex: Copy {
    fn to_usize(self) -> usize;
    fn from_usize(value: usize) -> Self;
    fn value_i64(self) -> i64;
}

/// Borrowed counterpart of upstream `StrictITISpan`.
#[derive(Clone, Copy, Debug)]
pub struct TypedSlice<'a, I, T> {
    values: &'a [T],
    index: PhantomData<fn(I) -> I>,
}

impl<'a, I: VectorIndex, T> TypedSlice<'a, I, T> {
    #[must_use]
    pub fn new(values: &'a [T]) -> Self {
        let _ = I::from_usize(values.len());
        Self {
            values,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn len(self) -> I {
        I::from_usize(self.values.len())
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub const fn as_slice(self) -> &'a [T] {
        self.values
    }

    pub fn iter(self) -> std::slice::Iter<'a, T> {
        self.values.iter()
    }
}

impl<I: VectorIndex, T> Index<I> for TypedSlice<'_, I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index.to_usize()]
    }
}

impl<'a, I: VectorIndex, T> IntoIterator for TypedSlice<'a, I, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

#[derive(Debug)]
pub struct TypedSliceMut<'a, I, T> {
    values: &'a mut [T],
    index: PhantomData<fn(I) -> I>,
}

impl<'a, I: VectorIndex, T> TypedSliceMut<'a, I, T> {
    #[must_use]
    pub fn new(values: &'a mut [T]) -> Self {
        let _ = I::from_usize(values.len());
        Self {
            values,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        I::from_usize(self.values.len())
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.values
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.values.iter_mut()
    }
}

impl<I: VectorIndex, T> Index<I> for TypedSliceMut<'_, I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index.to_usize()]
    }
}

impl<I: VectorIndex, T> IndexMut<I> for TypedSliceMut<'_, I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index.to_usize()]
    }
}

impl<'b, I: VectorIndex, T> IntoIterator for &'b TypedSliceMut<'_, I, T> {
    type Item = &'b T;
    type IntoIter = std::slice::Iter<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'b, I: VectorIndex, T> IntoIterator for &'b mut TypedSliceMut<'_, I, T> {
    type Item = &'b mut T;
    type IntoIter = std::slice::IterMut<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

macro_rules! vector_index {
    ($name:ident) => {
        impl VectorIndex for $name {
            fn to_usize(self) -> usize {
                usize::try_from(self.value()).expect("vector index must be nonnegative")
            }

            fn from_usize(value: usize) -> Self {
                Self::new(i32::try_from(value).expect("vector is too large for its index type"))
            }

            fn value_i64(self) -> i64 {
                i64::from(self.value())
            }
        }
    };
}

vector_index!(ColIndex);
vector_index!(RowIndex);

impl<I: VectorIndex, T> TypedVec<I, T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn from_vec(values: Vec<T>) -> Self {
        let _ = I::from_usize(values.len());
        Self {
            values,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        I::from_usize(self.values.len())
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn push(&mut self, value: T) {
        let _ = I::from_usize(self.values.len().saturating_add(1));
        self.values.push(value);
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    #[must_use]
    pub fn capacity(&self) -> I {
        I::from_usize(self.values.capacity())
    }

    pub fn reserve(&mut self, capacity: I) {
        let requested = capacity.to_usize();
        if self.values.capacity() < requested {
            self.values.reserve(requested - self.values.len());
        }
    }

    pub fn truncate(&mut self, size: I) {
        self.values.truncate(size.to_usize());
    }

    /// # Panics
    ///
    /// Panics if `size` is larger than the current logical size.
    pub fn resize_down(&mut self, size: I) {
        assert!(size.to_usize() <= self.values.len());
        self.values.truncate(size.to_usize());
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    #[must_use]
    pub fn view(&self) -> TypedSlice<'_, I, T> {
        TypedSlice::new(&self.values)
    }

    #[must_use]
    pub fn view_mut(&mut self) -> TypedSliceMut<'_, I, T> {
        TypedSliceMut::new(&mut self.values)
    }

    #[must_use]
    pub fn back(&self) -> Option<&T> {
        self.values.last()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
}

impl<'a, I: VectorIndex, T> IntoIterator for &'a TypedVec<I, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<I: VectorIndex, T: Clone> TypedVec<I, T> {
    #[must_use]
    pub fn filled(size: I, value: T) -> Self {
        Self::from_vec(vec![value; size.to_usize()])
    }

    pub fn resize(&mut self, size: I, value: T) {
        self.values.resize(size.to_usize(), value);
    }

    pub fn assign(&mut self, size: I, value: T) {
        self.values.clear();
        self.values.resize(size.to_usize(), value);
    }

    pub fn assign_from_view(&mut self, view: TypedSlice<'_, I, T>) {
        self.values.clear();
        self.values.extend_from_slice(view.as_slice());
    }
}

impl<I: VectorIndex, T: Default + Clone> TypedVec<I, T> {
    pub fn assign_to_zero(&mut self, size: I) {
        self.values.clear();
        self.values.resize(size.to_usize(), T::default());
    }
}

impl<I: VectorIndex, T> Index<I> for TypedVec<I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index.to_usize()]
    }
}

impl<I: VectorIndex, T> IndexMut<I> for TypedVec<I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index.to_usize()]
    }
}

impl<I: VectorIndex, T> FromIterator<T> for TypedVec<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

/// A compact typed bit vector used for row and column masks.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BitVec<I> {
    words: Vec<u64>,
    len: usize,
    index: PhantomData<fn(I) -> I>,
}

#[derive(Clone, Debug)]
pub struct BitOnesIter<'a, I> {
    words: &'a [u64],
    word_index: usize,
    current: u64,
    index: PhantomData<fn(I) -> I>,
}

#[derive(Clone, Copy, Debug)]
pub struct BitSlice<'a, I> {
    words: &'a [u64],
    index: PhantomData<fn(I) -> I>,
}

impl<I: VectorIndex> BitSlice<'_, I> {
    #[must_use]
    pub fn contains(self, index: I) -> bool {
        let position = index.to_usize();
        self.words[position / 64] & (1_u64 << (position % 64)) != 0
    }
}

#[derive(Debug)]
pub struct BitSliceMut<'a, I> {
    words: &'a mut [u64],
    index: PhantomData<fn(I) -> I>,
}

impl<I: VectorIndex> BitSliceMut<'_, I> {
    #[must_use]
    pub fn contains(&self, index: I) -> bool {
        let position = index.to_usize();
        self.words[position / 64] & (1_u64 << (position % 64)) != 0
    }

    pub fn set(&mut self, index: I) {
        let position = index.to_usize();
        self.words[position / 64] |= 1_u64 << (position % 64);
    }

    pub fn clear(&mut self, index: I) {
        let position = index.to_usize();
        self.words[position / 64] &= !(1_u64 << (position % 64));
    }
}

impl<I: VectorIndex> BitOnesIter<'_, I> {
    fn new(words: &[u64]) -> BitOnesIter<'_, I> {
        BitOnesIter {
            words,
            word_index: 0,
            current: words.first().copied().unwrap_or(0),
            index: PhantomData,
        }
    }
}

impl<I: VectorIndex> Iterator for BitOnesIter<'_, I> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                return Some(I::from_usize(64 * self.word_index + bit));
            }
            self.word_index += 1;
            self.current = *self.words.get(self.word_index)?;
        }
    }
}

impl<I: VectorIndex> BitVec<I> {
    #[must_use]
    pub fn new(size: I) -> Self {
        let len = size.to_usize();
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn len(&self) -> I {
        I::from_usize(self.len)
    }

    #[must_use]
    pub fn view(&self) -> BitSlice<'_, I> {
        BitSlice {
            words: &self.words,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn view_mut(&mut self) -> BitSliceMut<'_, I> {
        BitSliceMut {
            words: &mut self.words,
            index: PhantomData,
        }
    }

    #[must_use]
    pub fn contains(&self, index: I) -> bool {
        let position = index.to_usize();
        self.words[position / 64] & (1_u64 << (position % 64)) != 0
    }

    pub fn set(&mut self, index: I) {
        let position = index.to_usize();
        self.words[position / 64] |= 1_u64 << (position % 64);
    }

    pub fn set_to(&mut self, index: I, value: bool) {
        if value {
            self.set(index);
        } else {
            self.clear_bit(index);
        }
    }

    pub fn push(&mut self, value: bool) {
        let position = self.len;
        self.len += 1;
        self.words.resize(self.len.div_ceil(64), 0);
        self.set_to(I::from_usize(position), value);
    }

    pub fn clear_bit(&mut self, index: I) {
        let position = index.to_usize();
        self.words[position / 64] &= !(1_u64 << (position % 64));
    }

    pub fn clear_bucket(&mut self, index: I) {
        self.words[index.to_usize() / 64] = 0;
    }

    pub fn clear_two_bits(&mut self, index: I) {
        let position = index.to_usize();
        let first = position & !1;
        self.words[first / 64] &= !(3_u64 << (first % 64));
    }

    #[must_use]
    pub fn are_one_of_two_bits_set(&self, index: I) -> bool {
        let position = index.to_usize();
        let first = position & !1;
        self.words[first / 64] & (3_u64 << (first % 64)) != 0
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    pub fn resize(&mut self, size: I) {
        self.len = size.to_usize();
        self.words.resize(self.len.div_ceil(64), 0);
        if !self.len.is_multiple_of(64)
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1_u64 << (self.len % 64)) - 1;
        }
    }

    pub fn clear_and_resize(&mut self, size: I) {
        self.len = size.to_usize();
        self.words.clear();
        self.words.resize(self.len.div_ceil(64), 0);
    }

    pub fn intersection(&mut self, other: &Self) {
        let common = self.words.len().min(other.words.len());
        for position in 0..common {
            self.words[position] &= other.words[position];
        }
        self.words[common..].fill(0);
    }

    pub fn union(&mut self, other: &Self) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left |= *right;
        }
    }

    pub fn set_to_intersection_of(&mut self, left: &Self, right: &Self) {
        debug_assert_eq!(left.len, right.len);
        self.resize(I::from_usize(left.len));
        for position in 0..left.words.len() {
            self.words[position] = left.words[position] & right.words[position];
        }
    }

    pub fn copy_bucket_from(&mut self, other: &Self, index: I) {
        let bucket = index.to_usize() / 64;
        self.words[bucket] = other.words[bucket];
    }

    /// Copies the overlapping content without resizing. Bits above a shorter
    /// source's logical end remain unchanged, matching GLOP's `Bitset64`.
    pub fn set_content_from(&mut self, other: &Self) {
        let common = self.words.len().min(other.words.len());
        if common == 0 {
            return;
        }
        let saved_last = self.words[common - 1];
        self.words[..common].copy_from_slice(&other.words[..common]);
        if self.words.len() >= other.words.len() && !other.len.is_multiple_of(64) {
            let low_bits = (1_u64 << (other.len % 64)) - 1;
            self.words[common - 1] = (self.words[common - 1] & low_bits) | (saved_last & !low_bits);
        }
    }

    pub fn set_content_from_same_size(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        self.words.copy_from_slice(&other.words);
    }

    #[must_use]
    pub fn iter_ones(&self) -> BitOnesIter<'_, I> {
        BitOnesIter::new(&self.words)
    }

    #[must_use]
    pub fn is_all_false(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    #[must_use]
    pub fn debug_string(&self) -> String {
        (0..self.len)
            .map(|position| {
                if self.words[position / 64] & (1_u64 << (position % 64)) != 0 {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }
}

pub type RowBitVec = BitVec<RowIndex>;
pub type ColBitVec = BitVec<ColIndex>;

pub type DenseRow = TypedVec<ColIndex, Fractional>;
pub type DenseBooleanRow = TypedVec<ColIndex, bool>;
pub type ColMapping = TypedVec<ColIndex, ColIndex>;
pub type ColIndexVector = Vec<ColIndex>;
pub type RowIndexVector = Vec<RowIndex>;
pub type ColToRowMapping = TypedVec<ColIndex, RowIndex>;
pub type VariableTypeRow = TypedVec<ColIndex, VariableType>;
pub type VariableStatusRow = TypedVec<ColIndex, VariableStatus>;

pub type DenseColumn = TypedVec<RowIndex, Fractional>;
pub type DenseBooleanColumn = TypedVec<RowIndex, bool>;
pub type RowMapping = TypedVec<RowIndex, RowIndex>;
pub type RowToColMapping = TypedVec<RowIndex, ColIndex>;
pub type ConstraintStatusColumn = TypedVec<RowIndex, ConstraintStatus>;
pub type DenseBitRow = BitVec<ColIndex>;
pub type DenseBitColumn = BitVec<RowIndex>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i8)]
pub enum ProblemStatus {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    InfeasibleOrUnbounded,
    PrimalUnbounded,
    DualUnbounded,
    Init,
    PrimalFeasible,
    DualFeasible,
    Abnormal,
    InvalidProblem,
    Imprecise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i8)]
pub enum VariableType {
    Unconstrained,
    LowerBounded,
    UpperBounded,
    UpperAndLowerBounded,
    FixedVariable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i8)]
pub enum VariableStatus {
    Basic,
    FixedValue,
    AtLowerBound,
    AtUpperBound,
    Free,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i8)]
pub enum ConstraintStatus {
    Basic,
    FixedValue,
    AtLowerBound,
    AtUpperBound,
    Free,
}

macro_rules! display_as_upstream_name {
    ($type:ty, {$($variant:ident => $name:literal),+ $(,)?}) => {
        impl fmt::Display for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                let name = match self {
                    $(Self::$variant => $name),+
                };
                formatter.write_str(name)
            }
        }
    };
}

display_as_upstream_name!(ProblemStatus, {
    Optimal => "OPTIMAL",
    PrimalInfeasible => "PRIMAL_INFEASIBLE",
    DualInfeasible => "DUAL_INFEASIBLE",
    InfeasibleOrUnbounded => "INFEASIBLE_OR_UNBOUNDED",
    PrimalUnbounded => "PRIMAL_UNBOUNDED",
    DualUnbounded => "DUAL_UNBOUNDED",
    Init => "INIT",
    PrimalFeasible => "PRIMAL_FEASIBLE",
    DualFeasible => "DUAL_FEASIBLE",
    Abnormal => "ABNORMAL",
    InvalidProblem => "INVALID_PROBLEM",
    Imprecise => "IMPRECISE",
});

display_as_upstream_name!(VariableType, {
    Unconstrained => "UNCONSTRAINED",
    LowerBounded => "LOWER_BOUNDED",
    UpperBounded => "UPPER_BOUNDED",
    UpperAndLowerBounded => "UPPER_AND_LOWER_BOUNDED",
    FixedVariable => "FIXED_VARIABLE",
});

display_as_upstream_name!(VariableStatus, {
    Basic => "BASIC",
    FixedValue => "FIXED_VALUE",
    AtLowerBound => "AT_LOWER_BOUND",
    AtUpperBound => "AT_UPPER_BOUND",
    Free => "FREE",
});

display_as_upstream_name!(ConstraintStatus, {
    Basic => "BASIC",
    FixedValue => "FIXED_VALUE",
    AtLowerBound => "AT_LOWER_BOUND",
    AtUpperBound => "AT_UPPER_BOUND",
    Free => "FREE",
});

impl From<VariableStatus> for ConstraintStatus {
    fn from(status: VariableStatus) -> Self {
        match status {
            VariableStatus::Basic => Self::Basic,
            VariableStatus::FixedValue => Self::FixedValue,
            VariableStatus::AtLowerBound => Self::AtLowerBound,
            VariableStatus::AtUpperBound => Self::AtUpperBound,
            VariableStatus::Free => Self::Free,
        }
    }
}

/// Converts a count of floating-point operations to deterministic seconds.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn deterministic_time_for_fp_operations(operations: i64) -> f64 {
    2e-9 * operations as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_vectors_use_logical_indices() {
        let mut row = DenseRow::filled(ColIndex::new(3), 0.0);
        row[ColIndex::new(1)] = 2.5;
        assert_eq!(row[ColIndex::new(1)].to_bits(), 2.5_f64.to_bits());
        assert_eq!(row.len(), ColIndex::new(3));

        row.reserve(ColIndex::new(20));
        assert!(row.capacity() >= ColIndex::new(20));
        let view = row.view();
        assert_eq!(view.len(), ColIndex::new(3));
        assert_eq!(view[ColIndex::new(1)].to_bits(), 2.5_f64.to_bits());

        {
            let mut view = row.view_mut();
            view[ColIndex::new(2)] = -4.0;
        }
        let mut copy = DenseRow::new();
        copy.assign_from_view(row.view());
        copy.resize_down(ColIndex::new(2));
        assert_eq!(copy.as_slice(), &[0.0, 2.5]);
        copy.assign_to_zero(ColIndex::new(4));
        assert_eq!(copy.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn status_names_match_upstream() {
        assert_eq!(
            ProblemStatus::PrimalInfeasible.to_string(),
            "PRIMAL_INFEASIBLE"
        );
        assert_eq!(VariableType::FixedVariable.to_string(), "FIXED_VARIABLE");
        assert_eq!(VariableStatus::AtUpperBound.to_string(), "AT_UPPER_BOUND");
        assert_eq!(
            ConstraintStatus::from(VariableStatus::Free),
            ConstraintStatus::Free
        );
    }

    #[test]
    fn scalar_constants_and_deterministic_time_match_upstream() {
        assert!(is_finite(RANGE_MAX));
        assert!(!is_finite(INFINITY));
        assert_eq!(EPSILON.to_bits(), f64::EPSILON.to_bits());
        assert_eq!(
            deterministic_time_for_fp_operations(500_000_000).to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn typed_bit_vectors_resize_and_clear() {
        let mut bits = RowBitVec::new(RowIndex::new(70));
        bits.set(RowIndex::new(65));
        assert!(bits.contains(RowIndex::new(65)));
        bits.clear_bit(RowIndex::new(65));
        assert!(bits.is_all_false());
        bits.resize(RowIndex::new(130));
        bits.set(RowIndex::new(129));
        bits.clear();
        assert!(bits.is_all_false());

        {
            let mut view = bits.view_mut();
            view.set(RowIndex::new(4));
            assert!(view.contains(RowIndex::new(4)));
            view.clear(RowIndex::new(4));
        }
        assert!(!bits.view().contains(RowIndex::new(4)));
    }

    #[test]
    fn bit_vector_bulk_operations_match_bitset64_semantics() {
        let mut left = RowBitVec::new(RowIndex::new(130));
        left.set(RowIndex::new(1));
        left.set(RowIndex::new(65));
        left.set(RowIndex::new(129));
        let mut short = RowBitVec::new(RowIndex::new(66));
        short.set(RowIndex::new(1));
        short.set(RowIndex::new(64));

        let mut intersection = left.clone();
        intersection.intersection(&short);
        assert_eq!(
            intersection.iter_ones().collect::<Vec<_>>(),
            [RowIndex::new(1)]
        );

        let mut copied = left.clone();
        copied.set_content_from(&short);
        assert!(copied.contains(RowIndex::new(1)));
        assert!(copied.contains(RowIndex::new(64)));
        assert!(!copied.contains(RowIndex::new(65)));
        assert!(copied.contains(RowIndex::new(129)));

        short.push(true);
        assert!(short.contains(RowIndex::new(66)));
        short.clear_two_bits(RowIndex::new(65));
        assert!(!short.are_one_of_two_bits_set(RowIndex::new(64)));
        assert_eq!(&short.debug_string()[..2], "01");
    }
}
