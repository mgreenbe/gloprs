//! Common scalar, index, status, and dense-vector types used by GLOP.
//!
//! This is the Rust counterpart of upstream `ortools/lp_data/lp_types.{h,cc}`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Scalar used for all numerical LP computations.
pub type Fractional = f64;

pub const RANGE_MAX: Fractional = Fractional::MAX;
pub const INFINITY: Fractional = Fractional::INFINITY;
pub const EPSILON: Fractional = Fractional::EPSILON;

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

/// A vector that can only be indexed with its logical LP index type.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TypedVec<I, T> {
    values: Vec<T>,
    index: PhantomData<fn(I) -> I>,
}

pub trait VectorIndex: Copy {
    fn to_usize(self) -> usize;
    fn from_usize(value: usize) -> Self;
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

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.values
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

pub type DenseRow = TypedVec<ColIndex, Fractional>;
pub type DenseBooleanRow = TypedVec<ColIndex, bool>;
pub type ColMapping = TypedVec<ColIndex, ColIndex>;
pub type ColToRowMapping = TypedVec<ColIndex, RowIndex>;
pub type VariableTypeRow = TypedVec<ColIndex, VariableType>;
pub type VariableStatusRow = TypedVec<ColIndex, VariableStatus>;

pub type DenseColumn = TypedVec<RowIndex, Fractional>;
pub type DenseBooleanColumn = TypedVec<RowIndex, bool>;
pub type RowMapping = TypedVec<RowIndex, RowIndex>;
pub type RowToColMapping = TypedVec<RowIndex, ColIndex>;
pub type ConstraintStatusColumn = TypedVec<RowIndex, ConstraintStatus>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
pub enum VariableType {
    Unconstrained,
    LowerBounded,
    UpperBounded,
    UpperAndLowerBounded,
    FixedVariable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableStatus {
    Basic,
    FixedValue,
    AtLowerBound,
    AtUpperBound,
    Free,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
}
