//! Row-specialized sparse vectors from upstream `sparse_row.h`.

use std::ops::{Deref, DerefMut};

use crate::lp_types::{ColIndex, RowIndex, TypedVec};
use crate::permutation::ColumnPermutation;
use crate::sparse_vector::{SparseEntry, SparseVector, SparseVectorIter};

/// A sparse vector indexed by columns, with row-oriented method names matching
/// upstream `SparseRow`.
#[derive(Clone, Debug, Default)]
pub struct SparseRow {
    vector: SparseVector<ColIndex>,
}

impl SparseRow {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vector: SparseVector::new(),
        }
    }

    #[must_use]
    pub fn entry_col(&self, position: usize) -> ColIndex {
        self.vector.entry(position).index()
    }

    #[must_use]
    pub fn entry_coefficient(&self, position: usize) -> f64 {
        self.vector.entry(position).coefficient()
    }

    /// # Panics
    ///
    /// Panics if the row is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn first_col(&self) -> ColIndex {
        self.vector.first().unwrap().index()
    }

    /// # Panics
    ///
    /// Panics if the row is empty, matching upstream's unchecked accessor.
    #[must_use]
    pub fn last_col(&self) -> ColIndex {
        self.vector.last().unwrap().index()
    }

    pub fn apply_col_permutation(&mut self, permutation: &ColumnPermutation) {
        self.vector.apply_index_permutation(permutation);
    }

    pub fn apply_partial_col_permutation(&mut self, permutation: &ColumnPermutation) {
        self.vector.apply_partial_index_permutation(permutation);
    }
}

impl Deref for SparseRow {
    type Target = SparseVector<ColIndex>;

    fn deref(&self) -> &Self::Target {
        &self.vector
    }
}

impl DerefMut for SparseRow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vector
    }
}

impl<'a> IntoIterator for &'a SparseRow {
    type Item = SparseEntry<ColIndex>;
    type IntoIter = SparseVectorIter<'a, ColIndex>;

    fn into_iter(self) -> Self::IntoIter {
        self.vector.iter()
    }
}

pub type RowMajorSparseMatrix = TypedVec<RowIndex, SparseRow>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_names_and_permutations_delegate_to_sparse_vector_semantics() {
        let mut row = SparseRow::new();
        row.add_entry(ColIndex::new(1), 2.0);
        row.add_entry(ColIndex::new(3), -4.0);
        assert_eq!(row.entry_col(0), ColIndex::new(1));
        assert_eq!(row.entry_coefficient(1).to_bits(), (-4.0_f64).to_bits());
        assert_eq!(row.first_col(), ColIndex::new(1));
        assert_eq!(row.last_col(), ColIndex::new(3));

        let partial = ColumnPermutation::from_vec(vec![
            ColIndex::new(-1),
            ColIndex::new(2),
            ColIndex::new(-1),
            ColIndex::new(0),
        ]);
        row.apply_partial_col_permutation(&partial);
        assert_eq!(
            (&row)
                .into_iter()
                .map(|entry| (entry.index(), entry.coefficient().to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (ColIndex::new(2), 2.0_f64.to_bits()),
                (ColIndex::new(0), (-4.0_f64).to_bits())
            ]
        );
    }
}
