//! Bixby, triangular, and Maros initial-basis crash procedures.
//!
//! This is a direct port of `ortools/glop/initial_basis.{h,cc}`. The residual
//! pattern is maintained incrementally, so the triangular and Maros crashes do
//! not repeatedly scan the complete matrix.

#![allow(clippy::float_cmp, clippy::missing_panics_doc)]

use std::cmp::Ordering;

use lp_data::lp_types::{
    ColIndex, DenseRow, INVALID_COL, RowIndex, RowToColMapping, VariableType, VariableTypeRow,
    VectorIndex,
};
use lp_data::sparse::CompactSparseMatrix;

#[derive(Clone, Debug)]
struct MatrixNonZeroPattern {
    row_nonzeros: Vec<Vec<usize>>,
    row_degree: Vec<i32>,
    column_degree: Vec<i32>,
    column_deleted: Vec<bool>,
}

impl MatrixNonZeroPattern {
    fn new(rows: usize, columns: usize) -> Self {
        Self {
            row_nonzeros: vec![Vec::new(); rows],
            row_degree: vec![0; rows],
            column_degree: vec![0; columns],
            column_deleted: vec![false; columns],
        }
    }

    fn add_entry(&mut self, row: usize, column: usize) {
        self.row_nonzeros[row].push(column);
        self.row_degree[row] += 1;
        self.column_degree[column] += 1;
    }

    fn delete_row_and_column(&mut self, row: usize, column: usize) {
        self.column_deleted[column] = true;
        self.row_degree[row] = 0;
    }

    fn row_nonzeros(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        self.row_nonzeros[row]
            .iter()
            .copied()
            .filter(|&column| !self.column_deleted[column])
    }

    fn decrease_column_degree(&mut self, column: usize) {
        self.column_degree[column] -= 1;
    }
}

#[derive(Clone, Copy, Debug)]
struct TriangularCandidate {
    column: usize,
    category: i32,
    entries: usize,
    penalty: f64,
}

impl PartialEq for TriangularCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column
    }
}

impl Eq for TriangularCandidate {}

impl Ord for TriangularCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Upstream's priority queue prefers lower category, then sparser
        // columns, then lower Bixby penalty.
        other
            .category
            .cmp(&self.category)
            .then_with(|| other.entries.cmp(&self.entries))
            .then_with(|| other.penalty.total_cmp(&self.penalty))
    }
}

impl PartialOrd for TriangularCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `std::priority_queue` ordering, including its deterministic behavior when
/// the comparator considers entries equivalent. Rust's `BinaryHeap` makes a
/// different left/right choice on such ties, which changes crash bases.
#[derive(Debug)]
struct CandidateHeap {
    values: Vec<TriangularCandidate>,
}

impl CandidateHeap {
    fn from_vec(values: Vec<TriangularCandidate>) -> Self {
        let mut heap = Self { values };
        if heap.values.len() > 1 {
            for parent in (0..=(heap.values.len() - 2) / 2).rev() {
                heap.sift_down(parent);
            }
        }
        heap
    }

    fn push(&mut self, value: TriangularCandidate) {
        self.values.push(value);
        let mut child = self.values.len() - 1;
        while child != 0 {
            let parent = (child - 1) / 2;
            if !candidate_is_worse(&self.values[parent], &self.values[child]) {
                break;
            }
            self.values.swap(parent, child);
            child = parent;
        }
    }

    fn pop(&mut self) -> Option<TriangularCandidate> {
        if self.values.is_empty() {
            return None;
        }
        let length = self.values.len();
        if length == 1 {
            return self.values.pop();
        }
        // libc++ pop_heap(): Floyd-sift a hole to a leaf, put the last value
        // into that hole, sift it upward, and move the old root to the end.
        let top = self.values[0];
        let mut hole = 0;
        loop {
            let left = 2 * hole + 1;
            let right = left + 1;
            let child =
                if right < length && candidate_is_worse(&self.values[left], &self.values[right]) {
                    right
                } else {
                    left
                };
            self.values[hole] = self.values[child];
            hole = child;
            if child > (length - 2) / 2 {
                break;
            }
        }
        let last = length - 1;
        if hole == last {
            self.values[hole] = top;
        } else {
            self.values[hole] = self.values[last];
            self.values[last] = top;
            self.sift_up(hole);
        }
        self.values.pop()
    }

    fn sift_down(&mut self, parent: usize) {
        let length = self.values.len();
        let left = 2 * parent + 1;
        if left >= length {
            return;
        }
        let mut hole = parent;
        let top = self.values[parent];
        let mut child = left;
        if child + 1 < length && candidate_is_worse(&self.values[child], &self.values[child + 1]) {
            child += 1;
        }
        if candidate_is_worse(&self.values[child], &top) {
            return;
        }
        loop {
            self.values[hole] = self.values[child];
            hole = child;
            if child > (length - 2) / 2 {
                break;
            }
            child = 2 * child + 1;
            let right = child + 1;
            if right < length && candidate_is_worse(&self.values[child], &self.values[right]) {
                child = right;
            }
            if candidate_is_worse(&self.values[child], &top) {
                break;
            }
        }
        self.values[hole] = top;
    }

    fn sift_up(&mut self, mut child: usize) {
        if child == 0 {
            return;
        }
        let value = self.values[child];
        let mut parent = (child - 1) / 2;
        if candidate_is_worse(&self.values[parent], &value) {
            loop {
                self.values[child] = self.values[parent];
                child = parent;
                if child == 0 {
                    break;
                }
                parent = (child - 1) / 2;
                if !candidate_is_worse(&self.values[parent], &value) {
                    break;
                }
            }
            self.values[child] = value;
        }
    }
}

fn candidate_is_worse(left: &TriangularCandidate, right: &TriangularCandidate) -> bool {
    if left.category != right.category {
        return left.category > right.category;
    }
    if left.entries != right.entries {
        return left.entries > right.entries;
    }
    left.penalty > right.penalty
}

#[derive(Debug)]
pub struct InitialBasis<'a> {
    compact_matrix: &'a CompactSparseMatrix,
    objective: &'a DenseRow,
    lower_bound: &'a DenseRow,
    upper_bound: &'a DenseRow,
    variable_type: &'a VariableTypeRow,
    max_scaled_abs_cost: f64,
}

impl<'a> InitialBasis<'a> {
    #[must_use]
    pub const fn new(
        compact_matrix: &'a CompactSparseMatrix,
        objective: &'a DenseRow,
        lower_bound: &'a DenseRow,
        upper_bound: &'a DenseRow,
        variable_type: &'a VariableTypeRow,
    ) -> Self {
        Self {
            compact_matrix,
            objective,
            lower_bound,
            upper_bound,
            variable_type,
            max_scaled_abs_cost: 0.0,
        }
    }

    pub fn complete_bixby_basis(&mut self, num_columns: ColIndex, basis: &mut RowToColMapping) {
        let rows = self.compact_matrix.num_rows().to_usize();
        assert_eq!(basis.len().to_usize(), rows);
        let mut can_be_replaced = vec![false; rows];
        let mut has_zero_coefficient = vec![false; rows];
        for row in 0..rows {
            if basis[RowIndex::from_usize(row)] == INVALID_COL {
                can_be_replaced[row] = true;
                has_zero_coefficient[row] = true;
            }
        }
        let mut scaled_diagonal_abs = vec![f64::INFINITY; rows];
        let candidates = self.compute_candidates(num_columns);
        for candidate in candidates {
            let column = self.compact_matrix.column(candidate);
            let infinity_norm = column
                .iter()
                .fold(0.0_f64, |norm, (_, value)| norm.max(value.abs()));
            if infinity_norm != 1.0 {
                continue;
            }
            let (mut candidate_row, mut candidate_coefficient) =
                restricted_infinity_norm(column.iter(), &has_zero_coefficient);
            let mut enter_basis = candidate_coefficient > 0.99;
            if !enter_basis
                && column
                    .iter()
                    .all(|(row, value)| value.abs() <= scaled_diagonal_abs[row.to_usize()])
            {
                (candidate_row, candidate_coefficient) =
                    restricted_infinity_norm(column.iter(), &can_be_replaced);
                enter_basis = candidate_coefficient != 0.0;
            }
            if enter_basis {
                can_be_replaced[candidate_row] = false;
                for (row, _) in column.iter() {
                    has_zero_coefficient[row.to_usize()] = false;
                }
                scaled_diagonal_abs[candidate_row] = 0.01 * candidate_coefficient.abs();
                basis[RowIndex::from_usize(candidate_row)] = candidate;
            }
        }
    }

    pub fn complete_triangular_primal_basis(
        &mut self,
        num_columns: ColIndex,
        basis: &mut RowToColMapping,
    ) {
        self.complete_triangular_basis(num_columns, basis, false);
    }

    pub fn complete_triangular_dual_basis(
        &mut self,
        num_columns: ColIndex,
        basis: &mut RowToColMapping,
    ) {
        self.complete_triangular_basis(num_columns, basis, true);
    }

    pub fn get_primal_maros_basis(&self, num_columns: ColIndex, basis: &mut RowToColMapping) {
        self.get_maros_basis(num_columns, basis, false);
    }

    pub fn get_dual_maros_basis(&self, num_columns: ColIndex, basis: &mut RowToColMapping) {
        self.get_maros_basis(num_columns, basis, true);
    }

    pub fn compute_candidates(&mut self, num_columns: ColIndex) -> Vec<ColIndex> {
        let mut candidates = Vec::new();
        self.max_scaled_abs_cost = 0.0;
        for index in 0..num_columns.to_usize() {
            let column = ColIndex::from_usize(index);
            if self.variable_type[column] != VariableType::FixedVariable
                && !self.compact_matrix.column_is_empty(column)
            {
                candidates.push(column);
                self.max_scaled_abs_cost =
                    self.max_scaled_abs_cost.max(self.objective[column].abs());
            }
        }
        self.max_scaled_abs_cost = if self.max_scaled_abs_cost == 0.0 {
            1.0
        } else {
            1000.0 * self.max_scaled_abs_cost
        };
        candidates.sort_unstable_by(|&left, &right| {
            self.column_category(left)
                .cmp(&self.column_category(right))
                .then_with(|| {
                    self.column_penalty(left)
                        .total_cmp(&self.column_penalty(right))
                })
        });
        candidates
    }

    fn complete_triangular_basis(
        &mut self,
        num_columns: ColIndex,
        basis: &mut RowToColMapping,
        only_zero_cost: bool,
    ) {
        let rows = self.compact_matrix.num_rows().to_usize();
        assert_eq!(basis.len().to_usize(), rows);
        let mut can_be_replaced = vec![false; rows];
        for (row, replaceable) in can_be_replaced.iter_mut().enumerate() {
            *replaceable = basis[RowIndex::from_usize(row)] == INVALID_COL;
        }
        let columns = num_columns.to_usize();
        let mut pattern = MatrixNonZeroPattern::new(rows, columns);
        for column in 0..columns {
            let column_index = ColIndex::from_usize(column);
            if only_zero_cost && self.objective[column_index] != 0.0 {
                continue;
            }
            for (row, _) in self.compact_matrix.column(column_index).iter() {
                if can_be_replaced[row.to_usize()] {
                    pattern.add_entry(row.to_usize(), column);
                }
            }
        }
        self.max_scaled_abs_cost = (0..columns)
            .map(|column| self.objective[ColIndex::from_usize(column)].abs())
            .fold(0.0, f64::max);
        self.max_scaled_abs_cost = if self.max_scaled_abs_cost == 0.0 {
            1.0
        } else {
            1000.0 * self.max_scaled_abs_cost
        };
        let mut singleton_candidates = Vec::new();
        for column in 0..columns {
            if pattern.column_degree[column] == 1 {
                singleton_candidates.push(self.triangular_candidate(column));
            }
        }
        let mut queue = CandidateHeap::from_vec(singleton_candidates);
        while let Some(candidate) = queue.pop() {
            let column = candidate.column;
            if pattern.column_degree[column] != 1 {
                continue;
            }
            let column_index = ColIndex::from_usize(column);
            let mut pivot_row = None;
            let mut coefficient = 0.0;
            let mut maximum = 0.0_f64;
            for (row, value) in self.compact_matrix.column(column_index).iter() {
                maximum = maximum.max(value.abs());
                if can_be_replaced[row.to_usize()] {
                    pivot_row = Some(row.to_usize());
                    coefficient = value;
                    break;
                }
            }
            if coefficient.abs() < 0.01 * maximum {
                continue;
            }
            let row = pivot_row.expect("residual singleton has an active row");
            basis[RowIndex::from_usize(row)] = column_index;
            can_be_replaced[row] = false;
            // RowNonZero() is deliberately lazy upstream: it includes columns
            // deleted at earlier pivots. Preserve those entries while updating
            // their signed residual degrees.
            let row_columns = pattern.row_nonzeros[row].clone();
            pattern.delete_row_and_column(row, column);
            for other in row_columns {
                if other == column {
                    continue;
                }
                pattern.decrease_column_degree(other);
                if pattern.column_degree[other] == 1 {
                    queue.push(self.triangular_candidate(other));
                }
            }
        }
    }

    fn get_maros_basis(
        &self,
        num_columns: ColIndex,
        basis: &mut RowToColMapping,
        only_zero_cost: bool,
    ) {
        let rows = self.compact_matrix.num_rows().to_usize();
        let columns = num_columns.to_usize();
        assert_eq!(basis.len().to_usize(), rows);
        let first_slack = columns - rows;
        for row in 0..rows {
            basis[RowIndex::from_usize(row)] = ColIndex::from_usize(first_slack + row);
        }
        let mut available = vec![true; columns];
        for (column, slot) in available.iter_mut().enumerate().take(first_slack) {
            let index = ColIndex::from_usize(column);
            if self.variable_type[index] == VariableType::FixedVariable
                || (only_zero_cost && self.objective[index] != 0.0)
            {
                *slot = false;
            }
        }
        for (column, slot) in available.iter_mut().enumerate().skip(first_slack) {
            if self.variable_type[ColIndex::from_usize(column)] == VariableType::Unconstrained {
                *slot = false;
            }
        }
        let mut pattern = MatrixNonZeroPattern::new(rows, columns);
        for column in 0..first_slack {
            for (row, _) in self
                .compact_matrix
                .column(ColIndex::from_usize(column))
                .iter()
            {
                // This apparently surprising row-to-column lookup is literal
                // GLOP behavior (it is not the corresponding slack column).
                if available[row.to_usize()] && available[column] {
                    pattern.add_entry(row.to_usize(), column);
                }
            }
        }
        for row in 0..rows {
            if pattern.row_degree[row] == 0 {
                available[first_slack + row] = false;
            }
        }
        loop {
            let mut pivot_row = None;
            let mut best_row_priority = i32::MIN;
            for row in 0..rows {
                if !available[first_slack + row] {
                    continue;
                }
                let priority = 10 * (3 - self.maros_row_priority(row)) - pattern.row_degree[row];
                // Upstream uses strict greater-than and therefore keeps the
                // first row on ties.
                if priority > best_row_priority {
                    best_row_priority = priority;
                    pivot_row = Some(row);
                }
            }
            let Some(row) = pivot_row else { break };
            let mut best_column = None;
            let mut best_priority = i32::MIN;
            for column in pattern.row_nonzeros(row) {
                if !available[column] {
                    continue;
                }
                let priority =
                    10 * self.maros_column_priority(column) - pattern.column_degree[column];
                if priority > best_priority {
                    let mut pivot = 0.0_f64;
                    let mut maximum = 0.0_f64;
                    for (entry_row, value) in self
                        .compact_matrix
                        .column(ColIndex::from_usize(column))
                        .iter()
                    {
                        maximum = maximum.max(value.abs());
                        if entry_row.to_usize() == row {
                            pivot = value.abs();
                        }
                    }
                    if pivot >= 1e-3 * maximum {
                        best_priority = priority;
                        best_column = Some(column);
                    }
                }
            }
            let Some(column) = best_column else {
                available[first_slack + row] = false;
                continue;
            };
            if self.maros_row_priority(row) >= self.maros_column_priority(column) {
                available[first_slack + row] = false;
                continue;
            }
            basis[RowIndex::from_usize(row)] = ColIndex::from_usize(column);
            available[column] = false;
            available[first_slack + row] = false;
            let row_columns = pattern.row_nonzeros[row].clone();
            pattern.delete_row_and_column(row, column);
            for other in row_columns {
                available[other] = false;
            }
        }
    }

    fn triangular_candidate(&self, column: usize) -> TriangularCandidate {
        let index = ColIndex::from_usize(column);
        TriangularCandidate {
            column,
            category: self.column_category(index),
            entries: self.compact_matrix.column(index).len(),
            penalty: self.column_penalty(index),
        }
    }

    fn column_category(&self, column: ColIndex) -> i32 {
        match self.variable_type[column] {
            VariableType::Unconstrained => 2,
            VariableType::LowerBounded | VariableType::UpperBounded => 3,
            VariableType::UpperAndLowerBounded => 4,
            VariableType::FixedVariable => 5,
        }
    }

    fn column_penalty(&self, column: ColIndex) -> f64 {
        let bound_penalty = match self.variable_type[column] {
            VariableType::LowerBounded => self.lower_bound[column],
            VariableType::UpperBounded => -self.upper_bound[column],
            VariableType::UpperAndLowerBounded => {
                self.lower_bound[column] - self.upper_bound[column]
            }
            VariableType::Unconstrained | VariableType::FixedVariable => 0.0,
        };
        bound_penalty + self.objective[column].abs() / self.max_scaled_abs_cost
    }

    fn maros_column_priority(&self, column: usize) -> i32 {
        match self.variable_type[ColIndex::from_usize(column)] {
            VariableType::Unconstrained => 3,
            VariableType::LowerBounded | VariableType::UpperBounded => 2,
            VariableType::UpperAndLowerBounded => 1,
            VariableType::FixedVariable => 0,
        }
    }

    fn maros_row_priority(&self, row: usize) -> i32 {
        let first_slack =
            self.compact_matrix.num_cols().to_usize() - self.compact_matrix.num_rows().to_usize();
        self.maros_column_priority(first_slack + row)
    }
}

fn restricted_infinity_norm(
    entries: impl Iterator<Item = (RowIndex, f64)>,
    allowed: &[bool],
) -> (usize, f64) {
    let mut row = 0;
    let mut coefficient = 0.0_f64;
    for (candidate_row, candidate) in entries {
        if allowed[candidate_row.to_usize()] && candidate.abs() > coefficient {
            row = candidate_row.to_usize();
            coefficient = candidate.abs();
        }
    }
    (row, coefficient)
}
