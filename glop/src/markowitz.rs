//! Sparse Markowitz factorization machinery.
//!
//! This follows `ortools/glop/markowitz.{h,cc}`: pivot selection uses an
//! incrementally maintained residual pattern, singleton fast paths, a bucketed
//! degree queue, a Zlatev candidate set, and cached sparse left-looking
//! columns. The active matrix is never expanded into a dense array.

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;
use lp_data::sparse_vector::SparseColumn;
use lp_data::triangular_matrix::{Triangle, TriangularMatrix};

use crate::parameters::GlopParameters;

const INVALID: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pivot {
    pub row: usize,
    pub column: usize,
    pub markowitz: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SparseLu {
    /// Pivot position -> input row.
    pub row_permutation: Vec<usize>,
    /// Pivot position -> input column.
    pub column_permutation: Vec<usize>,
    pub num_fp_operations: i64,
    pub stats: Option<MarkowitzStats>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MarkowitzStats {
    pub basis_singleton_column_ratio: f64,
    pub basis_residual_singleton_column_ratio: f64,
    pub pivots_without_fill_in_ratio: f64,
    pub degree_two_pivot_columns: f64,
}

struct MarkowitzResult {
    pivot_rows: Vec<usize>,
    pivot_columns: Vec<usize>,
    row_permutation: Vec<usize>,
    num_fp_operations: i64,
    stats: Option<MarkowitzStats>,
}

#[derive(Clone, Copy)]
struct MatrixView<'a> {
    matrix: &'a SparseMatrix,
    columns: Option<&'a [usize]>,
}

impl<'a> MatrixView<'a> {
    fn full(matrix: &'a SparseMatrix) -> Self {
        Self {
            matrix,
            columns: None,
        }
    }

    fn selected(matrix: &'a SparseMatrix, columns: &'a [usize]) -> Self {
        Self {
            matrix,
            columns: Some(columns),
        }
    }

    fn num_rows(self) -> usize {
        self.matrix.num_rows().to_usize()
    }

    fn num_columns(self) -> usize {
        self.columns
            .map_or_else(|| self.matrix.num_cols().to_usize(), <[usize]>::len)
    }

    fn column(self, column: usize) -> &'a SparseColumn {
        let source = self.columns.map_or(column, |columns| columns[column]);
        self.matrix.column(ColIndex::from_usize(source))
    }
}

/// Symbolic residual matrix. Deleted columns remain lazily in row adjacency
/// lists; the separate degrees always describe the active submatrix.
#[derive(Clone, Debug, Default)]
struct MatrixNonZeroPattern {
    row_nonzeros: Vec<Vec<usize>>,
    row_degree: Vec<usize>,
    column_degree: Vec<usize>,
    deleted_columns: Vec<bool>,
    scratchpad: Vec<bool>,
    non_deleted_columns: usize,
}

impl MatrixNonZeroPattern {
    fn reset_from_matrix_subset(
        &mut self,
        matrix: MatrixView<'_>,
        row_permutation: &[usize],
        column_permutation: &[usize],
        singleton_columns: &mut Vec<usize>,
        singleton_rows: &mut Vec<usize>,
    ) {
        let rows = matrix.num_rows();
        let columns = matrix.num_columns();
        self.row_nonzeros.resize_with(rows, Vec::new);
        for row in &mut self.row_nonzeros {
            row.clear();
        }
        self.row_degree.resize(rows, 0);
        self.row_degree.fill(0);
        self.column_degree.resize(columns, 0);
        self.column_degree.fill(0);
        self.deleted_columns.resize(columns, false);
        self.deleted_columns.fill(false);
        self.scratchpad.resize(columns, false);
        self.scratchpad.fill(false);
        self.non_deleted_columns = columns;
        singleton_columns.clear();
        singleton_rows.clear();
        for (column, &permuted) in column_permutation.iter().enumerate().take(columns) {
            if permuted != INVALID {
                self.deleted_columns[column] = true;
                self.non_deleted_columns -= 1;
                continue;
            }
            for entry in matrix.column(column) {
                let row = entry.index().to_usize();
                if row_permutation[row] != INVALID {
                    continue;
                }
                self.row_nonzeros[row].push(column);
                self.row_degree[row] += 1;
                self.column_degree[column] += 1;
            }
        }
        singleton_columns.extend((0..columns).filter(|&column| self.column_degree[column] == 1));
        singleton_rows.extend((0..rows).filter(|&row| self.row_degree[row] == 1));
    }

    fn delete_row_and_column(&mut self, row: usize, column: usize) {
        debug_assert!(!self.deleted_columns[column]);
        self.deleted_columns[column] = true;
        self.non_deleted_columns -= 1;
        self.row_degree[row] = 0;
    }

    fn first_non_deleted_column(&self, row: usize) -> Option<usize> {
        self.row_nonzeros[row]
            .iter()
            .copied()
            .find(|&column| !self.deleted_columns[column])
    }

    fn clean_row(&mut self, row: usize) {
        let deleted = &self.deleted_columns;
        self.row_nonzeros[row].retain(|&column| !deleted[column]);
    }

    /// Symbolic outer-product update after the pivot row/column are deleted.
    fn update(&mut self, pivot_row: usize, pivot_column: usize, column: &SparseColumn) {
        debug_assert!(self.deleted_columns[pivot_column]);
        let maximum_row_degree = self.non_deleted_columns + 1;
        self.clean_row(pivot_row);
        for &column in &self.row_nonzeros[pivot_row] {
            self.column_degree[column] -= 1;
            self.scratchpad[column] = false;
        }
        for entry in column {
            let row = entry.index().to_usize();
            if row == pivot_row
                || entry.coefficient() == 0.0
                || self.row_degree[row] == maximum_row_degree
            {
                continue;
            }
            if self.row_nonzeros[row].len() > self.row_degree[row] + 4 {
                let deleted = &self.deleted_columns;
                self.row_nonzeros[row].retain(|&column| !deleted[column]);
            }
            let (pivot_pattern, target_pattern) = if row < pivot_row {
                let (before_pivot, pivot_and_after) = self.row_nonzeros.split_at_mut(pivot_row);
                (&pivot_and_after[0], &mut before_pivot[row])
            } else {
                let (pivot_and_before, after_pivot) = self.row_nonzeros.split_at_mut(row);
                (&pivot_and_before[pivot_row], &mut after_pivot[0])
            };
            for &column in target_pattern.iter() {
                self.scratchpad[column] = true;
            }
            let old_size = target_pattern.len();
            for &column in pivot_pattern {
                if self.scratchpad[column] {
                    self.scratchpad[column] = false;
                } else {
                    target_pattern.push(column);
                    self.column_degree[column] += 1;
                }
            }
            self.row_degree[row] += target_pattern.len() - old_size;
        }
    }

    /// Updates row degrees after deleting a computed pivot column.
    ///
    /// Structural zeros belong to the residual pattern and are counted. An
    /// exact-zero reachability overestimate does not: it can be distinguished
    /// by the absence of the pivot column from that row's symbolic pattern.
    fn remove_column(
        &mut self,
        pivot_column: usize,
        column: &SparseColumn,
        singleton_rows: &mut Vec<usize>,
    ) {
        for entry in column {
            let row = entry.index().to_usize();
            if self.row_degree[row] == 0
                || (entry.coefficient() == 0.0 && !self.row_nonzeros[row].contains(&pivot_column))
            {
                continue;
            }
            self.row_degree[row] -= 1;
            if self.row_degree[row] == 1 {
                singleton_rows.push(row);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn store_pivot_column(
    matrix: MatrixView<'_>,
    row: usize,
    column: usize,
    row_permutation: &mut [usize],
    column_permutation: &mut [usize],
    pivot_rows: &mut Vec<usize>,
    pivot_columns: &mut Vec<usize>,
    lower_factor: &mut TriangularMatrix,
    upper_factor: &mut TriangularMatrix,
) {
    let step = pivot_rows.len();
    let source = matrix.column(column);
    let pivot = source.look_up_coefficient(RowIndex::from_usize(row));
    lower_factor.add_diagonal_only_column(1.0);
    upper_factor.add_column_with_diagonal(
        source.iter().filter_map(|entry| {
            let entry_row = entry.index().to_usize();
            (entry_row != row && row_permutation[entry_row] != INVALID)
                .then_some((entry_row, entry.coefficient()))
        }),
        pivot,
    );
    pivot_rows.push(row);
    pivot_columns.push(column);
    row_permutation[row] = step;
    column_permutation[column] = step;
}

/// GLOP's adjustable bucket queue: no sorting or heap logarithm is required.
#[derive(Clone, Debug, Default)]
struct ColumnPriorityQueue {
    minimum_degree: usize,
    degree: Vec<usize>,
    previous: Vec<usize>,
    next: Vec<usize>,
    first_by_degree: Vec<usize>,
}

impl ColumnPriorityQueue {
    fn reset(&mut self, maximum_degree: usize, columns: usize) {
        self.minimum_degree = maximum_degree + 1;
        self.degree.resize(columns, 0);
        self.degree.fill(0);
        self.previous.resize(columns, INVALID);
        self.previous.fill(INVALID);
        self.next.resize(columns, INVALID);
        self.next.fill(INVALID);
        self.first_by_degree.resize(maximum_degree + 1, INVALID);
        self.first_by_degree.fill(INVALID);
    }

    fn remove(&mut self, column: usize, old_degree: usize) {
        let next = self.next[column];
        let previous = self.previous[column];
        if next != INVALID {
            self.previous[next] = previous;
        }
        if previous == INVALID {
            self.first_by_degree[old_degree] = next;
        } else {
            self.next[previous] = next;
        }
        self.degree[column] = 0;
    }

    fn insert(&mut self, column: usize, degree: usize) {
        let next = self.first_by_degree[degree];
        self.next[column] = next;
        if next != INVALID {
            self.previous[next] = column;
        }
        self.first_by_degree[degree] = column;
        self.previous[column] = INVALID;
        self.degree[column] = degree;
        self.minimum_degree = self.minimum_degree.min(degree);
    }

    fn push_or_adjust(&mut self, column: usize, degree: usize) {
        let old_degree = self.degree[column];
        if old_degree == degree {
            return;
        }
        if old_degree != 0 {
            self.remove(column, old_degree);
        }
        if degree != 0 {
            self.insert(column, degree);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        while self.minimum_degree < self.first_by_degree.len()
            && self.first_by_degree[self.minimum_degree] == INVALID
        {
            self.minimum_degree += 1;
        }
        if self.minimum_degree == self.first_by_degree.len() {
            return None;
        }
        let result = self.first_by_degree[self.minimum_degree];
        self.remove(result, self.minimum_degree);
        Some(result)
    }
}

#[derive(Clone, Debug, Default)]
struct CandidateColumn {
    needs_solve: bool,
    needs_split: bool,
}

/// GLOP's logical-column repository backed by a small reusable physical pool.
/// Pivoted columns release their allocation for later candidate columns.
#[derive(Clone, Debug)]
struct ReusableColumnMemory {
    mapping: Vec<usize>,
    free_columns: Vec<usize>,
    columns: Vec<SparseColumn>,
    empty_column: SparseColumn,
}

impl ReusableColumnMemory {
    fn new(num_columns: usize) -> Self {
        Self {
            mapping: vec![INVALID; num_columns],
            free_columns: Vec::new(),
            columns: Vec::new(),
            empty_column: SparseColumn::new(),
        }
    }

    fn reset(&mut self, num_columns: usize) {
        self.mapping.resize(num_columns, INVALID);
        self.mapping.fill(INVALID);
        self.free_columns.clear();
        for (slot, column) in self.columns.iter_mut().enumerate() {
            column.clear();
            self.free_columns.push(slot);
        }
    }

    fn column(&self, column: usize) -> &SparseColumn {
        let slot = self.mapping[column];
        if slot == INVALID {
            &self.empty_column
        } else {
            &self.columns[slot]
        }
    }

    fn mutable_column(&mut self, column: usize) -> &mut SparseColumn {
        if self.mapping[column] == INVALID {
            let slot = self.free_columns.pop().unwrap_or_else(|| {
                self.columns.push(SparseColumn::new());
                self.columns.len() - 1
            });
            self.mapping[column] = slot;
        }
        &mut self.columns[self.mapping[column]]
    }

    fn take_column(&mut self, column: usize) -> SparseColumn {
        std::mem::take(self.mutable_column(column))
    }

    fn restore_column(&mut self, column: usize, value: SparseColumn) {
        *self.mutable_column(column) = value;
    }

    fn clear_and_release_column(&mut self, column: usize) {
        let slot = self.mapping[column];
        debug_assert_ne!(slot, INVALID);
        self.columns[slot].clear();
        self.free_columns.push(slot);
        self.mapping[column] = INVALID;
    }
}

/// Allocated Markowitz state retained across basis refactorizations, as in
/// upstream's persistent `Markowitz` member.
#[derive(Clone, Debug)]
pub(crate) struct MarkowitzWorkspace {
    pattern: MatrixNonZeroPattern,
    candidates: Vec<CandidateColumn>,
    permuted_lower: ReusableColumnMemory,
    permuted_upper: ReusableColumnMemory,
    singleton_columns: Vec<usize>,
    singleton_rows: Vec<usize>,
    queue: ColumnPriorityQueue,
    examined: Vec<usize>,
}

impl Default for MarkowitzWorkspace {
    fn default() -> Self {
        Self {
            pattern: MatrixNonZeroPattern::default(),
            candidates: Vec::new(),
            permuted_lower: ReusableColumnMemory::new(0),
            permuted_upper: ReusableColumnMemory::new(0),
            singleton_columns: Vec::new(),
            singleton_rows: Vec::new(),
            queue: ColumnPriorityQueue::default(),
            examined: Vec::new(),
        }
    }
}

fn copy_input_column(matrix: MatrixView<'_>, column: usize, result: &mut SparseColumn) {
    let source = matrix.column(column);
    result.clear();
    result.reserve(source.num_entries());
    for entry in source {
        result.add_entry(entry.index(), entry.coefficient());
    }
}

fn structural_singleton_row(
    residual: &SparseColumn,
    pattern: &MatrixNonZeroPattern,
    column: usize,
) -> Option<usize> {
    residual
        .iter()
        .map(|entry| entry.index().to_usize())
        .find(|&row| pattern.row_nonzeros[row].contains(&column))
}

#[allow(clippy::too_many_arguments)]
fn compute_column<'a>(
    matrix: MatrixView<'_>,
    column: usize,
    row_permutation: &[usize],
    lower_factor: &mut TriangularMatrix,
    candidates: &mut [CandidateColumn],
    permuted_lower: &'a mut ReusableColumnMemory,
    permuted_upper: &mut ReusableColumnMemory,
    residual_degree: usize,
    num_fp_operations: &mut i64,
) -> &'a SparseColumn {
    let first_time =
        permuted_lower.column(column).is_empty() && permuted_upper.column(column).is_empty();
    let mut residual = permuted_lower.take_column(column);
    if candidates[column].needs_solve {
        if first_time {
            copy_input_column(matrix, column, &mut residual);
        }
        lower_factor.permuted_lower_sparse_solve(
            &mut residual,
            row_permutation,
            permuted_upper.mutable_column(column),
        );
        *num_fp_operations += lower_factor.num_fp_operations_in_last_permuted_lower_sparse_solve();
    } else {
        // GLOP performs this test before populating a column seen for the
        // first time. In particular, an empty residual column is returned
        // immediately when its structural degree is zero.
        if residual.num_entries() == residual_degree && !candidates[column].needs_split {
            permuted_lower.restore_column(column, residual);
            return permuted_lower.column(column);
        }
        if first_time {
            copy_input_column(matrix, column, &mut residual);
            *num_fp_operations += i64::try_from(residual.num_entries()).unwrap_or(i64::MAX);
        }
        *num_fp_operations += i64::try_from(residual.num_entries()).unwrap_or(i64::MAX);
        residual.move_entries_with_index_tags_to(
            row_permutation,
            INVALID,
            permuted_upper.mutable_column(column),
        );
        candidates[column].needs_split = false;
        permuted_lower.restore_column(column, residual);
        return permuted_lower.column(column);
    }
    candidates[column].needs_solve = false;
    candidates[column].needs_split = false;
    permuted_lower.restore_column(column, residual);
    debug_assert!(
        permuted_lower
            .column(column)
            .iter()
            .all(|entry| row_permutation[entry.index().to_usize()] == INVALID),
        "computed candidate column {column} retained a pivoted row"
    );
    permuted_lower.column(column)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn find_pivot(
    matrix: MatrixView<'_>,
    parameters: &GlopParameters,
    pattern: &MatrixNonZeroPattern,
    row_perm: &[usize],
    col_perm: &[usize],
    lower_factor: &mut TriangularMatrix,
    candidates: &mut [CandidateColumn],
    permuted_lower: &mut ReusableColumnMemory,
    permuted_upper: &mut ReusableColumnMemory,
    singleton_columns: &mut Vec<usize>,
    singleton_rows: &mut Vec<usize>,
    contains_only_singleton_columns: &mut bool,
    queue: &mut ColumnPriorityQueue,
    queue_initialized: &mut bool,
    examined: &mut Vec<usize>,
    num_fp_operations: &mut i64,
) -> Option<Pivot> {
    while let Some(column) = singleton_columns.pop() {
        if col_perm[column] != INVALID || pattern.column_degree[column] != 1 {
            continue;
        }
        if *contains_only_singleton_columns {
            let row = matrix
                .column(column)
                .into_iter()
                .find(|entry| row_perm[entry.index().to_usize()] == INVALID)?
                .index()
                .to_usize();
            return Some(Pivot {
                row,
                column,
                markowitz: 0,
            });
        }
        let residual = compute_column(
            matrix,
            column,
            row_perm,
            lower_factor,
            candidates,
            permuted_lower,
            permuted_upper,
            pattern.column_degree[column],
            num_fp_operations,
        );
        if let Some(row) = structural_singleton_row(residual, pattern, column) {
            return Some(Pivot {
                row,
                column,
                markowitz: 0,
            });
        }
    }
    *contains_only_singleton_columns = false;
    while let Some(row) = singleton_rows.pop() {
        if row_perm[row] != INVALID || pattern.row_degree[row] != 1 {
            continue;
        }
        let Some(column) = pattern.first_non_deleted_column(row) else {
            continue;
        };
        let residual = compute_column(
            matrix,
            column,
            row_perm,
            lower_factor,
            candidates,
            permuted_lower,
            permuted_upper,
            pattern.column_degree[column],
            num_fp_operations,
        );
        if residual.iter().any(|entry| entry.index().to_usize() == row) {
            return Some(Pivot {
                row,
                column,
                markowitz: 0,
            });
        }
    }
    if !*queue_initialized {
        *queue_initialized = true;
        queue.reset(row_perm.len(), col_perm.len());
        for (column, &permuted) in col_perm.iter().enumerate() {
            if permuted == INVALID {
                let degree = pattern.column_degree[column];
                if degree == 1 {
                    singleton_columns.push(column);
                } else {
                    queue.push_or_adjust(column, degree);
                }
            }
        }
        if !singleton_columns.is_empty() {
            return find_pivot(
                matrix,
                parameters,
                pattern,
                row_perm,
                col_perm,
                lower_factor,
                candidates,
                permuted_lower,
                permuted_upper,
                singleton_columns,
                singleton_rows,
                contains_only_singleton_columns,
                queue,
                queue_initialized,
                examined,
                num_fp_operations,
            );
        }
    }
    examined.clear();
    let mut best: Option<(usize, f64, usize, usize)> = None;
    let zlatev_parameter =
        usize::try_from(parameters.markowitz_zlatev_parameter).unwrap_or(usize::MAX);
    while examined.len() < zlatev_parameter {
        let Some(column) = queue.pop() else { break };
        if col_perm[column] != INVALID {
            continue;
        }
        let degree = pattern.column_degree[column];
        examined.push(column);
        if best.is_some_and(|candidate| candidate.0 < degree.saturating_sub(1)) {
            break;
        }
        let residual = compute_column(
            matrix,
            column,
            row_perm,
            lower_factor,
            candidates,
            permuted_lower,
            permuted_upper,
            degree,
            num_fp_operations,
        );
        let maximum = residual
            .iter()
            .map(|entry| entry.coefficient().abs())
            .fold(0.0, f64::max);
        if maximum == 0.0 {
            examined.pop();
            continue;
        }
        let minimum = parameters.lu_factorization_pivot_threshold * maximum;
        for entry in residual {
            let magnitude = entry.coefficient().abs();
            if magnitude < minimum {
                continue;
            }
            let row = entry.index().to_usize();
            let markowitz = (degree - 1) * (pattern.row_degree[row] - 1);
            // Preserve upstream's traversal-dependent tie behavior: for equal
            // Markowitz numbers only a *strictly* larger magnitude replaces
            // the current pivot. Row/column indices are not extra tie keys.
            if best.is_none_or(|current| {
                markowitz < current.0 || (markowitz == current.0 && magnitude > current.1)
            }) {
                best = Some((markowitz, magnitude, row, column));
            }
        }
    }
    let chosen = best.map(|candidate| candidate.3);
    for &column in examined.iter() {
        if Some(column) != chosen {
            queue.push_or_adjust(column, pattern.column_degree[column]);
        }
    }
    best.map(|(markowitz, _, row, column)| Pivot {
        row,
        column,
        markowitz,
    })
}

fn update_degree(
    column: usize,
    degree: usize,
    queue_initialized: bool,
    queue: &mut ColumnPriorityQueue,
    singleton_columns: &mut Vec<usize>,
) {
    if degree == 1 {
        singleton_columns.push(column);
    } else if queue_initialized {
        queue.push_or_adjust(column, degree);
    }
}

#[allow(clippy::too_many_lines)]
fn compute(
    matrix: MatrixView<'_>,
    parameters: &GlopParameters,
    workspace: &mut MarkowitzWorkspace,
    lower_factor: &mut TriangularMatrix,
    upper_factor: &mut TriangularMatrix,
) -> MarkowitzResult {
    let num_rows = matrix.num_rows();
    let num_columns = matrix.num_columns();
    let maximum_pivots = num_rows.min(num_columns);
    let mut row_perm = vec![INVALID; num_rows];
    let mut col_perm = vec![INVALID; num_columns];
    let mut pivot_rows = Vec::with_capacity(maximum_pivots);
    let mut pivot_columns = Vec::with_capacity(maximum_pivots);
    lower_factor.reset(num_rows, maximum_pivots);
    upper_factor.reset(num_rows, maximum_pivots);
    let pattern = &mut workspace.pattern;
    let candidates = &mut workspace.candidates;
    let permuted_lower = &mut workspace.permuted_lower;
    let permuted_upper = &mut workspace.permuted_upper;
    let singleton_columns = &mut workspace.singleton_columns;
    let singleton_rows = &mut workspace.singleton_rows;
    let queue = &mut workspace.queue;
    let examined = &mut workspace.examined;
    candidates.resize(num_columns, CandidateColumn::default());
    candidates.fill(CandidateColumn::default());
    permuted_lower.reset(num_columns);
    permuted_upper.reset(num_columns);
    let mut num_fp_operations = 0_i64;
    let matrix_is_empty = (0..num_columns).all(|column| matrix.column(column).is_empty());

    // Upstream first extracts true singleton columns. For competing singleton
    // columns on one row, the lowest input column wins; selected entries are
    // then ordered by row to keep the row permutation close to identity.
    let mut claimed_rows = vec![false; num_rows];
    let mut initial_singletons = Vec::new();
    for column in 0..num_columns {
        let source = matrix.column(column);
        if source.num_entries() == 1 {
            let row = source
                .first()
                .expect("singleton has one entry")
                .index()
                .to_usize();
            if !claimed_rows[row] {
                claimed_rows[row] = true;
                initial_singletons.push((row, column));
            }
        }
    }
    initial_singletons.sort_unstable_by_key(|entry| entry.0);
    for (row, column) in initial_singletons {
        store_pivot_column(
            matrix,
            row,
            column,
            &mut row_perm,
            &mut col_perm,
            &mut pivot_rows,
            &mut pivot_columns,
            lower_factor,
            upper_factor,
        );
    }
    let basis_singletons = pivot_rows.len();

    // Then make one input-order pass for columns that are singleton after the
    // rows selected above are removed. This mirrors
    // ExtractResidualSingletonColumns(), including its evolving row subset.
    for column in 0..num_columns {
        if col_perm[column] != INVALID {
            continue;
        }
        let mut residual_row = INVALID;
        let mut residual_degree = 0;
        for entry in matrix.column(column) {
            let row = entry.index().to_usize();
            if row_perm[row] == INVALID {
                residual_row = row;
                residual_degree += 1;
                if residual_degree > 1 {
                    break;
                }
            }
        }
        if residual_degree == 1 {
            store_pivot_column(
                matrix,
                residual_row,
                column,
                &mut row_perm,
                &mut col_perm,
                &mut pivot_rows,
                &mut pivot_columns,
                lower_factor,
                upper_factor,
            );
        }
    }
    let residual_singletons = pivot_rows.len();
    let mut pivots_without_fill_in = residual_singletons;

    pattern.reset_from_matrix_subset(
        matrix,
        &row_perm,
        &col_perm,
        singleton_columns,
        singleton_rows,
    );
    let mut queue_initialized = false;
    let mut contains_only_singleton_columns = true;
    let zlatev_parameter =
        usize::try_from(parameters.markowitz_zlatev_parameter).unwrap_or(usize::MAX);
    examined.clear();
    examined.reserve(zlatev_parameter.saturating_add(1));

    for step in pivot_rows.len()..maximum_pivots {
        let Some(pivot) = find_pivot(
            matrix,
            parameters,
            pattern,
            &row_perm,
            &col_perm,
            lower_factor,
            candidates,
            permuted_lower,
            permuted_upper,
            singleton_columns,
            singleton_rows,
            &mut contains_only_singleton_columns,
            queue,
            &mut queue_initialized,
            examined,
            &mut num_fp_operations,
        ) else {
            break;
        };
        // FindPivot() has already materialized every non-singleton candidate
        // it can return. GLOP consumes that cached column directly here; a
        // second ComputeColumn() call is not merely redundant because a
        // changed needs-solve flag can make it perform and charge more work.
        let empty_residual = SparseColumn::new();
        let residual = if contains_only_singleton_columns {
            &empty_residual
        } else {
            permuted_lower.column(pivot.column)
        };
        let pivot_value = if contains_only_singleton_columns {
            matrix
                .column(pivot.column)
                .look_up_coefficient(RowIndex::from_usize(pivot.row))
        } else {
            residual.look_up_coefficient(RowIndex::from_usize(pivot.row))
        };
        if !pivot_value.is_finite()
            || pivot_value.abs() <= parameters.markowitz_singularity_threshold
        {
            break;
        }

        let column_degree = pattern.column_degree[pivot.column];
        let row_degree = pattern.row_degree[pivot.row];
        pattern.delete_row_and_column(pivot.row, pivot.column);
        if pivot.markowitz == 0 && column_degree == 1 {
            pivots_without_fill_in += 1;
            for &column in &pattern.row_nonzeros[pivot.row] {
                if pattern.deleted_columns[column] {
                    continue;
                }
                pattern.column_degree[column] -= 1;
                candidates[column].needs_split = true;
                update_degree(
                    column,
                    pattern.column_degree[column],
                    queue_initialized,
                    queue,
                    singleton_columns,
                );
            }
        } else if pivot.markowitz == 0 {
            pivots_without_fill_in += 1;
            debug_assert_eq!(row_degree, 1);
            pattern.remove_column(pivot.column, residual, singleton_rows);
        } else {
            pattern.update(pivot.row, pivot.column, residual);
            for &column in &pattern.row_nonzeros[pivot.row] {
                if !pattern.deleted_columns[column] {
                    candidates[column].needs_solve = true;
                    update_degree(
                        column,
                        pattern.column_degree[column],
                        queue_initialized,
                        queue,
                        singleton_columns,
                    );
                }
            }
            pattern.remove_column(pivot.column, residual, singleton_rows);
        }

        if contains_only_singleton_columns {
            store_pivot_column(
                matrix,
                pivot.row,
                pivot.column,
                &mut row_perm,
                &mut col_perm,
                &mut pivot_rows,
                &mut pivot_columns,
                lower_factor,
                upper_factor,
            );
            continue;
        }

        // This is GLOP's AddAndNormalizeTriangularColumn(): it filters exact
        // cancellations while writing L directly into its final storage.
        lower_factor.add_and_normalize_triangular_column(
            residual,
            RowIndex::from_usize(pivot.row),
            pivot_value,
        );
        upper_factor.add_triangular_column_with_given_diagonal(
            permuted_upper.column(pivot.column),
            RowIndex::from_usize(pivot.row),
            pivot_value,
        );
        permuted_lower.clear_and_release_column(pivot.column);
        permuted_upper.clear_and_release_column(pivot.column);
        pivot_rows.push(pivot.row);
        pivot_columns.push(pivot.column);
        row_perm[pivot.row] = step;
        col_perm[pivot.column] = step;
    }

    let lower_entries = lower_factor.num_entries();
    let upper_entries = upper_factor.num_entries();
    // Native Markowitz returns immediately on a missing/small pivot. Its
    // final L/U storage charge is reached only for a complete permutation.
    if pivot_rows.len() == maximum_pivots {
        num_fp_operations +=
            10 * i64::try_from(lower_entries.saturating_add(upper_entries)).unwrap_or(i64::MAX);
    }
    #[allow(clippy::cast_precision_loss)]
    let stats = (!matrix_is_empty).then(|| MarkowitzStats {
        basis_singleton_column_ratio: basis_singletons as f64 / num_rows as f64,
        basis_residual_singleton_column_ratio: residual_singletons as f64 / num_rows as f64,
        pivots_without_fill_in_ratio: pivots_without_fill_in as f64 / num_rows as f64,
        // The counter increment is guarded by IF_STATS_ENABLED() upstream;
        // the pinned release reference does not define OR_STATS.
        degree_two_pivot_columns: 0.0,
    });
    MarkowitzResult {
        pivot_rows,
        pivot_columns,
        row_permutation: row_perm,
        num_fp_operations,
        stats,
    }
}

pub(crate) fn factorize(
    matrix: &SparseMatrix,
    parameters: &GlopParameters,
    workspace: &mut MarkowitzWorkspace,
    lower: &mut TriangularMatrix,
    upper: &mut TriangularMatrix,
) -> Result<SparseLu, usize> {
    let n = matrix.num_rows().to_usize();
    let result = compute(
        MatrixView::full(matrix),
        parameters,
        workspace,
        lower,
        upper,
    );
    factors_from_result(result, n, lower, upper)
}

pub(crate) fn factorize_selected(
    matrix: &SparseMatrix,
    columns: &[usize],
    parameters: &GlopParameters,
    workspace: &mut MarkowitzWorkspace,
    lower: &mut TriangularMatrix,
    upper: &mut TriangularMatrix,
) -> Result<SparseLu, usize> {
    let n = matrix.num_rows().to_usize();
    let result = compute(
        MatrixView::selected(matrix, columns),
        parameters,
        workspace,
        lower,
        upper,
    );
    factors_from_result(result, n, lower, upper)
}

fn factors_from_result(
    result: MarkowitzResult,
    n: usize,
    lower: &mut TriangularMatrix,
    upper: &mut TriangularMatrix,
) -> Result<SparseLu, usize> {
    if result.pivot_rows.len() != n {
        return Err(result.pivot_rows.len());
    }
    // Match Markowitz::ComputeLU(): permute stored off-diagonal row indices
    // in place, without changing their order or rebuilding either factor.
    lower.apply_row_permutation_to_non_diagonal_entries(&result.row_permutation);
    upper.apply_row_permutation_to_non_diagonal_entries(&result.row_permutation);
    debug_assert!((0..n).all(|column| lower.column(column).all(|(row, _)| row > column)));
    debug_assert!((0..n).all(|column| upper.column(column).all(|(row, _)| row < column)));
    Ok(SparseLu {
        row_permutation: result.pivot_rows,
        column_permutation: result.pivot_columns,
        num_fp_operations: result.num_fp_operations,
        stats: result.stats,
    })
}

pub(crate) fn compute_pivot_sequence(
    matrix: &SparseMatrix,
    columns: &[usize],
    parameters: &GlopParameters,
) -> (Vec<usize>, Vec<usize>) {
    let (rows, columns, _) = compute_pivot_sequence_with_operations(matrix, columns, parameters);
    (rows, columns)
}

pub(crate) fn compute_pivot_sequence_with_operations(
    matrix: &SparseMatrix,
    columns: &[usize],
    parameters: &GlopParameters,
) -> (Vec<usize>, Vec<usize>, i64) {
    let mut lower = TriangularMatrix::empty(Triangle::Lower, true);
    let mut upper = TriangularMatrix::empty(Triangle::Upper, false);
    let result = compute(
        MatrixView::selected(matrix, columns),
        parameters,
        &mut MarkowitzWorkspace::default(),
        &mut lower,
        &mut upper,
    );
    (
        result.pivot_rows,
        result.pivot_columns,
        result.num_fp_operations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_candidate_permutation_skips_final_storage_clock() {
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(2), lp_data::lp_types::ColIndex::new(2));
        for column in 0..2 {
            matrix
                .mutable_column(lp_data::lp_types::ColIndex::from_usize(column))
                .add_entry(RowIndex::new(0), 1.0);
        }
        let (rows, columns, operations) =
            compute_pivot_sequence_with_operations(&matrix, &[0, 1], &GlopParameters::default());
        assert_eq!(rows, vec![0]);
        assert_eq!(columns, vec![0]);
        assert_eq!(operations, 0);
    }

    #[test]
    fn adjustable_queue_returns_lowest_degree_without_sorting() {
        let mut queue = ColumnPriorityQueue::default();
        queue.reset(5, 4);
        queue.push_or_adjust(0, 3);
        queue.push_or_adjust(1, 1);
        queue.push_or_adjust(2, 2);
        queue.push_or_adjust(0, 1);
        assert_eq!(queue.pop(), Some(0));
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn candidate_columns_reuse_released_physical_storage() {
        let mut columns = ReusableColumnMemory::new(4);
        columns.mutable_column(0).add_entry(RowIndex::new(2), 3.0);
        assert_eq!(columns.columns.len(), 1);
        columns.clear_and_release_column(0);
        columns.mutable_column(3).add_entry(RowIndex::new(1), -2.0);
        assert_eq!(columns.columns.len(), 1);
        assert!(columns.column(0).is_empty());
        assert_eq!(
            columns
                .column(3)
                .look_up_coefficient(RowIndex::new(1))
                .to_bits(),
            (-2.0_f64).to_bits()
        );
    }

    #[test]
    fn deleted_row_disables_the_cardinality_only_split_shortcut() {
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(3), lp_data::lp_types::ColIndex::new(1));
        let mut lower_factor = TriangularMatrix::empty(Triangle::Lower, true);
        lower_factor.reset(3, 3);
        let row_permutation = [INVALID, 0, INVALID];
        let mut candidates = vec![CandidateColumn {
            needs_solve: false,
            needs_split: true,
        }];
        let mut permuted_lower = ReusableColumnMemory::new(1);
        permuted_lower
            .mutable_column(0)
            .add_entry(RowIndex::new(1), 2.0);
        permuted_lower
            .mutable_column(0)
            .add_entry(RowIndex::new(2), 0.0);
        let mut permuted_upper = ReusableColumnMemory::new(1);
        let mut operations = 0;

        let lower = compute_column(
            MatrixView::full(&matrix),
            0,
            &row_permutation,
            &mut lower_factor,
            &mut candidates,
            &mut permuted_lower,
            &mut permuted_upper,
            2,
            &mut operations,
        );

        assert_eq!(lower.num_entries(), 1);
        assert_eq!(lower.entry(0).index(), RowIndex::new(2));
        assert_eq!(lower.entry(0).coefficient().to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            permuted_upper
                .column(0)
                .look_up_coefficient(RowIndex::new(1))
                .to_bits(),
            2.0_f64.to_bits()
        );
    }

    #[test]
    fn residual_pattern_ignores_only_nonstructural_exact_zeros() {
        let mut pattern = MatrixNonZeroPattern {
            row_nonzeros: vec![Vec::new(), vec![0], vec![1]],
            row_degree: vec![0, 1, 1],
            column_degree: vec![1, 1],
            deleted_columns: vec![true, false],
            scratchpad: vec![false; 2],
            non_deleted_columns: 1,
        };
        let mut residual = SparseColumn::new();
        residual.add_entry(RowIndex::new(1), 0.0);
        residual.add_entry(RowIndex::new(2), 0.0);
        let mut singletons = Vec::new();

        pattern.remove_column(0, &residual, &mut singletons);

        assert_eq!(pattern.row_degree, [0, 0, 1]);
    }

    #[test]
    fn residual_singleton_ignores_a_reachability_overestimate_stored_first() {
        let pattern = MatrixNonZeroPattern {
            row_nonzeros: vec![Vec::new(), vec![2]],
            row_degree: vec![0, 1],
            column_degree: vec![0, 0, 1],
            deleted_columns: vec![false; 3],
            scratchpad: vec![false; 3],
            non_deleted_columns: 3,
        };
        let mut residual = SparseColumn::new();
        residual.add_entry(RowIndex::new(0), 0.0);
        residual.add_entry(RowIndex::new(1), 3.0);

        assert_eq!(structural_singleton_row(&residual, &pattern, 2), Some(1));
    }

    #[test]
    fn factorization_drops_exact_cancellations_from_lower_factor() {
        // Eliminating the first column cancels one entry of the second
        // column exactly. Upstream's AddAndNormalizeTriangularColumn() does
        // not store that zero in L.
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(3), ColIndex::new(3));
        for (column, entries) in [
            [(0, 1.0), (1, 1.0)].as_slice(),
            [(0, 1.0), (1, 1.0), (2, 1.0)].as_slice(),
            [(1, 1.0), (2, 1.0)].as_slice(),
        ]
        .into_iter()
        .enumerate()
        {
            for &(row, coefficient) in entries {
                matrix
                    .mutable_column(ColIndex::from_usize(column))
                    .add_entry(RowIndex::from_usize(row), coefficient);
            }
        }

        let mut lower = TriangularMatrix::empty(Triangle::Lower, true);
        let mut upper = TriangularMatrix::empty(Triangle::Upper, false);
        factorize(
            &matrix,
            &GlopParameters::default(),
            &mut MarkowitzWorkspace::default(),
            &mut lower,
            &mut upper,
        )
        .unwrap();

        assert!(
            (0..lower.num_cols())
                .flat_map(|column| lower.column(column))
                .all(|(_, coefficient)| coefficient != 0.0)
        );
    }

    #[test]
    fn workspace_reuse_preserves_factors_after_a_different_basis() {
        let build = |columns: [&[(usize, f64)]; 3]| {
            let mut matrix = SparseMatrix::new();
            matrix.populate_from_zero(RowIndex::new(3), ColIndex::new(3));
            for (column, entries) in columns.into_iter().enumerate() {
                for &(row, coefficient) in entries {
                    matrix
                        .mutable_column(ColIndex::from_usize(column))
                        .add_entry(RowIndex::from_usize(row), coefficient);
                }
            }
            matrix
        };
        let first = build([
            &[(0, 4.0), (1, 1.0)],
            &[(0, 1.0), (1, 5.0), (2, 1.0)],
            &[(1, 1.0), (2, 6.0)],
        ]);
        let second = build([
            &[(0, 2.0), (1, 1.0), (2, 1.0)],
            &[(0, 1.0), (1, 3.0), (2, 1.0)],
            &[(0, 1.0), (1, 1.0), (2, 4.0)],
        ]);
        let parameters = GlopParameters::default();
        let mut workspace = MarkowitzWorkspace::default();
        let mut lower = TriangularMatrix::empty(Triangle::Lower, true);
        let mut upper = TriangularMatrix::empty(Triangle::Upper, false);
        let expected =
            factorize(&first, &parameters, &mut workspace, &mut lower, &mut upper).unwrap();
        let expected_lower: Vec<_> = (0..3)
            .map(|column| lower.column(column).collect::<Vec<_>>())
            .collect();
        let expected_upper: Vec<_> = (0..3)
            .map(|column| {
                (
                    upper.diagonal(column),
                    upper.column(column).collect::<Vec<_>>(),
                )
            })
            .collect();
        factorize(&second, &parameters, &mut workspace, &mut lower, &mut upper).unwrap();
        let actual =
            factorize(&first, &parameters, &mut workspace, &mut lower, &mut upper).unwrap();
        assert_eq!(
            (0..3)
                .map(|column| lower.column(column).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            expected_lower
        );
        assert_eq!(
            (0..3)
                .map(|column| (
                    upper.diagonal(column),
                    upper.column(column).collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            expected_upper
        );
        assert_eq!(actual.row_permutation, expected.row_permutation);
        assert_eq!(actual.column_permutation, expected.column_permutation);
        assert_eq!(actual.num_fp_operations, expected.num_fp_operations);
    }
}
