//! Basis factorization and product-form updates.
//!
//! Like upstream `basis_representation`, this keeps a fresh LU plus eta updates
//! and can refactorize from the current basis when the update chain grows.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use lp_data::lp_types::{
    ColIndex, RowIndex, RowToColMapping, VectorIndex, deterministic_time_for_fp_operations,
};
use lp_data::lp_utils::clear_and_resize_vector_with_non_zeros;
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};
use lp_data::sparse_vector::SparseColumn;

use crate::lu_factorization::{FactorizationError, LuFactorization};
use crate::parameters::GlopParameters;
use crate::rank_one_update::{
    RankOneUpdateElementaryMatrix, RankOneUpdateFactorization, sparse_scalar_product,
};
use crate::stats::{DistributionKind, StatsGroup};

/// Classical product-form update from upstream `basis_representation.h`.
#[derive(Clone, Debug)]
struct EtaMatrix {
    column: usize,
    pivot: f64,
    coefficients: Vec<f64>,
    sparse_coefficients: Option<Vec<(usize, f64)>>,
}

impl EtaMatrix {
    fn new(column: usize, direction: &ScatteredColumn) -> Result<Self, &'static str> {
        if column >= direction.len().to_usize() {
            return Err("eta column is out of range");
        }
        let pivot = direction.value(RowIndex::from_usize(column));
        if pivot == 0.0 {
            return Err("eta update has a zero pivot");
        }
        let mut coefficients = direction.values().as_slice().to_vec();
        coefficients[column] = 0.0;
        #[allow(clippy::cast_precision_loss)]
        let use_sparse = (direction.non_zeros().len() as f64) < 0.5 * coefficients.len() as f64;
        let sparse_coefficients = use_sparse.then(|| {
            direction
                .non_zeros()
                .iter()
                .map(|row| row.to_usize())
                .filter(|&row| row != column)
                .map(|row| (row, coefficients[row]))
                .collect::<Vec<_>>()
        });
        // Upstream uses an empty SparseColumn to select the dense eta column.
        let sparse_coefficients = sparse_coefficients.filter(|entries| !entries.is_empty());
        Ok(Self {
            column,
            pivot,
            coefficients,
            sparse_coefficients,
        })
    }

    fn left_solve(&self, values: &mut [f64]) {
        let mut value = values[self.column];
        if let Some(entries) = &self.sparse_coefficients {
            for &(row, coefficient) in entries {
                value = (-values[row]).mul_add(coefficient, value);
            }
        } else {
            for (row, &coefficient) in self.coefficients.iter().enumerate() {
                value = (-values[row]).mul_add(coefficient, value);
            }
        }
        values[self.column] = value / self.pivot;
    }

    fn sparse_left_solve(&self, values: &mut [f64], positions: &mut Vec<ColIndex>) {
        let mut value = values[self.column];
        let mut contains_column = false;
        let original_size = positions.len();
        for &position in &positions[..original_size] {
            let column = position.to_usize();
            if column == self.column {
                contains_column = true;
            } else {
                value = (-values[column]).mul_add(self.coefficients[column], value);
            }
        }
        values[self.column] = value / self.pivot;
        if !contains_column {
            positions.push(ColIndex::from_usize(self.column));
        }
    }

    fn right_solve(&self, values: &mut [f64]) {
        if values[self.column] == 0.0 {
            return;
        }
        let multiplier = values[self.column] / self.pivot;
        if let Some(entries) = &self.sparse_coefficients {
            for &(row, coefficient) in entries {
                values[row] = (-coefficient).mul_add(multiplier, values[row]);
            }
        } else {
            for (row, &coefficient) in self.coefficients.iter().enumerate() {
                values[row] = (-coefficient).mul_add(multiplier, values[row]);
            }
        }
        values[self.column] = multiplier;
    }
}

#[derive(Clone, Debug, Default)]
struct EtaFactorization {
    matrices: Vec<EtaMatrix>,
}

impl EtaFactorization {
    fn clear(&mut self) {
        self.matrices.clear();
    }

    fn update(&mut self, matrix: EtaMatrix) {
        self.matrices.push(matrix);
    }

    fn len(&self) -> usize {
        self.matrices.len()
    }

    fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    fn left_solve(&self, values: &mut [f64]) {
        for matrix in self.matrices.iter().rev() {
            matrix.left_solve(values);
        }
    }

    fn sparse_left_solve(&self, values: &mut [f64], positions: &mut Vec<ColIndex>) {
        for matrix in self.matrices.iter().rev() {
            matrix.sparse_left_solve(values, positions);
        }
    }

    fn right_solve(&self, values: &mut [f64]) {
        for matrix in &self.matrices {
            matrix.right_solve(values);
        }
    }
}

#[derive(Clone, Debug)]
struct TauState {
    value: ScatteredColumn,
    computation_can_be_optimized: bool,
    is_computed: bool,
}

#[derive(Clone, Debug)]
enum BasisMatrix {
    Owned(SparseMatrix),
    View {
        matrix: Rc<SparseMatrix>,
        columns: Vec<usize>,
    },
}

enum ReplacementColumn {
    Owned(SparseColumn),
    Problem(ColIndex),
}

impl BasisMatrix {
    fn dimension(&self) -> usize {
        match self {
            Self::Owned(matrix) => matrix.num_rows().to_usize(),
            Self::View { columns, .. } => columns.len(),
        }
    }

    fn install(&mut self, position: usize, replacement: ReplacementColumn) {
        match (self, replacement) {
            (Self::Owned(matrix), ReplacementColumn::Owned(column)) => {
                matrix.replace_column(ColIndex::from_usize(position), column);
            }
            (Self::View { columns, .. }, ReplacementColumn::Problem(column)) => {
                columns[position] = column.to_usize();
            }
            _ => panic!("replacement column does not match the basis representation"),
        }
    }

    fn apply_column_permutation(&mut self, permutation: &[usize]) {
        match self {
            Self::Owned(matrix) => matrix.apply_column_permutation(permutation),
            Self::View { columns, .. } => {
                let old = columns.clone();
                for (source, &destination) in permutation.iter().enumerate() {
                    columns[destination] = old[source];
                }
            }
        }
    }

    fn factorize(
        &self,
        parameters: &GlopParameters,
    ) -> Result<LuFactorization, FactorizationError> {
        match self {
            Self::Owned(matrix) => LuFactorization::factorize_with_parameters(matrix, parameters),
            Self::View { matrix, columns } => {
                LuFactorization::factorize_selected_with_parameters(matrix, columns, parameters)
            }
        }
    }

    fn compute_factorization(
        &self,
        factorization: &mut LuFactorization,
        parameters: &GlopParameters,
    ) -> Result<(), FactorizationError> {
        match self {
            Self::Owned(matrix) => {
                factorization.compute_factorization_with_parameters(matrix, parameters)
            }
            Self::View { matrix, columns } => factorization
                .compute_factorization_selected_with_parameters(matrix, columns, parameters),
        }
    }

    fn one_norm(&self) -> f64 {
        match self {
            Self::Owned(matrix) => matrix.one_norm(),
            Self::View { matrix, columns } => columns.iter().fold(0.0, |maximum, &column| {
                maximum.max(
                    matrix
                        .column(ColIndex::from_usize(column))
                        .iter()
                        .map(|e| e.coefficient().abs())
                        .sum(),
                )
            }),
        }
    }

    fn infinity_norm(&self) -> f64 {
        match self {
            Self::Owned(matrix) => matrix.infinity_norm(),
            Self::View { matrix, columns } => {
                let mut sums = vec![0.0; matrix.num_rows().to_usize()];
                for &column in columns {
                    for entry in matrix.column(ColIndex::from_usize(column)) {
                        sums[entry.index().to_usize()] += entry.coefficient().abs();
                    }
                }
                sums.into_iter().fold(0.0_f64, f64::max)
            }
        }
    }

    #[allow(clippy::float_cmp)] // GLOP requires an exact unit coefficient.
    fn is_identity(&self) -> bool {
        match self {
            Self::Owned(matrix) => {
                matrix.num_rows().to_usize() == matrix.num_cols().to_usize()
                    && (0..matrix.num_cols().to_usize()).all(|position| {
                        let column = matrix.column(ColIndex::from_usize(position));
                        column.num_entries() == 1
                            && column.entry(0).index().to_usize() == position
                            && column.entry(0).coefficient() == 1.0
                    })
            }
            Self::View { matrix, columns } => {
                columns.iter().enumerate().all(|(position, &source)| {
                    let column = matrix.column(ColIndex::from_usize(source));
                    column.num_entries() == 1
                        && column.entry(0).index().to_usize() == position
                        && column.entry(0).coefficient() == 1.0
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasisRepresentation {
    basis: BasisMatrix,
    factorization: LuFactorization,
    updates: RankOneUpdateFactorization,
    eta_updates: EtaFactorization,
    parameters: GlopParameters,
    max_updates: usize,
    last_factorization_deterministic_time: f64,
    deterministic_time: Cell<f64>,
    tau: RefCell<TauState>,
    left_storage: RefCell<CompactSparseMatrix>,
    left_pool_mapping: RefCell<Vec<Option<ColIndex>>>,
    right_storage: RefCell<CompactSparseMatrix>,
    right_pool_mapping: RefCell<Vec<Option<ColIndex>>>,
    stats: StatsGroup,
}

impl BasisRepresentation {
    /// Factorizes an initial square basis.
    ///
    /// # Errors
    ///
    /// Propagates matrix factorization errors.
    pub fn new(
        basis: SparseMatrix,
        pivot_threshold: f64,
        max_updates: usize,
    ) -> Result<Self, FactorizationError> {
        let parameters = GlopParameters {
            lu_factorization_pivot_threshold: pivot_threshold,
            basis_refactorization_period: i32::try_from(max_updates).unwrap_or(i32::MAX),
            ..GlopParameters::default()
        };
        Self::new_with_parameters(basis, &parameters)
    }

    /// Factorizes an initial basis using GLOP's shared parameter bundle.
    ///
    /// # Errors
    ///
    /// Propagates parameter and matrix factorization errors.
    #[allow(clippy::float_cmp)] // GLOP's IsIdentityBasis() requires an exact one.
    pub fn new_with_parameters(
        basis: SparseMatrix,
        parameters: &GlopParameters,
    ) -> Result<Self, FactorizationError> {
        Self::initialize(BasisMatrix::Owned(basis), parameters)
    }

    pub(crate) fn new_for_basis(
        matrix: Rc<SparseMatrix>,
        columns: &RowToColMapping,
        parameters: &GlopParameters,
    ) -> Result<Self, FactorizationError> {
        let columns = columns
            .as_slice()
            .iter()
            .map(|column| column.to_usize())
            .collect();
        Self::initialize(BasisMatrix::View { matrix, columns }, parameters)
    }

    /// Reinitializes this factorization for a replacement basis.
    ///
    /// GLOP uses the same `BasisFactorization` object when an advanced crash
    /// basis is rejected. Its `Clear()` deliberately retains the cumulative
    /// deterministic clock and the last factorization cost, which the dynamic
    /// refactorization-period heuristic subsequently uses. Preserve that
    /// lifecycle instead of replacing the whole object.
    pub(crate) fn reinitialize_for_basis(
        &mut self,
        matrix: Rc<SparseMatrix>,
        columns: &RowToColMapping,
        parameters: &GlopParameters,
    ) -> Result<(), FactorizationError> {
        let deterministic_time = self.deterministic_time.get();
        let last_factorization_deterministic_time = self.last_factorization_deterministic_time;
        let mut replacement = Self::new_for_basis(matrix, columns, parameters)?;
        replacement
            .deterministic_time
            .set(deterministic_time + replacement.deterministic_time.get());
        if replacement.last_factorization_deterministic_time == 0.0 {
            replacement.last_factorization_deterministic_time =
                last_factorization_deterministic_time;
        }
        std::mem::swap(&mut replacement.stats, &mut self.stats);
        *self = replacement;
        Ok(())
    }

    fn initialize(
        basis: BasisMatrix,
        parameters: &GlopParameters,
    ) -> Result<Self, FactorizationError> {
        #[allow(clippy::float_cmp)] // GLOP requires an exact unit coefficient.
        let is_identity_basis = match &basis {
            BasisMatrix::Owned(matrix) => {
                matrix.num_rows().to_usize() == matrix.num_cols().to_usize()
                    && (0..matrix.num_cols().to_usize()).all(|column| {
                        let basis_column = matrix.column(ColIndex::from_usize(column));
                        basis_column.num_entries() == 1
                            && basis_column.entry(0).index().to_usize() == column
                            && basis_column.entry(0).coefficient() == 1.0
                    })
            }
            BasisMatrix::View { matrix, columns } => {
                columns.iter().enumerate().all(|(row, &column)| {
                    let basis_column = matrix.column(ColIndex::from_usize(column));
                    basis_column.num_entries() == 1
                        && basis_column.entry(0).index().to_usize() == row
                        && basis_column.entry(0).coefficient() == 1.0
                })
            }
        };
        let factorization = if is_identity_basis {
            // BasisFactorization::Initialize() leaves its cleared LU object in
            // the dimension-independent identity state for a slack basis.
            LuFactorization::new()
        } else {
            basis.factorize(parameters)?
        };
        let last_factorization_deterministic_time =
            factorization.deterministic_time_of_last_factorization();
        let dimension = basis.dimension();
        let updates = RankOneUpdateFactorization::default();
        updates.reset_deterministic_time();
        let mut left_storage = CompactSparseMatrix::default();
        left_storage.reset(lp_data::lp_types::RowIndex::from_usize(dimension));
        let mut right_storage = CompactSparseMatrix::default();
        right_storage.reset(lp_data::lp_types::RowIndex::from_usize(dimension));
        Ok(Self {
            basis,
            factorization,
            updates,
            eta_updates: EtaFactorization::default(),
            parameters: parameters.clone(),
            max_updates: usize::try_from(parameters.basis_refactorization_period)
                .unwrap_or(usize::MAX),
            last_factorization_deterministic_time,
            deterministic_time: Cell::new(last_factorization_deterministic_time),
            tau: RefCell::new(TauState {
                value: ScatteredColumn::new(lp_data::lp_types::RowIndex::from_usize(dimension)),
                computation_can_be_optimized: false,
                is_computed: false,
            }),
            left_storage: RefCell::new(left_storage),
            left_pool_mapping: RefCell::new(Vec::new()),
            right_storage: RefCell::new(right_storage),
            right_pool_mapping: RefCell::new(Vec::new()),
            stats: StatsGroup::new("BasisFactorization"),
        })
    }

    /// Updates the controls used by subsequent updates and refactorizations.
    ///
    /// # Errors
    ///
    /// Returns an invalid-parameter error.
    pub fn set_parameters(
        &mut self,
        parameters: &GlopParameters,
    ) -> Result<(), FactorizationError> {
        parameters
            .validate()
            .map_err(FactorizationError::InvalidParameters)?;
        self.parameters = parameters.clone();
        self.max_updates =
            usize::try_from(parameters.basis_refactorization_period).unwrap_or(usize::MAX);
        Ok(())
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.basis.dimension()
    }

    /// Returns the materialized matrix used by the standalone owning path.
    ///
    /// # Panics
    ///
    /// Panics for the simplex view-based representation.
    #[must_use]
    pub fn basis(&self) -> &SparseMatrix {
        match &self.basis {
            BasisMatrix::Owned(matrix) => matrix,
            BasisMatrix::View { .. } => panic!("a viewed basis is not materialized"),
        }
    }

    #[must_use]
    pub fn column_permutation(&self) -> &[usize] {
        self.factorization.column_permutation()
    }

    /// Clears the LU column permutation after it has been incorporated into
    /// the caller's row-indexed basis mapping, as `RevisedSimplex::PermuteBasis`
    /// does upstream.
    pub fn set_column_permutation_to_identity(&mut self) {
        self.basis
            .apply_column_permutation(self.factorization.column_permutation());
        self.factorization.set_column_permutation_to_identity();
        self.clear_partial_solve_storage();
    }

    /// Resets the numerical representation to GLOP's dimension-independent
    /// identity state without changing the referenced basis or accumulated
    /// deterministic time.
    pub fn clear(&mut self) {
        self.updates.clear();
        self.updates.reset_deterministic_time();
        self.eta_updates.clear();
        self.factorization.clear();
        self.clear_partial_solve_storage();
        self.tau.get_mut().computation_can_be_optimized = false;
    }

    /// Solves `B x = rhs` through the LU and subsequent eta updates.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if !self.parameters.use_middle_product_form_update {
            let mut result = self.factorization.solve(rhs)?;
            self.eta_updates.right_solve(&mut result);
            self.bump_deterministic_time_for_solve(self.dimension());
            return Ok(result);
        }
        let mut result = self.factorization.right_solve_lower(rhs)?;
        self.updates.right_solve(&mut result);
        let result = self.factorization.right_solve_upper(&result)?;
        self.bump_deterministic_time_for_solve(self.dimension());
        Ok(result)
    }

    /// Solves `B^T x = rhs` through the eta updates and LU.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn transpose_solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FactorizationError> {
        if rhs.len() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        if !self.parameters.use_middle_product_form_update {
            let mut transformed = rhs.to_vec();
            self.eta_updates.left_solve(&mut transformed);
            let result = self.factorization.transpose_solve(&transformed)?;
            self.bump_deterministic_time_for_solve(self.dimension());
            return Ok(result);
        }
        let mut transformed = self.factorization.left_solve_upper(rhs)?;
        self.updates.left_solve(&mut transformed);
        let result = self.factorization.left_solve_lower(&transformed)?;
        self.bump_deterministic_time_for_solve(self.dimension());
        Ok(result)
    }

    /// Sparse/scattered counterpart of [`Self::solve`].
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn solve_with_nonzeros(&self, rhs: &mut ScatteredColumn) -> Result<(), FactorizationError> {
        if !self.parameters.use_middle_product_form_update {
            if rhs.len().to_usize() != self.dimension() {
                return Err(FactorizationError::DimensionMismatch);
            }
            let mut result = self.factorization.solve(rhs.values().as_slice())?;
            self.eta_updates.right_solve(&mut result);
            rhs.values_mut().as_mut_slice().copy_from_slice(&result);
            rhs.non_zeros_mut().clear();
            self.bump_deterministic_time_for_solve(self.dimension());
            return Ok(());
        }
        self.factorization.right_solve_lower_with_nonzeros(rhs)?;
        self.updates.right_solve_with_nonzeros(rhs);
        self.factorization.right_solve_upper_with_nonzeros(rhs)?;
        rhs.sort_non_zeros_if_needed();
        self.bump_deterministic_time_for_solve(rhs.num_non_zeros_estimate());
        Ok(())
    }

    /// Sparse/scattered counterpart of [`Self::transpose_solve`].
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the factorization.
    pub fn transpose_solve_with_nonzeros(
        &self,
        rhs: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        if !self.parameters.use_middle_product_form_update {
            if rhs.len().to_usize() != self.dimension() {
                return Err(FactorizationError::DimensionMismatch);
            }
            let mut transformed = rhs.values().as_slice().to_vec();
            self.eta_updates.left_solve(&mut transformed);
            let result = self.factorization.transpose_solve(&transformed)?;
            rhs.values_mut().as_mut_slice().copy_from_slice(&result);
            rhs.non_zeros_mut().clear();
            self.bump_deterministic_time_for_solve(self.dimension());
            return Ok(());
        }
        self.factorization.left_solve_upper_with_nonzeros(rhs)?;
        self.updates.left_solve_with_nonzeros(rhs);
        self.factorization.left_solve_lower_with_nonzeros(rhs)?;
        self.bump_deterministic_time_for_solve(rhs.num_non_zeros_estimate());
        Ok(())
    }

    /// Computes `B^-T e_j`, retaining GLOP's optional tau intermediate.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch or a triangular-solve error.
    pub fn left_solve_for_unit_row(
        &self,
        row: usize,
        result: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        if row >= self.dimension() || result.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        clear_and_resize_vector_with_non_zeros(ColIndex::from_usize(self.dimension()), result);
        if !self.parameters.use_middle_product_form_update {
            let column = ColIndex::from_usize(row);
            let (values, positions) = result.mutable_parts();
            values[row] = 1.0;
            positions.push(column);
            {
                let (values, positions) = result.mutable_parts();
                self.eta_updates.sparse_left_solve(values, positions);
            }
            let solved = self
                .factorization
                .transpose_solve(result.values().as_slice())?;
            result.values_mut().as_mut_slice().copy_from_slice(&solved);
            self.bump_deterministic_time_for_solve(result.num_non_zeros_estimate());
            return Ok(());
        }
        let stored_column = {
            let mut mapping = self.left_pool_mapping.borrow_mut();
            if mapping.len() <= row {
                mapping.resize(row + 1, None);
            }
            mapping[row]
        };
        if let Some(column) = stored_column {
            let storage = self.left_storage.borrow();
            let (values, positions) = result.mutable_parts();
            for (stored_row, value) in storage.column(column).iter() {
                values[stored_row.to_usize()] = value;
                positions.push(ColIndex::from_usize(stored_row.to_usize()));
            }
        } else {
            let start = self
                .factorization
                .left_solve_upper_for_unit_row(row, result)?;
            result.sort_non_zeros_if_needed();
            let mut storage = self.left_storage.borrow_mut();
            let column = storage.num_cols();
            if result.non_zeros().is_empty() {
                for (position, &value) in result.values().as_slice().iter().enumerate().skip(start)
                {
                    if value != 0.0 {
                        storage.add_entry_to_current_column(
                            lp_data::lp_types::RowIndex::from_usize(position),
                            value,
                        );
                    }
                }
            } else {
                for entry in result.iter() {
                    if entry.coefficient() != 0.0 {
                        storage.add_entry_to_current_column(
                            lp_data::lp_types::RowIndex::from_usize(entry.index().to_usize()),
                            entry.coefficient(),
                        );
                    }
                }
            }
            storage.close_current_column();
            self.left_pool_mapping.borrow_mut()[row] = Some(column);
        }
        self.updates.left_solve_with_nonzeros(result);

        let mut tau = self.tau.borrow_mut();
        tau.computation_can_be_optimized = if tau.is_computed {
            self.factorization
                .left_solve_lower_with_nonzeros_and_cache(result, &mut tau.value)?
        } else {
            self.factorization.left_solve_lower_with_nonzeros(result)?;
            false
        };
        tau.is_computed = false;
        result.sort_non_zeros_if_needed();
        self.bump_deterministic_time_for_solve(result.num_non_zeros_estimate());
        Ok(())
    }

    /// Computes `B^-T e_j` without changing tau-reuse state.
    ///
    /// As in GLOP, this specialized temporary solve is valid only immediately
    /// after refactorization, when no rank-one update is pending.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch or triangular-solve error.
    ///
    /// # Panics
    ///
    /// Panics if rank-one updates are pending.
    pub fn temporary_left_solve_for_unit_row(
        &self,
        row: usize,
        result: &mut ScatteredRow,
    ) -> Result<(), FactorizationError> {
        assert!(self.is_refactorized());
        if row >= self.dimension() || result.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        result.clear();
        result.set(ColIndex::from_usize(row), 1.0);
        self.factorization.left_solve_upper_with_nonzeros(result)?;
        self.factorization.left_solve_lower_with_nonzeros(result)?;
        result.sort_non_zeros_if_needed();
        self.bump_deterministic_time_for_solve(result.num_non_zeros_estimate());
        Ok(())
    }

    /// Computes `B^-1 a` for dual steepest-edge tau, reusing the intermediate
    /// retained by the preceding [`Self::left_solve_for_unit_row`] when GLOP
    /// would do so.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch or a triangular-solve error.
    pub fn right_solve_for_tau(
        &self,
        input: &ScatteredRow,
    ) -> Result<Vec<f64>, FactorizationError> {
        if input.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        let mut tau = self.tau.borrow_mut();
        if !self.parameters.use_middle_product_form_update {
            let mut solved = self.factorization.solve(input.values().as_slice())?;
            self.eta_updates.right_solve(&mut solved);
            tau.value.clear();
            tau.value
                .values_mut()
                .as_mut_slice()
                .copy_from_slice(&solved);
            tau.value.non_zeros_mut().clear();
            tau.is_computed = true;
            self.bump_deterministic_time_for_solve(tau.value.num_non_zeros_estimate());
            return Ok(solved);
        }
        if tau.computation_can_be_optimized {
            tau.computation_can_be_optimized = false;
            self.factorization
                .right_solve_lower_with_permuted_input(&mut tau.value)?;
        } else {
            tau.value.clear();
            if input.non_zeros().is_empty() {
                tau.value
                    .values_mut()
                    .as_mut_slice()
                    .copy_from_slice(input.values().as_slice());
                tau.value.non_zeros_mut().clear();
            } else {
                for entry in input {
                    tau.value.set(
                        lp_data::lp_types::RowIndex::from_usize(entry.index().to_usize()),
                        entry.coefficient(),
                    );
                }
            }
            self.factorization
                .right_solve_lower_with_nonzeros(&mut tau.value)?;
        }
        self.updates.right_solve_with_nonzeros(&mut tau.value);
        self.factorization
            .right_solve_upper_with_nonzeros(&mut tau.value)?;
        tau.is_computed = true;
        self.bump_deterministic_time_for_solve(tau.value.num_non_zeros_estimate());
        Ok(tau.value.values().as_slice().to_vec())
    }

    /// Solves for a problem column and retains the post-lower, pre-upper
    /// intermediate consumed by a subsequent middle-product update.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch or triangular-solve error.
    pub fn right_solve_for_problem_column(
        &self,
        problem_column: usize,
        column: &SparseColumn,
        result: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        if result.len().to_usize() != self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        if !self.parameters.use_middle_product_form_update {
            let mut dense = vec![0.0; self.dimension()];
            for entry in column {
                if entry.index().to_usize() >= self.dimension() {
                    return Err(FactorizationError::DimensionMismatch);
                }
                dense[entry.index().to_usize()] = entry.coefficient();
            }
            let mut solved = self.factorization.solve(&dense)?;
            self.eta_updates.right_solve(&mut solved);
            result.clear();
            result.values_mut().as_mut_slice().copy_from_slice(&solved);
            result.non_zeros_mut().clear();
            self.bump_deterministic_time_for_solve(result.num_non_zeros_estimate());
            return Ok(());
        }
        self.factorization
            .right_solve_lower_for_column(column, result)?;
        self.updates.right_solve_with_nonzeros(result);

        let mut storage = self.right_storage.borrow_mut();
        let stored_column = storage.num_cols();
        if result.non_zeros().is_empty() {
            for (position, &value) in result.values().as_slice().iter().enumerate() {
                if value != 0.0 {
                    storage.add_entry_to_current_column(
                        lp_data::lp_types::RowIndex::from_usize(position),
                        value,
                    );
                }
            }
        } else {
            result.sort_non_zeros_if_needed();
            for entry in result.iter() {
                if entry.coefficient() != 0.0 {
                    storage.add_entry_to_current_column(entry.index(), entry.coefficient());
                }
            }
        }
        storage.close_current_column();
        drop(storage);
        let mut mapping = self.right_pool_mapping.borrow_mut();
        if mapping.len() <= problem_column {
            mapping.resize(problem_column + 1, None);
        }
        mapping[problem_column] = Some(stored_column);

        self.factorization.right_solve_upper_with_nonzeros(result)?;
        result.sort_non_zeros_if_needed();
        self.bump_deterministic_time_for_solve(result.num_non_zeros_estimate());
        Ok(())
    }

    /// Replaces a basis column and records its product-form inverse update.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid pivot or failed refactorization.
    pub fn replace_column_after_solve(
        &mut self,
        problem_column: usize,
        leaving_column: usize,
        direction: &ScatteredColumn,
        mut entering_column: SparseColumn,
    ) -> Result<(), FactorizationError> {
        entering_column.clean_up();
        self.replace_column_after_solve_impl(
            problem_column,
            leaving_column,
            direction,
            ReplacementColumn::Owned(entering_column),
        )
    }

    pub(crate) fn update_after_solve(
        &mut self,
        problem_column: ColIndex,
        leaving_column: usize,
        direction: &ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        self.replace_column_after_solve_impl(
            problem_column.to_usize(),
            leaving_column,
            direction,
            ReplacementColumn::Problem(problem_column),
        )
    }

    fn replace_column_after_solve_impl(
        &mut self,
        problem_column: usize,
        leaving_column: usize,
        direction: &ScatteredColumn,
        entering_column: ReplacementColumn,
    ) -> Result<(), FactorizationError> {
        if self.parameters.use_middle_product_form_update {
            return self.replace_column_from_partial_solves_impl(
                problem_column,
                leaving_column,
                entering_column,
            );
        }
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.tau.get_mut().computation_can_be_optimized = false;
        if self.num_updates() >= self.max_updates
            && (!self.parameters.dynamically_adjust_refactorization_period
                || self.last_factorization_deterministic_time
                    < self.updates.deterministic_time_since_last_reset())
        {
            self.basis.install(leaving_column, entering_column);
            self.force_refactorization()?;
            return Ok(());
        }
        let update = EtaMatrix::new(leaving_column, direction).map_err(|_| {
            FactorizationError::Singular {
                step: leaving_column,
            }
        })?;
        self.basis.install(leaving_column, entering_column);
        self.eta_updates.update(update);
        Ok(())
    }

    /// Replaces a basis column and records its product-form inverse update.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid pivot or failed refactorization.
    pub fn replace_column(
        &mut self,
        leaving_column: usize,
        mut entering_column: SparseColumn,
    ) -> Result<(), FactorizationError> {
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        entering_column.clean_up();
        if !self.parameters.use_middle_product_form_update {
            let mut direction =
                ScatteredColumn::new(lp_data::lp_types::RowIndex::from_usize(self.dimension()));
            for entry in &entering_column {
                direction.set(entry.index(), entry.coefficient());
            }
            self.solve_with_nonzeros(&mut direction)?;
            let mut unit_left_inverse = ScatteredRow::new(ColIndex::from_usize(self.dimension()));
            self.left_solve_for_unit_row(leaving_column, &mut unit_left_inverse)?;
            return self.replace_column_after_solve(0, leaving_column, &direction, entering_column);
        }
        self.tau.get_mut().computation_can_be_optimized = false;
        let mut right_update =
            ScatteredColumn::new(lp_data::lp_types::RowIndex::from_usize(self.dimension()));
        for entry in &entering_column {
            right_update.set(entry.index(), entry.coefficient());
        }
        // With B = P^-1 L R U Q, replacing input-coordinate column e by a
        // gives R_new = R (I + u v^T), where
        //   u = R^-1 L^-1 P a - U Q e,
        //   v^T = e^T Q^-1 U^-1.
        // This is GLOP's default middle-product-form update.
        self.factorization
            .right_solve_lower_with_nonzeros(&mut right_update)?;
        self.updates.right_solve_with_nonzeros(&mut right_update);
        // RevisedSimplex has already computed the full entering direction via
        // RightSolveForProblemColumn() before Update(). Preserve that work and
        // its deterministic-time charge even though the middle-product update
        // itself retains the pre-U intermediate above.
        let mut direction = right_update.clone();
        self.factorization
            .right_solve_upper_with_nonzeros(&mut direction)?;
        self.bump_deterministic_time_for_solve(direction.num_non_zeros_estimate());
        let mut left_update = ScatteredRow::new(ColIndex::from_usize(self.dimension()));
        left_update.set(ColIndex::from_usize(leaving_column), 1.0);
        self.factorization
            .left_solve_upper_with_nonzeros(&mut left_update)?;
        // RevisedSimplex computes the complete leaving-row inverse before
        // calling BasisFactorization::Update(). Its pass through the rank-one
        // factors contributes to the dynamic refactorization clock, although
        // the middle-product update itself retains the pre-R solve above.
        let mut unit_left_inverse = left_update.clone();
        self.updates
            .left_solve_with_nonzeros(&mut unit_left_inverse);
        self.factorization
            .left_solve_lower_with_nonzeros(&mut unit_left_inverse)?;
        self.bump_deterministic_time_for_solve(unit_left_inverse.num_non_zeros_estimate());

        self.finish_column_replacement(
            leaving_column,
            ReplacementColumn::Owned(entering_column),
            right_update,
            &left_update,
        )
    }

    /// Installs a replacement basis column and rebuilds the LU factors.
    ///
    /// This is the branch used by `RevisedSimplex::UpdateAndPivot()` when the
    /// pivot computed by FTRAN disagrees with the independently computed
    /// update-row pivot. No eta or middle-product update is retained.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or a factorization failure.
    pub fn replace_column_and_refactorize(
        &mut self,
        leaving_column: usize,
        mut entering_column: SparseColumn,
    ) -> Result<(), FactorizationError> {
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        entering_column.clean_up();
        self.basis
            .install(leaving_column, ReplacementColumn::Owned(entering_column));
        self.force_refactorization()
    }

    pub(crate) fn update_and_refactorize(
        &mut self,
        leaving_column: usize,
        entering_column: ColIndex,
    ) -> Result<(), FactorizationError> {
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.basis
            .install(leaving_column, ReplacementColumn::Problem(entering_column));
        self.force_refactorization()
    }

    /// Consumes partial solves previously produced for a problem column and
    /// leaving row, matching GLOP's `Update()`/middle-product protocol.
    ///
    /// If either partial result is absent, GLOP abandons the rank-one update
    /// and refactorizes after installing the new basis column; this method does
    /// the same.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid pivot or failed refactorization.
    pub fn replace_column_from_partial_solves(
        &mut self,
        problem_column: usize,
        leaving_column: usize,
        mut entering_column: SparseColumn,
    ) -> Result<(), FactorizationError> {
        entering_column.clean_up();
        self.replace_column_from_partial_solves_impl(
            problem_column,
            leaving_column,
            ReplacementColumn::Owned(entering_column),
        )
    }

    fn replace_column_from_partial_solves_impl(
        &mut self,
        problem_column: usize,
        leaving_column: usize,
        entering_column: ReplacementColumn,
    ) -> Result<(), FactorizationError> {
        if leaving_column >= self.dimension() {
            return Err(FactorizationError::DimensionMismatch);
        }
        self.tau.get_mut().computation_can_be_optimized = false;
        let right_index = self
            .right_pool_mapping
            .get_mut()
            .get(problem_column)
            .copied()
            .flatten();
        let left_index = self
            .left_pool_mapping
            .get_mut()
            .get(leaving_column)
            .copied()
            .flatten();
        let (Some(right_index), Some(left_index)) = (right_index, left_index) else {
            self.basis.install(leaving_column, entering_column);
            self.force_refactorization()?;
            return Ok(());
        };
        let mut right_update =
            ScatteredColumn::new(lp_data::lp_types::RowIndex::from_usize(self.dimension()));
        for (row, value) in self.right_storage.get_mut().column(right_index).iter() {
            right_update.set(row, value);
        }
        let mut left_update = ScatteredRow::new(ColIndex::from_usize(self.dimension()));
        for (row, value) in self.left_storage.get_mut().column(left_index).iter() {
            left_update.set(ColIndex::from_usize(row.to_usize()), value);
        }
        self.finish_column_replacement(leaving_column, entering_column, right_update, &left_update)
    }

    fn finish_column_replacement(
        &mut self,
        leaving_column: usize,
        entering_column: ReplacementColumn,
        mut right_update: ScatteredColumn,
        left_update: &ScatteredRow,
    ) -> Result<(), FactorizationError> {
        if self.updates.len() >= self.max_updates
            && (!self.parameters.dynamically_adjust_refactorization_period
                || self.last_factorization_deterministic_time
                    < self.updates.deterministic_time_since_last_reset())
        {
            // In GLOP the caller has already changed the external basis
            // mapping when Update() elects to refactorize. This type owns its
            // basis, so install the same change before rebuilding LU. No
            // rank-one update is appended on this path.
            self.basis.install(leaving_column, entering_column);
            self.force_refactorization()?;
            return Ok(());
        }

        let right_update_is_dense = right_update.non_zeros().is_empty();
        if !right_update_is_dense {
            // The partial solve deliberately leaves the scattered membership
            // mask cleared. Rebuild it before accumulating `-U[:, p]`; unlike
            // GLOP's AddAndClearColumnWithNonZeros() storage path, the Rust
            // packed update is formed directly from the position list and
            // therefore must not admit duplicate positions here.
            right_update.repopulate_sparse_mask();
        }
        for (row, value) in self.factorization.column_of_upper(leaving_column) {
            right_update.add(lp_data::lp_types::RowIndex::from_usize(row), -value);
        }
        let u: Vec<_> = if right_update_is_dense {
            right_update
                .values()
                .as_slice()
                .iter()
                .copied()
                .enumerate()
                .filter(|entry| entry.1 != 0.0)
                .collect()
        } else {
            right_update
                .iter()
                .map(|entry| (entry.index().to_usize(), entry.coefficient()))
                .filter(|entry| entry.1 != 0.0)
                .collect()
        };
        let v: Vec<_> = if left_update.non_zeros().is_empty() {
            left_update
                .values()
                .as_slice()
                .iter()
                .copied()
                .enumerate()
                .filter(|entry| entry.1 != 0.0)
                .collect()
        } else {
            left_update
                .iter()
                .map(|entry| (entry.index().to_usize(), entry.coefficient()))
                .filter(|entry| entry.1 != 0.0)
                .collect()
        };
        // GLOP obtains this denominator from
        // CompactSparseMatrix::ColumnScalarProduct(). Preserve its
        // four-accumulator reduction order; a linear fold changes later
        // pricing ties after a sufficiently long update chain.
        let u_dot_v = sparse_scalar_product(&v, right_update.values().as_slice());
        let update = RankOneUpdateElementaryMatrix::new(u, v, u_dot_v);
        if update.is_singular() {
            return Err(FactorizationError::Singular {
                step: leaving_column,
            });
        }
        self.basis.install(leaving_column, entering_column);
        self.updates.update(update);
        Ok(())
    }

    /// Rebuilds LU only when updates are pending.
    ///
    /// # Errors
    ///
    /// Propagates factorization failures.
    pub fn refactorize(&mut self) -> Result<(), FactorizationError> {
        if self.is_refactorized() {
            return Ok(());
        }
        self.force_refactorization()
    }

    /// Rebuilds LU from the current basis even when no updates are pending.
    ///
    /// # Errors
    ///
    /// Propagates factorization failures.
    #[allow(clippy::cast_precision_loss)]
    pub fn force_refactorization(&mut self) -> Result<(), FactorizationError> {
        self.stats.add(
            "refactorization_interval",
            DistributionKind::Integer,
            self.num_updates() as f64,
        );
        self.basis
            .compute_factorization(&mut self.factorization, &self.parameters)?;
        self.last_factorization_deterministic_time = self
            .factorization
            .deterministic_time_of_last_factorization();
        self.deterministic_time
            .set(self.deterministic_time.get() + self.last_factorization_deterministic_time);
        self.updates.clear();
        self.updates.reset_deterministic_time();
        self.eta_updates.clear();
        self.clear_partial_solve_storage();
        Ok(())
    }

    /// Number of middle-product updates since the last LU factorization.
    #[must_use]
    pub fn num_updates(&self) -> usize {
        if self.parameters.use_middle_product_form_update {
            self.updates.len()
        } else {
            self.eta_updates.len()
        }
    }

    /// Whether the current representation consists of LU alone.
    #[must_use]
    pub fn is_refactorized(&self) -> bool {
        if self.parameters.use_middle_product_form_update {
            self.updates.is_empty()
        } else {
            self.eta_updates.is_empty()
        }
    }

    /// Computes `||B^-1 a||_2^2` with GLOP's input-density time charge.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the LU factorization.
    ///
    /// # Panics
    ///
    /// Panics if rank-one updates are pending.
    pub fn right_solve_squared_norm(
        &self,
        column: &SparseColumn,
    ) -> Result<f64, FactorizationError> {
        assert!(self.is_refactorized());
        let entries: Vec<_> = column
            .into_iter()
            .map(|entry| (entry.index().to_usize(), entry.coefficient()))
            .collect();
        self.bump_deterministic_time_for_solve(column.num_entries());
        self.factorization.right_solve_squared_norm(&entries)
    }

    /// Computes `||B^-T e_row||_2^2` with a unit-input time charge.
    ///
    /// # Errors
    ///
    /// Returns a dimension mismatch from the LU factorization.
    ///
    /// # Panics
    ///
    /// Panics if rank-one updates are pending.
    pub fn dual_edge_squared_norm(&self, row: usize) -> Result<f64, FactorizationError> {
        assert!(self.is_refactorized());
        self.bump_deterministic_time_for_solve(1);
        self.factorization.dual_edge_squared_norm(row)
    }

    /// Computes `||B||_1 ||B^-1||_1` exactly via basis solves.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn one_norm_condition_number(&self) -> Result<f64, FactorizationError> {
        Ok(self.one_norm() * self.inverse_one_norm()?)
    }

    #[must_use]
    pub fn one_norm(&self) -> f64 {
        if self.is_identity_basis() {
            1.0
        } else {
            self.basis.one_norm()
        }
    }

    #[must_use]
    pub fn infinity_norm(&self) -> f64 {
        if self.is_identity_basis() {
            1.0
        } else {
            self.basis.infinity_norm()
        }
    }

    /// Computes `||B^-1||_1` by solving for every unit column.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn inverse_one_norm(&self) -> Result<f64, FactorizationError> {
        if self.is_identity_basis() {
            return Ok(1.0);
        }
        let mut norm: f64 = 0.0;
        for column in 0..self.dimension() {
            let mut unit = vec![0.0; self.dimension()];
            unit[column] = 1.0;
            let inverse_column = self.solve(&unit)?;
            let column_norm: f64 = inverse_column.iter().map(|value| value.abs()).sum();
            norm = norm.max(column_norm);
        }
        Ok(norm)
    }

    /// Computes `||B^-1||_infinity` by accumulating solved unit columns.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn inverse_infinity_norm(&self) -> Result<f64, FactorizationError> {
        if self.is_identity_basis() {
            return Ok(1.0);
        }
        let mut row_sums = vec![0.0; self.dimension()];
        for column in 0..self.dimension() {
            let mut unit = vec![0.0; self.dimension()];
            unit[column] = 1.0;
            for (sum, value) in row_sums.iter_mut().zip(self.solve(&unit)?) {
                *sum += value.abs();
            }
        }
        Ok(row_sums.into_iter().fold(0.0_f64, f64::max))
    }

    /// Computes `||B||_infinity ||B^-1||_infinity`.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn infinity_norm_condition_number(&self) -> Result<f64, FactorizationError> {
        Ok(self.infinity_norm() * self.inverse_infinity_norm()?)
    }

    #[must_use]
    pub fn infinity_norm_condition_number_upper_bound(&self) -> f64 {
        if self.is_identity_basis() {
            return 1.0;
        }
        self.bump_deterministic_time_for_solve(self.dimension());
        self.infinity_norm() * self.factorization.inverse_infinity_norm_upper_bound()
    }

    #[must_use]
    pub fn deterministic_time(&self) -> f64 {
        self.deterministic_time.get()
    }

    #[must_use]
    pub fn stat_string(&self) -> String {
        self.stats.stat_string() + &self.factorization.stat_string()
    }

    pub fn reset_stats(&mut self) {
        // This intentionally does not reset LU/Markowitz statistics, matching
        // BasisFactorization::ResetStats().
        self.stats.reset();
    }

    #[must_use]
    pub fn number_of_entries_in_lu(&self) -> usize {
        self.factorization.number_of_entries()
    }

    #[must_use]
    pub fn number_of_entries_in_updates(&self) -> usize {
        self.updates.num_entries()
    }

    #[must_use]
    pub fn last_update_entry_counts(&self) -> Option<(usize, usize)> {
        self.updates.last_entry_counts()
    }

    #[allow(clippy::float_cmp)] // Pinned GLOP requires an exact unit coefficient.
    pub fn is_identity_basis(&self) -> bool {
        self.basis.is_identity()
    }

    fn bump_deterministic_time_for_solve(&self, num_entries: usize) {
        if self.dimension() == 0 {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let density = num_entries as f64 / self.dimension() as f64;
        let time = density
            * deterministic_time_for_fp_operations(
                i64::try_from(self.factorization.number_of_entries()).unwrap_or(i64::MAX),
            )
            + deterministic_time_for_fp_operations(
                i64::try_from(self.updates.num_entries()).unwrap_or(i64::MAX),
            );
        self.deterministic_time
            .set(self.deterministic_time.get() + time);
    }

    fn clear_partial_solve_storage(&mut self) {
        let dimension = self.dimension();
        self.left_storage
            .get_mut()
            .reset(lp_data::lp_types::RowIndex::from_usize(dimension));
        self.left_pool_mapping.get_mut().clear();
        self.right_storage
            .get_mut()
            .reset(lp_data::lp_types::RowIndex::from_usize(dimension));
        self.right_pool_mapping.get_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use lp_data::lp_types::RowIndex;

    use super::*;

    fn matrix(values: &[&[f64]]) -> SparseMatrix {
        let mut result = SparseMatrix::new();
        result.populate_from_zero(
            RowIndex::from_usize(values.len()),
            ColIndex::from_usize(values.len()),
        );
        for (row, entries) in values.iter().enumerate() {
            for (column, &value) in entries.iter().enumerate() {
                if value != 0.0 {
                    result
                        .mutable_column(ColIndex::from_usize(column))
                        .add_entry(RowIndex::from_usize(row), value);
                }
            }
        }
        result.clean_up();
        result
    }

    fn eta_direction(with_positions: bool) -> ScatteredColumn {
        let mut direction = ScatteredColumn::new(RowIndex::new(5));
        if with_positions {
            direction.set(RowIndex::new(1), -2.0);
            direction.set(RowIndex::new(3), 4.0);
        } else {
            direction.values_mut()[RowIndex::new(1)] = -2.0;
            direction.values_mut()[RowIndex::new(3)] = 4.0;
        }
        direction
    }

    #[test]
    fn eta_dense_sentinel_and_sparse_representation_solve_identically() {
        let dense = EtaMatrix::new(3, &eta_direction(false)).unwrap();
        let sparse = EtaMatrix::new(3, &eta_direction(true)).unwrap();

        let rhs = [1.0, 2.0, 3.0, 8.0, 5.0];
        let mut dense_right = rhs;
        let mut sparse_right = rhs;
        dense.right_solve(&mut dense_right);
        sparse.right_solve(&mut sparse_right);
        assert!(
            dense_right
                .iter()
                .zip(sparse_right)
                .all(|(dense, sparse)| dense.to_bits() == sparse.to_bits())
        );

        let mut dense_left = rhs;
        let mut sparse_left = rhs;
        dense.left_solve(&mut dense_left);
        sparse.left_solve(&mut sparse_left);
        assert!(
            dense_left
                .iter()
                .zip(sparse_left)
                .all(|(dense, sparse)| dense.to_bits() == sparse.to_bits())
        );
    }

    #[test]
    fn tau_solve_preserves_dense_sentinel_input() {
        let basis_matrix = matrix(&[&[2.0, 0.0], &[1.0, 3.0]]);
        let basis = BasisRepresentation::new(basis_matrix, 0.01, 64).unwrap();
        let mut input = ScatteredRow::new(ColIndex::new(2));
        input.values_mut()[ColIndex::new(0)] = 4.0;
        input.values_mut()[ColIndex::new(1)] = 7.0;

        let expected = basis.solve(input.values().as_slice()).unwrap();
        let actual = basis.right_solve_for_tau(&input).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn optimized_tau_cache_preserves_dense_sentinel() {
        let basis_matrix = matrix(&[&[2.0, 0.0], &[1.0, 3.0]]);
        let basis = BasisRepresentation::new(basis_matrix, 0.01, 64).unwrap();
        let mut input = ScatteredRow::new(ColIndex::new(2));
        input.values_mut()[ColIndex::new(0)] = 4.0;
        input.values_mut()[ColIndex::new(1)] = 7.0;
        basis.right_solve_for_tau(&input).unwrap();

        let mut unit_row = ScatteredRow::new(ColIndex::new(2));
        basis.left_solve_for_unit_row(0, &mut unit_row).unwrap();

        let tau = basis.tau.borrow();
        assert!(tau.computation_can_be_optimized);
        assert!(tau.value.non_zeros().is_empty());
        assert!(
            tau.value
                .values()
                .as_slice()
                .iter()
                .any(|&value| value != 0.0)
        );
    }

    #[test]
    fn viewed_basis_refactorizes_from_problem_columns_without_copying_them() {
        let mut problem = SparseMatrix::new();
        problem.populate_from_zero(RowIndex::new(3), ColIndex::new(4));
        for (column, values) in [
            [2.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            for (row, value) in values.into_iter().enumerate() {
                if value != 0.0 {
                    problem
                        .mutable_column(ColIndex::from_usize(column))
                        .add_entry(RowIndex::from_usize(row), value);
                }
            }
        }
        let problem = Rc::new(problem);
        let columns =
            RowToColMapping::from_vec(vec![ColIndex::new(0), ColIndex::new(2), ColIndex::new(3)]);
        let parameters = GlopParameters::default();
        let mut viewed =
            BasisRepresentation::new_for_basis(Rc::clone(&problem), &columns, &parameters).unwrap();
        assert_eq!(Rc::strong_count(&problem), 2);

        viewed.update_and_refactorize(1, ColIndex::new(1)).unwrap();
        let expected = BasisRepresentation::new(
            matrix(&[&[2.0, 1.0, 1.0], &[1.0, 4.0, 0.0], &[0.0, 1.0, 2.0]]),
            parameters.lu_factorization_pivot_threshold,
            usize::try_from(parameters.basis_refactorization_period).unwrap(),
        )
        .unwrap();
        let rhs = [3.0, -2.0, 5.0];
        assert_eq!(viewed.solve(&rhs).unwrap(), expected.solve(&rhs).unwrap());
    }

    #[test]
    fn reinitializing_with_identity_basis_retains_factorization_clock() {
        let mut problem = SparseMatrix::new();
        problem.populate_from_zero(RowIndex::new(2), ColIndex::new(4));
        for (column, entries) in [
            &[(0, 2.0), (1, 1.0)][..],
            &[(0, 1.0), (1, 3.0)][..],
            &[(0, 1.0)][..],
            &[(1, 1.0)][..],
        ]
        .into_iter()
        .enumerate()
        {
            for &(row, coefficient) in entries {
                problem
                    .mutable_column(ColIndex::from_usize(column))
                    .add_entry(RowIndex::from_usize(row), coefficient);
            }
        }
        let problem = Rc::new(problem);
        let parameters = GlopParameters::default();
        let crash = RowToColMapping::from_vec(vec![ColIndex::new(0), ColIndex::new(1)]);
        let mut basis =
            BasisRepresentation::new_for_basis(Rc::clone(&problem), &crash, &parameters).unwrap();
        let last_factorization_time = basis.last_factorization_deterministic_time;
        let cumulative_time = basis.deterministic_time();
        assert!(last_factorization_time > 0.0);

        let identity = RowToColMapping::from_vec(vec![ColIndex::new(2), ColIndex::new(3)]);
        basis
            .reinitialize_for_basis(Rc::clone(&problem), &identity, &parameters)
            .unwrap();

        assert!(basis.is_identity_basis());
        assert_eq!(
            basis.last_factorization_deterministic_time.to_bits(),
            last_factorization_time.to_bits()
        );
        assert_eq!(
            basis.deterministic_time().to_bits(),
            cumulative_time.to_bits()
        );
    }

    #[test]
    fn update_solve_agrees_with_fresh_refactorization() {
        let initial = matrix(&[&[2.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let mut updated = BasisRepresentation::new(initial, 0.1, 10).unwrap();
        let mut entering = SparseColumn::new();
        entering.add_entry(RowIndex::new(0), 1.0);
        entering.add_entry(RowIndex::new(1), -1.0);
        entering.add_entry(RowIndex::new(2), 3.0);
        updated.replace_column(1, entering).unwrap();

        let fresh = BasisRepresentation::new(updated.basis().clone(), 0.1, 10).unwrap();
        let rhs = [1.0, 2.0, -1.0];
        let left = updated.solve(&rhs).unwrap();
        let right = fresh.solve(&rhs).unwrap();
        assert!(
            left.iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
        let left = updated.transpose_solve(&rhs).unwrap();
        let right = fresh.transpose_solve(&rhs).unwrap();
        assert!(
            left.iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-12)
        );
    }

    #[test]
    fn unit_row_partial_solve_pool_is_reused_until_refactorization() {
        let initial = matrix(&[&[2.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let mut basis = BasisRepresentation::new(initial, 0.1, 10).unwrap();
        let mut result = ScatteredRow::new(ColIndex::new(3));
        basis.left_solve_for_unit_row(1, &mut result).unwrap();
        assert_eq!(basis.left_storage.borrow().num_cols(), ColIndex::new(1));
        let expected = result.values().as_slice().to_vec();
        basis.left_solve_for_unit_row(1, &mut result).unwrap();
        assert_eq!(basis.left_storage.borrow().num_cols(), ColIndex::new(1));
        assert_eq!(result.values().as_slice(), expected);

        let mut entering = SparseColumn::new();
        entering.add_entry(RowIndex::new(0), 1.0);
        entering.add_entry(RowIndex::new(1), -1.0);
        entering.add_entry(RowIndex::new(2), 3.0);
        basis.replace_column(1, entering).unwrap();
        assert_eq!(basis.left_storage.borrow().num_cols(), ColIndex::new(1));
        basis.refactorize().unwrap();
        assert_eq!(basis.left_storage.borrow().num_cols(), ColIndex::new(0));
        assert!(basis.left_pool_mapping.borrow().is_empty());
    }

    #[test]
    fn unit_row_solve_uses_upstream_density_aware_workspace_clear() {
        let size = 40;
        let mut columns = Vec::with_capacity(size);
        for column in 0..size {
            let mut entries = vec![0.0; size];
            entries[column] = 1.0;
            columns.push(entries);
        }
        let column_refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let basis = BasisRepresentation::new(matrix(&column_refs), 0.1, 10).unwrap();
        let mut result = ScatteredRow::new(ColIndex::from_usize(size));

        // Two recorded positions reach GLOP's 5% dense-clear boundary. The
        // unrecorded value models dense workspace state outside that sparse
        // support and must therefore be cleared as well.
        result.set(ColIndex::new(0), 2.0);
        result.set(ColIndex::new(1), 3.0);
        result.values_mut()[ColIndex::new(10)] = 4.0;

        basis.left_solve_for_unit_row(5, &mut result).unwrap();

        for (column, &value) in result.values().as_slice().iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                if column == 5 { 1.0_f64 } else { 0.0_f64 }.to_bits()
            );
        }
        assert_eq!(result.non_zeros(), &[ColIndex::new(5)]);
    }

    #[test]
    fn problem_column_partial_solve_is_stored_and_consumed_by_update() {
        let initial = matrix(&[&[2.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let mut basis = BasisRepresentation::new(initial, 0.1, 10).unwrap();
        let mut entering = SparseColumn::new();
        entering.add_entry(RowIndex::new(0), 1.0);
        entering.add_entry(RowIndex::new(1), -1.0);
        entering.add_entry(RowIndex::new(2), 3.0);
        let mut direction = ScatteredColumn::new(lp_data::lp_types::RowIndex::new(3));
        basis
            .right_solve_for_problem_column(7, &entering, &mut direction)
            .unwrap();
        let mut unit_left_inverse = ScatteredRow::new(ColIndex::new(3));
        basis
            .left_solve_for_unit_row(1, &mut unit_left_inverse)
            .unwrap();
        assert_eq!(basis.right_storage.borrow().num_cols(), ColIndex::new(1));
        assert_eq!(basis.right_pool_mapping.borrow()[7], Some(ColIndex::new(0)));
        basis
            .replace_column_from_partial_solves(7, 1, entering)
            .unwrap();
        assert_eq!(basis.num_updates(), 1);
        assert_eq!(basis.right_storage.borrow().num_cols(), ColIndex::new(1));
        basis.refactorize().unwrap();
        assert_eq!(basis.right_storage.borrow().num_cols(), ColIndex::new(0));
        assert!(basis.right_pool_mapping.borrow().is_empty());
    }
}
