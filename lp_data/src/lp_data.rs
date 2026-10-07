//! Linear-program model representation.
//!
//! This is the solver-facing subset of upstream `ortools/lp_data/lp_data.*`.

use std::collections::HashMap;

use crate::lp_types::{
    ColIndex, ConstraintStatus, ConstraintStatusColumn, DenseColumn, DenseRow, EntryIndex,
    Fractional, INFINITY, ProblemStatus, RowIndex, TypedVec, VariableStatus, VariableStatusRow,
    VectorIndex,
};
use crate::sparse::SparseMatrix;
use crate::sparse_vector::SparseColumn;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModelVariableType {
    #[default]
    Continuous,
    Integer,
    ImpliedInteger,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinearProgram {
    name: String,
    matrix: SparseMatrix,
    constraint_lower_bounds: DenseColumn,
    constraint_upper_bounds: DenseColumn,
    objective_coefficients: DenseRow,
    variable_lower_bounds: DenseRow,
    variable_upper_bounds: DenseRow,
    variable_types: TypedVec<ColIndex, ModelVariableType>,
    variable_names: TypedVec<ColIndex, String>,
    constraint_names: TypedVec<RowIndex, String>,
    variable_ids: HashMap<String, ColIndex>,
    constraint_ids: HashMap<String, RowIndex>,
    objective_offset: Fractional,
    objective_scaling_factor: Fractional,
    maximize: bool,
}

impl LinearProgram {
    #[must_use]
    pub fn new() -> Self {
        Self {
            objective_scaling_factor: 1.0,
            ..Self::default()
        }
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_new_variable(&mut self) -> ColIndex {
        let column = self.matrix.append_empty_column();
        self.objective_coefficients.push(0.0);
        self.variable_lower_bounds.push(0.0);
        self.variable_upper_bounds.push(INFINITY);
        self.variable_types.push(ModelVariableType::Continuous);
        self.variable_names.push(String::new());
        column
    }

    pub fn create_new_constraint(&mut self) -> RowIndex {
        let row = self.constraint_lower_bounds.len();
        self.constraint_lower_bounds.push(0.0);
        self.constraint_upper_bounds.push(0.0);
        self.constraint_names.push(String::new());
        self.matrix.set_num_rows(self.constraint_lower_bounds.len());
        row
    }

    pub fn find_or_create_variable(&mut self, id: &str) -> ColIndex {
        if let Some(&column) = self.variable_ids.get(id) {
            return column;
        }
        let column = self.create_new_variable();
        id.clone_into(&mut self.variable_names[column]);
        self.variable_ids.insert(id.to_owned(), column);
        column
    }

    pub fn find_or_create_constraint(&mut self, id: &str) -> RowIndex {
        if let Some(&row) = self.constraint_ids.get(id) {
            return row;
        }
        let row = self.create_new_constraint();
        id.clone_into(&mut self.constraint_names[row]);
        self.constraint_ids.insert(id.to_owned(), row);
        row
    }

    pub fn set_variable_name(&mut self, column: ColIndex, name: impl Into<String>) {
        self.variable_names[column] = name.into();
    }

    pub fn set_constraint_name(&mut self, row: RowIndex, name: impl Into<String>) {
        self.constraint_names[row] = name.into();
    }

    pub fn set_variable_type(&mut self, column: ColIndex, variable_type: ModelVariableType) {
        self.variable_types[column] = variable_type;
    }

    pub fn set_variable_bounds(&mut self, column: ColIndex, lower: Fractional, upper: Fractional) {
        self.variable_lower_bounds[column] = lower;
        self.variable_upper_bounds[column] = upper;
    }

    pub fn set_constraint_bounds(&mut self, row: RowIndex, lower: Fractional, upper: Fractional) {
        while self.num_constraints() <= row {
            self.create_new_constraint();
        }
        self.constraint_lower_bounds[row] = lower;
        self.constraint_upper_bounds[row] = upper;
    }

    pub fn set_coefficient(&mut self, row: RowIndex, column: ColIndex, value: Fractional) {
        self.matrix
            .mutable_column(column)
            .set_coefficient(row, value);
    }

    pub fn set_objective_coefficient(&mut self, column: ColIndex, value: Fractional) {
        self.objective_coefficients[column] = value;
    }

    pub fn set_objective_offset(&mut self, offset: Fractional) {
        self.objective_offset = offset;
    }

    pub fn set_objective_scaling_factor(&mut self, factor: Fractional) {
        self.objective_scaling_factor = factor;
    }

    pub fn set_maximization_problem(&mut self, maximize: bool) {
        self.maximize = maximize;
    }

    pub fn clean_up(&mut self) {
        self.matrix.clean_up();
    }

    #[must_use]
    pub fn is_cleaned_up(&self) -> bool {
        self.matrix.is_cleaned_up()
    }

    #[must_use]
    pub fn num_variables(&self) -> ColIndex {
        self.matrix.num_cols()
    }

    #[must_use]
    pub fn num_constraints(&self) -> RowIndex {
        self.matrix.num_rows()
    }

    #[must_use]
    pub fn num_entries(&self) -> EntryIndex {
        self.matrix.num_entries()
    }

    #[must_use]
    pub fn matrix(&self) -> &SparseMatrix {
        &self.matrix
    }

    #[must_use]
    pub fn sparse_column(&self, column: ColIndex) -> &SparseColumn {
        self.matrix.column(column)
    }

    #[must_use]
    pub fn constraint_lower_bounds(&self) -> &DenseColumn {
        &self.constraint_lower_bounds
    }

    #[must_use]
    pub fn constraint_upper_bounds(&self) -> &DenseColumn {
        &self.constraint_upper_bounds
    }

    #[must_use]
    pub fn objective_coefficients(&self) -> &DenseRow {
        &self.objective_coefficients
    }

    #[must_use]
    pub fn variable_lower_bounds(&self) -> &DenseRow {
        &self.variable_lower_bounds
    }

    #[must_use]
    pub fn variable_upper_bounds(&self) -> &DenseRow {
        &self.variable_upper_bounds
    }

    #[must_use]
    pub fn variable_types(&self) -> &TypedVec<ColIndex, ModelVariableType> {
        &self.variable_types
    }

    #[must_use]
    pub fn variable_name(&self, column: ColIndex) -> String {
        let name = &self.variable_names[column];
        if name.is_empty() {
            format!("C{}", column.value())
        } else {
            name.clone()
        }
    }

    #[must_use]
    pub fn constraint_name(&self, row: RowIndex) -> String {
        let name = &self.constraint_names[row];
        if name.is_empty() {
            format!("R{}", row.value())
        } else {
            name.clone()
        }
    }

    #[must_use]
    pub const fn objective_offset(&self) -> Fractional {
        self.objective_offset
    }

    #[must_use]
    pub const fn objective_scaling_factor(&self) -> Fractional {
        self.objective_scaling_factor
    }

    #[must_use]
    pub const fn is_maximization_problem(&self) -> bool {
        self.maximize
    }

    /// Checks dimensions, bounds, coefficients, and objective metadata.
    ///
    /// # Errors
    ///
    /// Returns a description of the first inconsistent or invalid value.
    pub fn validate(&self) -> Result<(), String> {
        let columns = self.num_variables().to_usize();
        let rows = self.num_constraints().to_usize();
        for (name, len) in [
            (
                "objective coefficients",
                self.objective_coefficients.as_slice().len(),
            ),
            (
                "variable lower bounds",
                self.variable_lower_bounds.as_slice().len(),
            ),
            (
                "variable upper bounds",
                self.variable_upper_bounds.as_slice().len(),
            ),
            ("variable types", self.variable_types.as_slice().len()),
            ("variable names", self.variable_names.as_slice().len()),
        ] {
            if len != columns {
                return Err(format!(
                    "{name} length {len} differs from {columns} columns"
                ));
            }
        }
        for (name, len) in [
            (
                "constraint lower bounds",
                self.constraint_lower_bounds.as_slice().len(),
            ),
            (
                "constraint upper bounds",
                self.constraint_upper_bounds.as_slice().len(),
            ),
            ("constraint names", self.constraint_names.as_slice().len()),
        ] {
            if len != rows {
                return Err(format!("{name} length {len} differs from {rows} rows"));
            }
        }
        for column in 0..columns {
            let column = ColIndex::from_usize(column);
            validate_bounds(
                self.variable_lower_bounds[column],
                self.variable_upper_bounds[column],
            )?;
            for entry in self.sparse_column(column) {
                if entry.index().to_usize() >= rows || !entry.coefficient().is_finite() {
                    return Err(format!("invalid matrix entry in column {column}"));
                }
            }
        }
        for row in 0..rows {
            let row = RowIndex::from_usize(row);
            validate_bounds(
                self.constraint_lower_bounds[row],
                self.constraint_upper_bounds[row],
            )?;
        }
        if !self.objective_offset.is_finite()
            || !self.objective_scaling_factor.is_finite()
            || self.objective_scaling_factor <= 0.0
            || self
                .objective_coefficients
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("invalid objective data".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            name: self.name.clone(),
            rows: self.num_constraints().to_usize(),
            columns: self.num_variables().to_usize(),
            nonzeros: self.num_entries().value().try_into().unwrap_or(usize::MAX),
            maximize: self.maximize,
            objective_offset: self.objective_offset,
        }
    }

    /// Returns a deterministic fingerprint of bounds and objective metadata.
    /// This is used to compare parser output with the pinned native GLOP model.
    #[must_use]
    pub fn data_fingerprint(&self) -> u64 {
        let mut hash = 14_695_981_039_346_656_037_u64;
        let mut add_byte = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        };
        add_byte(u8::from(self.maximize));
        for value in std::iter::once(self.objective_offset)
            .chain(std::iter::once(self.objective_scaling_factor))
            .chain(self.objective_coefficients.iter().copied())
            .chain(self.variable_lower_bounds.iter().copied())
            .chain(self.variable_upper_bounds.iter().copied())
            .chain(self.constraint_lower_bounds.iter().copied())
            .chain(self.constraint_upper_bounds.iter().copied())
        {
            for byte in value.to_bits().to_le_bytes() {
                add_byte(byte);
            }
        }
        hash
    }
}

fn validate_bounds(lower: Fractional, upper: Fractional) -> Result<(), String> {
    if lower.is_nan() || upper.is_nan() || lower == INFINITY || upper == -INFINITY || lower > upper
    {
        Err(format!("invalid bounds [{lower}, {upper}]"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelSummary {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub nonzeros: usize,
    pub maximize: bool,
    pub objective_offset: Fractional,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {} rows, {} columns, {} nonzeros, {}, objective offset {:.17e}",
            self.name,
            self.rows,
            self.columns,
            self.nonzeros,
            if self.maximize {
                "maximize"
            } else {
                "minimize"
            },
            self.objective_offset
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProblemSolution {
    pub status: ProblemStatus,
    pub primal_values: DenseRow,
    pub dual_values: DenseColumn,
    pub variable_statuses: VariableStatusRow,
    pub constraint_statuses: ConstraintStatusColumn,
}

impl ProblemSolution {
    #[must_use]
    pub fn new(num_rows: RowIndex, num_columns: ColIndex) -> Self {
        Self {
            status: ProblemStatus::Optimal,
            primal_values: DenseRow::filled(num_columns, 0.0),
            dual_values: DenseColumn::filled(num_rows, 0.0),
            variable_statuses: VariableStatusRow::filled(num_columns, VariableStatus::Free),
            constraint_statuses: ConstraintStatusColumn::filled(num_rows, ConstraintStatus::Free),
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn model_defaults_validation_and_solution_statuses_match_upstream() {
        let mut model = LinearProgram::new();
        let column = model.find_or_create_variable("x");
        let row = model.find_or_create_constraint("r");
        model.set_coefficient(row, column, 2.0);
        model.clean_up();
        assert!(model.validate().is_ok());
        assert_eq!(model.variable_lower_bounds()[column], 0.0);
        assert_eq!(model.variable_upper_bounds()[column], INFINITY);

        let solution = ProblemSolution::new(RowIndex::new(1), ColIndex::new(1));
        assert_eq!(solution.variable_statuses[column], VariableStatus::Free);
        assert_eq!(solution.constraint_statuses[row], ConstraintStatus::Free);
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        let mut model = LinearProgram::new();
        let column = model.create_new_variable();
        model.set_variable_bounds(column, 2.0, 1.0);
        assert!(model.validate().is_err());
    }
}
