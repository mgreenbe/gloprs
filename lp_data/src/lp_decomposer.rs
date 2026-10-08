//! Independent-component decomposition of a linear program.
//!
//! This directly follows pinned `ortools/lp_data/lp_decomposer.cc`. Rust's
//! shared borrow replaces the C++ raw problem pointer and mutex: it prevents
//! mutation of the source model for the decomposer's lifetime.

use crate::lp_data::LinearProgram;
use crate::lp_types::{ColIndex, DenseRow, VectorIndex};

#[derive(Debug, Default)]
pub struct LpDecomposer<'a> {
    original_problem: Option<&'a LinearProgram>,
    clusters: Vec<Vec<ColIndex>>,
}

impl<'a> LpDecomposer<'a> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            original_problem: None,
            clusters: Vec::new(),
        }
    }

    /// Decomposes variables into connected components of the constraint graph.
    pub fn decompose(&mut self, problem: &'a LinearProgram) {
        self.original_problem = Some(problem);
        self.clusters.clear();
        let n = problem.num_variables().to_usize();
        let transpose = problem.transpose_sparse_matrix();
        let mut partition = MergingPartition::new(n);
        for constraint in 0..problem.num_constraints().to_usize() {
            let column = transpose.column(ColIndex::from_usize(constraint));
            if column.num_entries() > 1 {
                let first = column.first_row().to_usize();
                for entry in column.iter().skip(1) {
                    partition.merge(first, entry.index().to_usize());
                }
            }
        }
        let classes = partition.equivalence_classes();
        let number = classes.iter().copied().max().map_or(0, |class| class + 1);
        self.clusters.resize_with(number, Vec::new);
        for (column, class) in classes.into_iter().enumerate() {
            self.clusters[class].push(ColIndex::from_usize(column));
        }
        for cluster in &mut self.clusters {
            cluster.sort_unstable();
        }
    }

    #[must_use]
    pub fn number_of_problems(&self) -> usize {
        self.clusters.len()
    }

    /// # Panics
    /// Panics if `decompose()` has not been called.
    #[must_use]
    pub fn original_problem(&self) -> &'a LinearProgram {
        self.original_problem
            .expect("decompose must be called first")
    }

    /// Extracts one component in GLOP's global-column/global-row order.
    ///
    /// # Panics
    /// Panics if the problem index is invalid or decomposition has not run.
    #[must_use]
    pub fn extract_local_problem(&self, problem_index: usize) -> LinearProgram {
        let original = self.original_problem();
        let cluster = &self.clusters[problem_index];
        let mut local = LinearProgram::new();
        local.set_maximization_problem(original.is_maximization_problem());
        let mut global_to_local = vec![None; original.num_variables().to_usize()];
        let mut constraints_to_use = vec![false; original.num_constraints().to_usize()];
        let mut touched_constraints = Vec::new();
        for &global_column in cluster {
            let local_column = local.create_new_variable();
            global_to_local[global_column.to_usize()] = Some(local_column);
            local.set_variable_name(local_column, original.variable_name(global_column));
            local.set_variable_type(local_column, original.variable_type(global_column));
            local.set_variable_bounds(
                local_column,
                original.variable_lower_bounds()[global_column],
                original.variable_upper_bounds()[global_column],
            );
            local.set_objective_coefficient(
                local_column,
                original.objective_coefficients()[global_column],
            );
            for entry in original.matrix().column(global_column) {
                let row = entry.index().to_usize();
                if !constraints_to_use[row] {
                    constraints_to_use[row] = true;
                    touched_constraints.push(entry.index());
                }
            }
        }
        let transpose = original.transpose_sparse_matrix();
        for global_row in touched_constraints {
            let local_row = local.create_new_constraint();
            local.set_constraint_name(local_row, original.constraint_name(global_row));
            local.set_constraint_bounds(
                local_row,
                original.constraint_lower_bounds()[global_row],
                original.constraint_upper_bounds()[global_row],
            );
            for entry in transpose.column(ColIndex::new(global_row.value())) {
                let global_column = entry.index().to_usize();
                let local_column = global_to_local[global_column]
                    .expect("all variables in a selected constraint belong to the cluster");
                local.set_coefficient(local_row, local_column, entry.coefficient());
            }
        }
        local
    }

    /// # Panics
    /// Panics unless there is one correctly sized assignment per component.
    #[must_use]
    pub fn aggregate_assignments(&self, assignments: &[DenseRow]) -> DenseRow {
        assert_eq!(assignments.len(), self.clusters.len());
        let mut global = DenseRow::from_vec(vec![
            0.0;
            self.original_problem().num_variables().to_usize()
        ]);
        for (local, cluster) in assignments.iter().zip(&self.clusters) {
            for (position, &value) in local.as_slice().iter().enumerate() {
                global[cluster[position]] = value;
            }
        }
        global
    }

    /// # Panics
    /// Panics if the problem index or global assignment size is invalid.
    #[must_use]
    pub fn extract_local_assignment(
        &self,
        problem_index: usize,
        assignment: &DenseRow,
    ) -> DenseRow {
        assert_eq!(assignment.len(), self.original_problem().num_variables());
        DenseRow::from_vec(
            self.clusters[problem_index]
                .iter()
                .map(|&column| assignment[column])
                .collect(),
        )
    }
}

#[derive(Debug)]
struct MergingPartition {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl MergingPartition {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn root(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut current = node;
        while self.parent[current] != current {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }
    fn merge(&mut self, left: usize, right: usize) {
        let mut left_root = self.root(left);
        let mut right_root = self.root(right);
        if left_root == right_root {
            return;
        }
        if self.size[left_root] < self.size[right_root]
            || (self.size[left_root] == self.size[right_root] && left_root > right_root)
        {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.size[left_root] += self.size[right_root];
        self.parent[right_root] = left_root;
    }
    fn equivalence_classes(&mut self) -> Vec<usize> {
        let mut root_class = vec![None; self.parent.len()];
        let mut classes = Vec::with_capacity(self.parent.len());
        let mut count = 0;
        for node in 0..self.parent.len() {
            let root = self.root(node);
            let class = *root_class[root].get_or_insert_with(|| {
                let value = count;
                count += 1;
                value
            });
            classes.push(class);
        }
        classes
    }
}

#[cfg(test)]
mod tests {
    use super::LpDecomposer;
    use crate::lp_data::LinearProgram;
    use crate::lp_types::{ColIndex, DenseRow, RowIndex};

    #[test]
    fn connected_components_and_assignments_follow_global_order() {
        let mut lp = LinearProgram::new();
        for name in ["x", "y", "z", "t", "u"] {
            lp.find_or_create_variable(name);
        }
        for row in 0..3 {
            lp.create_new_constraint();
            lp.set_constraint_bounds(RowIndex::new(row), 0.0, 1.0);
        }
        for (row, column) in [(0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 4)] {
            lp.set_coefficient(RowIndex::new(row), ColIndex::new(column), 1.0);
        }
        let mut decomposer = LpDecomposer::new();
        decomposer.decompose(&lp);
        assert_eq!(decomposer.number_of_problems(), 2);
        assert_eq!(
            decomposer.extract_local_problem(0).num_variables(),
            ColIndex::new(3)
        );
        let global = DenseRow::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let parts = [
            decomposer.extract_local_assignment(0, &global),
            decomposer.extract_local_assignment(1, &global),
        ];
        assert_eq!(decomposer.aggregate_assignments(&parts), global);
    }
}
