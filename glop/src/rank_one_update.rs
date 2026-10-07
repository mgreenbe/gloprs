//! Product-form inverse (eta) updates used between basis refactorizations.

#[derive(Clone, Debug)]
pub struct RankOneUpdate {
    leaving_row: usize,
    direction: Vec<f64>,
}

impl RankOneUpdate {
    /// Creates the eta matrix for replacing one basis column.
    ///
    /// # Errors
    ///
    /// Returns an error when the pivot is zero or indices are inconsistent.
    pub fn new(leaving_row: usize, direction: Vec<f64>) -> Result<Self, &'static str> {
        if leaving_row >= direction.len() {
            return Err("leaving row is out of range");
        }
        if direction[leaving_row] == 0.0 || !direction[leaving_row].is_finite() {
            return Err("rank-one update has an invalid pivot");
        }
        Ok(Self {
            leaving_row,
            direction,
        })
    }

    pub fn solve(&self, values: &mut [f64]) {
        let pivot_value = values[self.leaving_row] / self.direction[self.leaving_row];
        for (row, value) in values.iter_mut().enumerate() {
            if row != self.leaving_row {
                *value -= self.direction[row] * pivot_value;
            }
        }
        values[self.leaving_row] = pivot_value;
    }

    pub fn transpose_solve(&self, values: &mut [f64]) {
        let mut value = values[self.leaving_row];
        for (row, &direction) in self.direction.iter().enumerate() {
            if row != self.leaving_row {
                value -= direction * values[row];
            }
        }
        values[self.leaving_row] = value / self.direction[self.leaving_row];
    }
}
