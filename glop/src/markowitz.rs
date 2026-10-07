//! Markowitz pivot selection for sparse Gaussian elimination.
//!
//! This preserves the defining upstream choice: among numerically acceptable
//! active entries, minimize `(row_count - 1) * (column_count - 1)`, with stable
//! row/column tie breaking.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pivot {
    pub row: usize,
    pub column: usize,
    pub markowitz: usize,
}

#[must_use]
pub fn choose_pivot(matrix: &[Vec<f64>], start: usize, threshold: f64) -> Option<Pivot> {
    let n = matrix.len();
    if start >= n {
        return None;
    }
    let mut row_counts = vec![0_usize; n];
    let mut column_counts = vec![0_usize; n];
    let mut column_maxima = vec![0.0_f64; n];
    for row in start..n {
        for column in start..n {
            let magnitude = matrix[row][column].abs();
            if magnitude != 0.0 {
                row_counts[row] += 1;
                column_counts[column] += 1;
                column_maxima[column] = column_maxima[column].max(magnitude);
            }
        }
    }

    let mut best: Option<(usize, f64, usize, usize)> = None;
    for column in start..n {
        let minimum_magnitude = threshold * column_maxima[column];
        for row in start..n {
            let magnitude = matrix[row][column].abs();
            if magnitude == 0.0 || magnitude < minimum_magnitude {
                continue;
            }
            let count = (row_counts[row] - 1).saturating_mul(column_counts[column] - 1);
            let candidate = (count, -magnitude, row, column);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }
    }
    best.map(|(markowitz, _, row, column)| Pivot {
        row,
        column,
        markowitz,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooses_low_fill_acceptable_entry_deterministically() {
        let matrix = vec![
            vec![10.0, 0.0, 1.0],
            vec![1.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0],
        ];
        assert_eq!(
            choose_pivot(&matrix, 0, 0.1),
            Some(Pivot {
                row: 0,
                column: 0,
                markowitz: 1
            })
        );
    }
}
