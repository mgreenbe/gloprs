//! Matrix predicates and transformations from GLOP's `matrix_utils.{h,cc}`.

#![allow(clippy::float_cmp)] // These APIs promise exact equality with GLOP.

use crate::lp_types::{ColIndex, ColMapping, Fractional, INVALID_COL, RowIndex, VectorIndex};
use crate::sparse::{CompactSparseMatrix, SparseMatrix};
use crate::sparse_vector::SparseColumn;

fn mix(mut a: u64, mut b: u64, mut c: u64) -> u64 {
    a = a.wrapping_sub(b).wrapping_sub(c);
    a ^= c >> 43;
    b = b.wrapping_sub(c).wrapping_sub(a);
    b ^= a << 9;
    c = c.wrapping_sub(a).wrapping_sub(b);
    c ^= b >> 8;
    a = a.wrapping_sub(b).wrapping_sub(c);
    a ^= c >> 38;
    b = b.wrapping_sub(c).wrapping_sub(a);
    b ^= a << 23;
    c = c.wrapping_sub(a).wrapping_sub(b);
    c ^= b >> 5;
    a = a.wrapping_sub(b).wrapping_sub(c);
    a ^= c >> 35;
    b = b.wrapping_sub(c).wrapping_sub(a);
    b ^= a << 49;
    c = c.wrapping_sub(a).wrapping_sub(b);
    c ^= b >> 11;
    a = a.wrapping_sub(b).wrapping_sub(c);
    a ^= c >> 12;
    b = b.wrapping_sub(c).wrapping_sub(a);
    b ^= a << 18;
    c = c.wrapping_sub(a).wrapping_sub(b);
    c ^ (b >> 22)
}

fn hash(number: u64, seed: u64) -> u64 {
    mix(number, 0xe08c_1d66_8b75_6f82, seed)
}

#[derive(Clone, Copy)]
struct ColumnFingerprint {
    column: ColIndex,
    hash: i64,
    value: f64,
}

fn fingerprint(column: ColIndex, vector: &SparseColumn) -> ColumnFingerprint {
    let mut pattern_hash = 0_u64;
    let mut minimum = Fractional::MAX;
    let mut maximum: Fractional = 0.0;
    let mut sum = 0.0;
    for entry in vector {
        pattern_hash = hash(
            u64::try_from(entry.index().value()).expect("sparse row must be nonnegative"),
            pattern_hash,
        );
        sum += entry.coefficient();
        minimum = minimum.min(entry.coefficient().abs());
        maximum = maximum.max(entry.coefficient().abs());
    }
    let inverse_dynamic_range = minimum / maximum;
    #[allow(clippy::cast_precision_loss)]
    let scaled_average = sum.abs() / (vector.num_entries() as f64 * maximum);
    ColumnFingerprint {
        column,
        hash: pattern_hash.cast_signed(),
        value: inverse_dynamic_range + scaled_average,
    }
}

fn are_columns_proportional(
    left: &SparseColumn,
    right: &SparseColumn,
    tolerance: Fractional,
) -> bool {
    debug_assert!(left.is_cleaned_up());
    debug_assert!(right.is_cleaned_up());
    if left.num_entries() != right.num_entries() {
        return false;
    }
    let mut multiple = 0.0;
    let mut left_is_larger = true;
    for (left_entry, right_entry) in left.iter().zip(right.iter()) {
        if left_entry.index() != right_entry.index() {
            return false;
        }
        let left_value = left_entry.coefficient();
        let right_value = right_entry.coefficient();
        if multiple == 0.0 {
            left_is_larger = left_value.abs() > right_value.abs();
            multiple = if left_is_larger {
                left_value / right_value
            } else {
                right_value / left_value
            };
        } else {
            let current = if left_is_larger {
                left_value / right_value
            } else {
                right_value / left_value
            };
            if (current - multiple).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Finds proportional columns using GLOP's hashed-pattern fingerprint pass.
#[must_use]
pub fn find_proportional_columns(matrix: &SparseMatrix, tolerance: Fractional) -> ColMapping {
    let mut mapping = ColMapping::filled(matrix.num_cols(), INVALID_COL);
    let mut fingerprints = Vec::new();
    for position in 0..matrix.num_cols().to_usize() {
        let column = ColIndex::from_usize(position);
        if !matrix.column(column).is_empty() {
            fingerprints.push(fingerprint(column, matrix.column(column)));
        }
    }
    fingerprints.sort_unstable_by(|left, right| {
        left.hash.cmp(&right.hash).then_with(|| {
            left.value
                .partial_cmp(&right.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    for left_position in 0..fingerprints.len() {
        let left = fingerprints[left_position].column;
        if mapping[left] != INVALID_COL {
            continue;
        }
        for right_fingerprint in &fingerprints[left_position + 1..] {
            let right = right_fingerprint.column;
            if mapping[right] != INVALID_COL {
                continue;
            }
            if fingerprints[left_position].hash != right_fingerprint.hash
                || (fingerprints[left_position].value - right_fingerprint.value).abs() >= tolerance
            {
                break;
            }
            if are_columns_proportional(matrix.column(left), matrix.column(right), tolerance) {
                mapping[right] = left;
            }
        }
    }
    for position in 0..matrix.num_cols().to_usize() {
        let column = ColIndex::from_usize(position);
        if mapping[column] == INVALID_COL {
            continue;
        }
        let representative = mapping[mapping[column]];
        if representative != INVALID_COL {
            mapping[column] = representative;
        } else if mapping[column] > column {
            let old_representative = mapping[column];
            mapping[old_representative] = column;
            mapping[column] = INVALID_COL;
        }
    }
    mapping
}

/// Reference `O(num_cols * num_entries)` proportional-column implementation.
#[must_use]
pub fn find_proportional_columns_using_simple_algorithm(
    matrix: &SparseMatrix,
    tolerance: Fractional,
) -> ColMapping {
    let mut mapping = ColMapping::filled(matrix.num_cols(), INVALID_COL);
    for left_position in 0..matrix.num_cols().to_usize() {
        let left = ColIndex::from_usize(left_position);
        if matrix.column(left).is_empty() || mapping[left] != INVALID_COL {
            continue;
        }
        for right_position in left_position + 1..matrix.num_cols().to_usize() {
            let right = ColIndex::from_usize(right_position);
            if matrix.column(right).is_empty() || mapping[right] != INVALID_COL {
                continue;
            }
            if are_columns_proportional(matrix.column(left), matrix.column(right), tolerance) {
                mapping[right] = left;
            }
        }
    }
    mapping
}

/// Tests exact equality in the leading row/column rectangle, preserving the
/// pinned GLOP traversal and comparison order.
#[must_use]
pub fn are_first_columns_and_rows_exactly_equal(
    num_rows: RowIndex,
    num_cols: ColIndex,
    left: &SparseMatrix,
    right: &CompactSparseMatrix,
) -> bool {
    debug_assert!(left.is_cleaned_up());
    if num_rows > left.num_rows()
        || num_rows > right.num_rows()
        || num_cols > left.num_cols()
        || num_cols > right.num_cols()
    {
        return false;
    }
    for position in 0..num_cols.to_usize() {
        let column = ColIndex::from_usize(position);
        let left_column = left.column(column);
        let right_column = right.column(column);
        let end = left_column.num_entries().min(right_column.len());
        if end < left_column.num_entries() && left_column.entry(end).index() < num_rows {
            return false;
        }
        if end < right_column.len() && right_column.entry_row(end) < num_rows {
            return false;
        }
        for entry in 0..end {
            let left_row = left_column.entry(entry).index();
            let right_row = right_column.entry_row(entry);
            if left_row != right_row {
                if left_row < num_rows || right_row < num_rows {
                    return false;
                }
                break;
            }
            if left_column.entry(entry).coefficient() != right_column.entry_coefficient(entry) {
                return false;
            }
            if left_column.num_entries() > end && left_column.entry(end).index() < num_rows {
                return false;
            }
            if right_column.len() > end && right_column.entry_row(end) < num_rows {
                return false;
            }
        }
    }
    true
}

/// Returns whether GLOP recognizes the rightmost square block as identity.
#[must_use]
pub fn is_rightmost_square_matrix_identity(matrix: &SparseMatrix) -> bool {
    debug_assert!(matrix.is_cleaned_up());
    if matrix.num_rows().value() > matrix.num_cols().value() {
        return false;
    }
    let first = matrix.num_cols().value() - matrix.num_rows().value();
    for position in first..matrix.num_cols().value() {
        let column = matrix.column(ColIndex::new(position));
        if column.num_entries() != 1 || column.entry(0).coefficient() != 1.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_columns_and_matrix_predicates_follow_upstream() {
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(2), ColIndex::new(4));
        for (column, values) in [
            (0, [(0, 1.0), (1, 2.0)]),
            (1, [(0, -2.0), (1, -4.0)]),
            (2, [(0, 1.0), (1, 0.0)]),
            (3, [(0, 0.0), (1, 1.0)]),
        ] {
            for (row, value) in values {
                matrix
                    .mutable_column(ColIndex::new(column))
                    .set_coefficient(RowIndex::new(row), value);
            }
        }
        matrix.clean_up();
        let mapping = find_proportional_columns_using_simple_algorithm(&matrix, 1e-12);
        assert_eq!(mapping[ColIndex::new(1)], ColIndex::new(0));
        assert_eq!(mapping[ColIndex::new(0)], INVALID_COL);
        assert_eq!(find_proportional_columns(&matrix, 1e-12), mapping);
        assert!(is_rightmost_square_matrix_identity(&matrix));
        let compact = CompactSparseMatrix::from_sparse(&matrix);
        assert!(are_first_columns_and_rows_exactly_equal(
            RowIndex::new(2),
            ColIndex::new(4),
            &matrix,
            &compact
        ));
    }
}
