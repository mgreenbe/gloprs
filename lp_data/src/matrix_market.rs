//! Reader for real, general, coordinate Matrix Market sparse matrices.
//!
//! Entries retain their order within each column, including explicit zeros.
//! This matters for deterministic Markowitz tie traversal in factorization
//! probes. Parsing and grouping happen outside benchmark timing.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::lp_types::{RowIndex, VectorIndex};
use crate::sparse::CompactSparseMatrix;

#[derive(Debug)]
pub struct MatrixMarketError(String);

impl fmt::Display for MatrixMarketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MatrixMarketError {}

fn error(line: usize, message: impl fmt::Display) -> MatrixMarketError {
    MatrixMarketError(format!("line {line}: {message}"))
}

/// Reads a Matrix Market file into GLOP's compact column representation.
///
/// # Errors
///
/// Rejects I/O failures, unsupported Matrix Market variants, malformed
/// dimensions or entries, duplicate coordinates, and nonfinite coefficients.
pub fn read_matrix_market_file(
    path: impl AsRef<Path>,
) -> Result<CompactSparseMatrix, MatrixMarketError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|cause| MatrixMarketError(format!("{}: {cause}", path.display())))?;
    parse_matrix_market(&contents)
        .map_err(|cause| MatrixMarketError(format!("{}: {cause}", path.display())))
}

/// Parses a real, general, coordinate Matrix Market matrix.
///
/// # Errors
///
/// Rejects malformed or unsupported input, including duplicate coordinates.
pub fn parse_matrix_market(contents: &str) -> Result<CompactSparseMatrix, MatrixMarketError> {
    let mut lines = contents.lines().enumerate();
    let (header_line, header) = lines.next().ok_or_else(|| error(1, "missing header"))?;
    if header.trim_end() != "%%MatrixMarket matrix coordinate real general" {
        return Err(error(
            header_line + 1,
            "expected real general coordinate matrix",
        ));
    }
    let (shape_line, shape) = lines
        .by_ref()
        .find(|(_, line)| !line.trim().is_empty() && !line.starts_with('%'))
        .ok_or_else(|| error(2, "missing dimensions"))?;
    let dimensions: Vec<_> = shape.split_whitespace().collect();
    if dimensions.len() != 3 {
        return Err(error(shape_line + 1, "expected rows columns entries"));
    }
    let parse_size = |field: &str| {
        field
            .parse::<usize>()
            .map_err(|_| error(shape_line + 1, "invalid dimension or entry count"))
    };
    let rows = parse_size(dimensions[0])?;
    let columns = parse_size(dimensions[1])?;
    let entries = parse_size(dimensions[2])?;
    if rows == 0 || columns == 0 || rows > i32::MAX as usize || columns > i32::MAX as usize {
        return Err(error(shape_line + 1, "matrix dimensions are out of range"));
    }
    let mut grouped = vec![Vec::<(usize, f64)>::new(); columns];
    let mut seen = vec![HashSet::<usize>::new(); columns];
    let mut count = 0;
    for (line_number, line) in lines {
        if line.trim().is_empty() || line.starts_with('%') {
            continue;
        }
        if count == entries {
            return Err(error(line_number + 1, "more entries than declared"));
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(error(line_number + 1, "expected row column coefficient"));
        }
        let row = fields[0]
            .parse::<usize>()
            .map_err(|_| error(line_number + 1, "invalid row"))?;
        let column = fields[1]
            .parse::<usize>()
            .map_err(|_| error(line_number + 1, "invalid column"))?;
        let value = fields[2]
            .parse::<f64>()
            .map_err(|_| error(line_number + 1, "invalid coefficient"))?;
        if row == 0 || row > rows || column == 0 || column > columns || !value.is_finite() {
            return Err(error(line_number + 1, "entry is out of range or nonfinite"));
        }
        if !seen[column - 1].insert(row - 1) {
            return Err(error(line_number + 1, "duplicate coordinate"));
        }
        grouped[column - 1].push((row - 1, value));
        count += 1;
    }
    if count != entries {
        return Err(error(
            contents.lines().count(),
            "fewer entries than declared",
        ));
    }
    let mut matrix = CompactSparseMatrix::default();
    matrix.reset(RowIndex::from_usize(rows));
    for column in grouped {
        for (row, value) in column {
            matrix.add_entry_to_current_column(RowIndex::from_usize(row), value);
        }
        matrix.close_current_column();
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::parse_matrix_market;
    use crate::lp_types::{ColIndex, RowIndex};

    #[test]
    fn groups_columns_without_reordering_entries() {
        let input = "%%MatrixMarket matrix coordinate real general\n% comment\n3 2 4\n3 2 5\n2 1 -0.0\n1 2 2\n3 1 7\n";
        let matrix = parse_matrix_market(input).unwrap();
        let first: Vec<_> = matrix.column(ColIndex::new(0)).iter().collect();
        let second: Vec<_> = matrix.column(ColIndex::new(1)).iter().collect();
        assert_eq!(first[0].0, RowIndex::new(1));
        assert!(first[0].1.is_sign_negative());
        assert_eq!(first[1], (RowIndex::new(2), 7.0));
        assert_eq!(second, [(RowIndex::new(2), 5.0), (RowIndex::new(0), 2.0)]);
    }

    #[test]
    fn rejects_malformed_and_duplicate_entries() {
        for input in [
            "%%MatrixMarket matrix coordinate integer general\n1 1 0\n",
            "%%MatrixMarket matrix coordinate real general\n1 1 2\n1 1 1\n1 1 2\n",
            "%%MatrixMarket matrix coordinate real general\n1 1 1\n2 1 1\n",
            "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 NaN\n",
            "%%MatrixMarket matrix coordinate real general\n1 1 1\n",
        ] {
            assert!(parse_matrix_market(input).is_err(), "{input}");
        }
    }
}
