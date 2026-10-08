//! Fixed- and free-field MPS parsing.
//!
//! The section semantics follow upstream `mps_reader_template.{h,cc}`. The
//! scanner accepts both traditional fixed-field files and free-field files by
//! treating the fixed fields as whitespace-delimited tokens; MPS identifiers
//! cannot themselves contain whitespace.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::lp_data::{LinearProgram, ModelVariableType};
use crate::lp_types::{ColIndex, Fractional, INFINITY, RowIndex, VectorIndex};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Section {
    None,
    ObjSense,
    Rows,
    LazyRows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Indicators,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowKind {
    Less,
    Equal,
    Greater,
    Free,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpsError {
    line: usize,
    message: String,
}

impl MpsError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for MpsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for MpsError {}

/// Parses an MPS file into a canonical column-oriented model.
///
/// # Errors
///
/// Returns an error when the file cannot be read or contains invalid MPS data.
pub fn parse_mps_file(path: impl AsRef<Path>) -> Result<LinearProgram, MpsError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|error| MpsError::new(0, format!("cannot read {}: {error}", path.display())))?;
    parse_mps(&contents)
}

/// Parses an MPS file in a requested format and returns the format used.
///
/// # Errors
///
/// Returns an error when the file cannot be read or the requested parse (or
/// both auto-detection attempts) fails.
pub fn parse_mps_file_with_format(
    path: impl AsRef<Path>,
    format: MpsFormat,
) -> Result<(LinearProgram, MpsFormat), MpsError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|error| MpsError::new(0, format!("cannot read {}: {error}", path.display())))?;
    parse_mps_with_format(&contents, format)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MpsFormat {
    AutoDetect,
    Free,
    Fixed,
}

/// Parses fixed- or free-field MPS text.
///
/// # Errors
///
/// Returns an error containing the source line for malformed or unsupported
/// input.
pub fn parse_mps(contents: &str) -> Result<LinearProgram, MpsError> {
    parse_mps_with_format(contents, MpsFormat::AutoDetect).map(|(model, _)| model)
}

/// Parses MPS text in a requested format and returns the format used.
///
/// Auto-detection follows GLOP exactly: parse the complete input as fixed
/// format first, and retry the complete input as free format on any error.
///
/// # Errors
///
/// Returns an error containing the source line when the requested parse (or
/// both auto-detection attempts) fails.
pub fn parse_mps_with_format(
    contents: &str,
    format: MpsFormat,
) -> Result<(LinearProgram, MpsFormat), MpsError> {
    match format {
        MpsFormat::AutoDetect => parse_mps_as(contents, MpsFormat::Fixed)
            .map(|model| (model, MpsFormat::Fixed))
            .or_else(|_| {
                parse_mps_as(contents, MpsFormat::Free).map(|model| (model, MpsFormat::Free))
            }),
        MpsFormat::Free | MpsFormat::Fixed => {
            parse_mps_as(contents, format).map(|model| (model, format))
        }
    }
}

fn parse_mps_as(contents: &str, format: MpsFormat) -> Result<LinearProgram, MpsError> {
    let mut parser = Parser::new(format);
    for (position, raw_line) in contents.lines().enumerate() {
        parser.parse_line(position + 1, raw_line)?;
    }
    Ok(parser.finish())
}

struct Parser {
    lp: LinearProgram,
    section: Section,
    objective_name: Option<String>,
    row_kinds: HashMap<String, (RowIndex, RowKind)>,
    binary_by_default: Vec<bool>,
    ended: bool,
    integer_mode: bool,
    format: MpsFormat,
}

impl Parser {
    fn new(format: MpsFormat) -> Self {
        debug_assert_ne!(format, MpsFormat::AutoDetect);
        Self {
            lp: LinearProgram::new(),
            section: Section::None,
            objective_name: None,
            row_kinds: HashMap::new(),
            binary_by_default: Vec::new(),
            ended: false,
            integer_mode: false,
            format,
        }
    }

    fn parse_line(&mut self, line_number: usize, raw_line: &str) -> Result<(), MpsError> {
        let line = raw_line.trim_end();
        if self.format == MpsFormat::Fixed && line.contains('\t') {
            return Err(MpsError::new(line_number, "fixed format contains a tab"));
        }
        if line.is_empty() || line.starts_with('*') || self.ended {
            return Ok(());
        }
        if self.format == MpsFormat::Fixed && !is_fixed_format(line) {
            return Err(MpsError::new(line_number, "line is not in fixed format"));
        }
        let trimmed = line.trim_start();
        let whitespace_tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if self.format == MpsFormat::Free && whitespace_tokens.len() > 6 {
            return Err(MpsError::new(line_number, "found too many fields"));
        }
        if whitespace_tokens.is_empty() {
            return Ok(());
        }

        // Both pinned formats identify section cards by column one. Section
        // mnemonics are case-sensitive; indented lookalikes are data cards.
        if !line.starts_with(' ') {
            if self.format == MpsFormat::Fixed && whitespace_tokens[0] == "NAME" {
                let free_name = whitespace_tokens.get(1).copied().unwrap_or("");
                let fixed_name = line.get(14..line.len().min(22)).unwrap_or("").trim_end();
                if free_name != fixed_name {
                    return Err(MpsError::new(
                        line_number,
                        "fixed NAME differs between free and fixed fields",
                    ));
                }
            }
            return self.start_section(line_number, &whitespace_tokens);
        }

        let fixed_tokens = (self.format == MpsFormat::Fixed && self.section != Section::ObjSense)
            .then(|| fixed_fields(line, self.section))
            .flatten();
        let tokens: Vec<&str> =
            if self.format == MpsFormat::Fixed && self.section != Section::ObjSense {
                let fields = fixed_tokens
                    .as_ref()
                    .ok_or_else(|| MpsError::new(line_number, "invalid fixed-format data card"))?;
                fields.iter().map(String::as_str).collect()
            } else {
                whitespace_tokens
            };

        match self.section {
            Section::ObjSense => self.parse_objective_sense(line_number, &tokens),
            Section::Rows | Section::LazyRows => self.parse_row(line_number, &tokens),
            Section::Columns => self.parse_column(line_number, &tokens),
            Section::Rhs => self.parse_rhs(line_number, &tokens),
            Section::Ranges => self.parse_range(line_number, &tokens),
            Section::Bounds => self.parse_bound(line_number, &tokens),
            Section::Indicators => Err(MpsError::new(
                line_number,
                "indicator constraints are unsupported by LinearProgram",
            )),
            Section::None => Err(MpsError::new(line_number, "data before first section")),
        }
    }

    fn start_section(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        match tokens[0] {
            "NAME" => {
                if let Some(name) = tokens.get(1) {
                    self.lp.set_name(*name);
                }
                Ok(())
            }
            "OBJSENSE" => {
                self.section = Section::ObjSense;
                if tokens.len() > 1 {
                    let sense = tokens[1];
                    if sense.contains("MIN") {
                        self.lp.set_maximization_problem(false);
                    } else if sense.contains("MAX") {
                        self.lp.set_maximization_problem(true);
                    } else {
                        return Err(MpsError::new(line, "invalid inline objective sense"));
                    }
                }
                Ok(())
            }
            "ROWS" => {
                self.section = Section::Rows;
                Ok(())
            }
            "LAZYCONS" => {
                self.section = Section::LazyRows;
                Ok(())
            }
            "COLUMNS" => {
                self.section = Section::Columns;
                Ok(())
            }
            "RHS" => {
                self.section = Section::Rhs;
                Ok(())
            }
            "RANGES" => {
                self.section = Section::Ranges;
                Ok(())
            }
            "BOUNDS" => {
                self.section = Section::Bounds;
                Ok(())
            }
            "INDICATORS" => {
                self.section = Section::Indicators;
                Ok(())
            }
            "ENDATA" => {
                self.ended = true;
                Ok(())
            }
            section => Err(MpsError::new(
                line,
                format!("unsupported section {section}"),
            )),
        }
    }

    fn parse_objective_sense(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        let sense = tokens
            .first()
            .ok_or_else(|| MpsError::new(line, "missing objective sense"))?;
        match *sense {
            "MIN" => self.lp.set_maximization_problem(false),
            "MAX" => self.lp.set_maximization_problem(true),
            _ => {
                return Err(MpsError::new(
                    line,
                    format!("invalid objective sense {sense}"),
                ));
            }
        }
        Ok(())
    }

    fn parse_row(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        if tokens.len() < 2 {
            return Err(MpsError::new(line, "row requires a type and name"));
        }
        let name = tokens[1];
        let kind = match tokens[0] {
            "N" => {
                if self.objective_name.is_none() {
                    self.objective_name = Some(name.to_owned());
                    return Ok(());
                }
                RowKind::Free
            }
            "L" => RowKind::Less,
            "E" => RowKind::Equal,
            "G" => RowKind::Greater,
            kind => return Err(MpsError::new(line, format!("invalid row type {kind}"))),
        };
        if self.row_kinds.contains_key(name) {
            return Err(MpsError::new(line, format!("duplicate row {name}")));
        }
        let row = self.lp.find_or_create_constraint(name);
        let bounds = match kind {
            RowKind::Less => (-INFINITY, 0.0),
            RowKind::Equal => (0.0, 0.0),
            RowKind::Greater => (0.0, INFINITY),
            RowKind::Free => (-INFINITY, INFINITY),
        };
        self.lp.set_constraint_bounds(row, bounds.0, bounds.1);
        self.row_kinds.insert(name.to_owned(), (row, kind));
        Ok(())
    }

    fn parse_column(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        if tokens.len() >= 3 && tokens[1].trim_matches('\'') == "MARKER" {
            let marker = tokens[2..]
                .iter()
                .find(|token| !token.is_empty())
                .ok_or_else(|| MpsError::new(line, "missing marker value"))?;
            match marker.trim_matches('\'') {
                "INTORG" if self.integer_mode => {
                    return Err(MpsError::new(line, "INTORG inside integer section"));
                }
                "INTORG" => self.integer_mode = true,
                "INTEND" if !self.integer_mode => {
                    return Err(MpsError::new(line, "INTEND outside integer section"));
                }
                "INTEND" => self.integer_mode = false,
                marker => return Err(MpsError::new(line, format!("unknown marker {marker}"))),
            }
            return Ok(());
        }
        if tokens.len() < 3 || tokens.len().is_multiple_of(2) {
            return Err(MpsError::new(
                line,
                "column requires one or two row/value pairs",
            ));
        }
        let column_name = tokens[0].to_owned();
        let column = self.lp.find_or_create_variable(&column_name);
        if self.binary_by_default.len() <= column.to_usize() {
            self.binary_by_default.resize(column.to_usize() + 1, false);
        }
        if self.integer_mode {
            self.lp
                .set_variable_type(column, ModelVariableType::Integer);
            self.lp.set_variable_bounds(column, 0.0, 1.0);
            self.binary_by_default[column.to_usize()] = true;
        } else {
            // This assignment is performed for every ordinary COLUMNS card by
            // pinned GLOP, not just when the variable is first encountered.
            self.lp.set_variable_bounds(column, 0.0, INFINITY);
        }
        self.for_pairs(line, &tokens[1..], |parser, row_name, value| {
            parser.store_coefficient(line, column, row_name, value)
        })
    }

    fn store_coefficient(
        &mut self,
        line: usize,
        column: ColIndex,
        row_name: &str,
        value: &str,
    ) -> Result<(), MpsError> {
        let coefficient = parse_number(line, value)?;
        if !coefficient.is_finite() {
            return Err(MpsError::new(line, "matrix coefficient must be finite"));
        }
        if coefficient == 0.0 {
            return Ok(());
        }
        if Some(row_name) == self.objective_name.as_deref() {
            self.lp.set_objective_coefficient(column, coefficient);
        } else {
            let row = self.lp.find_or_create_constraint(row_name);
            self.lp.set_coefficient(row, column, coefficient);
        }
        Ok(())
    }

    fn parse_rhs(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        self.parse_named_pairs(line, tokens, NamedVector::Rhs)
    }

    fn parse_range(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        self.parse_named_pairs(line, tokens, NamedVector::Range)
    }

    fn parse_named_pairs(
        &mut self,
        line: usize,
        tokens: &[&str],
        vector: NamedVector,
    ) -> Result<(), MpsError> {
        if tokens.len() < 2 {
            return Err(MpsError::new(line, "named vector requires row/value pairs"));
        }
        let pairs = if tokens.len().is_multiple_of(2) {
            tokens
        } else {
            &tokens[1..]
        };
        // As in GLOP, the leading vector name is syntactically consumed but
        // otherwise ignored; cards with different names are all applied.
        self.for_pairs(line, pairs, |parser, row_name, value| match vector {
            NamedVector::Rhs => parser.store_rhs(line, row_name, value),
            NamedVector::Range => parser.store_range(line, row_name, value),
        })
    }

    fn store_rhs(&mut self, line: usize, row_name: &str, value: &str) -> Result<(), MpsError> {
        let value = parse_number(line, value)?;
        if Some(row_name) == self.objective_name.as_deref() {
            self.lp.set_objective_offset(-value);
            return Ok(());
        }
        let row = self.lp.find_or_create_constraint(row_name);
        let lower = self.lp.constraint_lower_bounds()[row];
        let upper = self.lp.constraint_upper_bounds()[row];
        self.lp.set_constraint_bounds(
            row,
            if lower == -INFINITY { -INFINITY } else { value },
            if upper == INFINITY { INFINITY } else { value },
        );
        Ok(())
    }

    #[allow(clippy::float_cmp)] // GLOP distinguishes exact row-bound encodings.
    fn store_range(&mut self, line: usize, row_name: &str, value: &str) -> Result<(), MpsError> {
        let value = parse_number(line, value)?;
        let row = self.lp.find_or_create_constraint(row_name);
        let mut lower = self.lp.constraint_lower_bounds()[row];
        let mut upper = self.lp.constraint_upper_bounds()[row];
        if lower == upper {
            if value < 0.0 {
                lower += value;
            } else {
                upper += value;
            }
        }
        if lower == -INFINITY {
            lower = upper - value.abs();
        }
        if upper == INFINITY {
            upper = lower + value.abs();
        }
        self.lp.set_constraint_bounds(row, lower, upper);
        Ok(())
    }

    fn parse_bound(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        if tokens.len() < 3 {
            return Err(MpsError::new(
                line,
                "bound requires type, vector, and column",
            ));
        }
        let column = self.lp.find_or_create_variable(tokens[2]);
        if self.binary_by_default.len() <= column.to_usize() {
            self.binary_by_default.resize(column.to_usize() + 1, false);
        }
        let mut lower = self.lp.variable_lower_bounds()[column];
        let mut upper = self.lp.variable_upper_bounds()[column];
        if self.binary_by_default[column.to_usize()] {
            lower = 0.0;
            upper = INFINITY;
        }
        let kind = tokens[0];
        let required_value = || {
            tokens
                .get(3)
                .ok_or_else(|| MpsError::new(line, format!("missing value for {kind} bound")))
                .and_then(|token| parse_number(line, token))
        };
        match kind {
            "LO" => self
                .lp
                .set_variable_bounds(column, required_value()?, upper),
            "UP" => self
                .lp
                .set_variable_bounds(column, lower, required_value()?),
            "FX" => {
                let value = required_value()?;
                self.lp.set_variable_bounds(column, value, value);
            }
            "FR" => self.lp.set_variable_bounds(column, -INFINITY, INFINITY),
            "MI" => self.lp.set_variable_bounds(column, -INFINITY, upper),
            "PL" => self.lp.set_variable_bounds(column, lower, INFINITY),
            "BV" => {
                self.lp
                    .set_variable_type(column, ModelVariableType::Integer);
                self.lp.set_variable_bounds(column, 0.0, 1.0);
            }
            "LI" => {
                let value = required_value()?;
                self.lp
                    .set_variable_type(column, ModelVariableType::Integer);
                if value == 0.0 {
                    upper = INFINITY;
                }
                self.lp.set_variable_bounds(column, value, upper);
            }
            "UI" => {
                let value = required_value()?;
                self.lp
                    .set_variable_type(column, ModelVariableType::Integer);
                self.lp.set_variable_bounds(column, lower, value);
            }
            kind => {
                return Err(MpsError::new(
                    line,
                    format!("unsupported bound type {kind}"),
                ));
            }
        }
        self.binary_by_default[column.to_usize()] = false;
        Ok(())
    }

    fn for_pairs(
        &mut self,
        line: usize,
        tokens: &[&str],
        mut consume: impl FnMut(&mut Self, &str, &str) -> Result<(), MpsError>,
    ) -> Result<(), MpsError> {
        for pair in tokens.chunks_exact(2) {
            if pair[0] == "$" {
                break;
            }
            consume(self, pair[0], pair[1])?;
        }
        if !tokens.len().is_multiple_of(2) {
            return Err(MpsError::new(line, "incomplete name/value pair"));
        }
        Ok(())
    }

    fn finish(mut self) -> LinearProgram {
        self.lp.clean_up();
        self.lp
    }
}

#[derive(Clone, Copy)]
enum NamedVector {
    Rhs,
    Range,
}

fn parse_number(line: usize, token: &str) -> Result<Fractional, MpsError> {
    let value: Fractional = token
        .parse()
        .map_err(|_| MpsError::new(line, format!("invalid number {token}")))?;
    if value.is_nan() {
        return Err(MpsError::new(line, "NaN value"));
    }
    Ok(value)
}

fn fixed_fields(line: &str, section: Section) -> Option<Vec<String>> {
    let bytes = line.as_bytes();
    let row_line = matches!(section, Section::Rows | Section::LazyRows);
    if bytes.len() < if row_line { 5 } else { 14 }
        || bytes.get(3).is_some_and(|byte| !byte.is_ascii_whitespace())
        || (!row_line
            && bytes
                .get(12..14)
                .is_some_and(|pair| pair.iter().any(|byte| !byte.is_ascii_whitespace())))
    {
        return None;
    }
    let ranges: &[(usize, usize)] = match section {
        Section::Rows | Section::LazyRows => &[(1, 3), (4, 12)],
        Section::Columns | Section::Rhs | Section::Ranges => {
            &[(4, 12), (14, 22), (24, 36), (39, 47), (49, 61)]
        }
        Section::Bounds => &[(1, 3), (4, 12), (14, 22), (24, 36)],
        _ => return None,
    };
    let mut fields: Vec<String> = ranges
        .iter()
        .map(|&(start, end)| {
            line.get(start..end.min(line.len()))
                .map_or_else(String::new, |field| field.trim().to_owned())
        })
        .collect();
    while fields.last().is_some_and(String::is_empty) {
        fields.pop();
    }
    if section == Section::Rows {
        fields.retain(|field| !field.is_empty());
    }
    (!fields.is_empty()).then_some(fields)
}

fn is_fixed_format(line: &str) -> bool {
    const REQUIRED_SPACES: [usize; 12] = [12, 13, 22, 23, 36, 37, 38, 47, 48, 61, 62, 63];
    if !line.starts_with(' ') {
        let first_word = line.split_once(' ').map_or(line, |(word, _)| word);
        return first_word == line || first_word == "NAME";
    }
    if line.len() > 61 {
        return false;
    }
    REQUIRED_SPACES
        .into_iter()
        .take_while(|&position| position < line.len())
        .all(|position| line.as_bytes()[position] == b' ')
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn parses_ranges_bounds_objective_offset_and_duplicate_entries() {
        let model = parse_mps(
            "NAME TEST\nOBJSENSE\n MAX\nROWS\n N OBJ\n L LIM\n E EQ\nCOLUMNS\n X LIM 1 OBJ 2\n X LIM 3 EQ -1\nRHS\n RHS LIM 5 EQ 7\n RHS OBJ 4\nRANGES\n RNG LIM 2 EQ -3\nBOUNDS\n FR BND X\nENDATA\n",
        )
        .unwrap();
        assert!(model.is_maximization_problem());
        assert_eq!(model.objective_offset(), -4.0);
        assert_eq!(model.num_entries(), crate::lp_types::EntryIndex::new(2));
        assert_eq!(
            model
                .matrix()
                .look_up_value(RowIndex::new(0), ColIndex::new(0)),
            3.0
        );
        assert_eq!(model.constraint_lower_bounds()[RowIndex::new(0)], 3.0);
        assert_eq!(model.constraint_upper_bounds()[RowIndex::new(0)], 5.0);
        assert_eq!(model.constraint_lower_bounds()[RowIndex::new(1)], 4.0);
        assert_eq!(model.constraint_upper_bounds()[RowIndex::new(1)], 7.0);
        assert_eq!(model.variable_lower_bounds()[ColIndex::new(0)], -INFINITY);
    }

    #[test]
    fn fixed_fields_preserve_embedded_spaces_and_blank_vector_names() {
        let model = parse_mps(
            "NAME          FIXED\nROWS\n N  COST\n E  ROW A  1\nCOLUMNS\n    X         ROW A  1           2.\nRHS\n              ROW A  1           3.\nBOUNDS\n UP           X                   4.\nENDATA\n",
        )
        .unwrap();
        assert_eq!(model.constraint_name(RowIndex::new(0)), "ROW A  1");
        assert_eq!(model.num_variables(), ColIndex::new(1));
        assert_eq!(model.constraint_lower_bounds()[RowIndex::new(0)], 3.0);
        assert_eq!(model.variable_upper_bounds()[ColIndex::new(0)], 4.0);
    }

    #[test]
    fn matches_glop_integer_markers_and_ignores_vector_names() {
        let model = parse_mps(
            "NAME TEST\nROWS\n E R\nCOLUMNS\n M 'MARKER' 'INTORG'\n X R 1\n M 'MARKER' 'INTEND'\nRHS\n RHS1 R 2\n RHS2 R 3\nBOUNDS\n LO BND1 X -4\n UP BND2 X 7\nENDATA\n",
        )
        .unwrap();
        let x = ColIndex::new(0);
        let r = RowIndex::new(0);
        assert_eq!(model.variable_types()[x], ModelVariableType::Integer);
        assert_eq!(model.variable_lower_bounds()[x], -4.0);
        assert_eq!(model.variable_upper_bounds()[x], 7.0);
        assert_eq!(model.constraint_lower_bounds()[r], 3.0);
        assert_eq!(model.constraint_upper_bounds()[r], 3.0);
    }

    #[test]
    fn accepts_no_objective_row_and_preserves_later_free_rows() {
        let feasibility =
            parse_mps("NAME FEAS\nROWS\n E R\nCOLUMNS\n X R 1\nRHS\n RHS R 2\nENDATA\n").unwrap();
        assert_eq!(feasibility.objective_coefficients()[ColIndex::new(0)], 0.0);

        let free_row = parse_mps(
            "NAME FREE\nROWS\n N OBJ\n N FREE_ROW\nCOLUMNS\n X FREE_ROW 2\nRHS\n RHS FREE_ROW 3\nENDATA\n",
        )
        .unwrap();
        let row = RowIndex::new(0);
        assert_eq!(free_row.constraint_lower_bounds()[row], -INFINITY);
        assert_eq!(free_row.constraint_upper_bounds()[row], INFINITY);
    }

    #[test]
    fn auto_detection_retries_whole_input_and_reports_the_format() {
        let free = "NAME FREE\nROWS\n N OBJ\nENDATA\n";
        assert!(parse_mps_with_format(free, MpsFormat::Fixed).is_err());
        assert_eq!(
            parse_mps_with_format(free, MpsFormat::AutoDetect)
                .unwrap()
                .1,
            MpsFormat::Free
        );

        let fixed = "NAME          FIXED\nROWS\n N  OBJ\nENDATA\n";
        assert_eq!(
            parse_mps_with_format(fixed, MpsFormat::AutoDetect)
                .unwrap()
                .1,
            MpsFormat::Fixed
        );
    }

    #[test]
    fn signed_zero_coefficients_are_not_stored() {
        let model =
            parse_mps("NAME ZERO\nROWS\n N OBJ\n E R\nCOLUMNS\n X OBJ -0 R -0\nRHS\n RHS R -0\n")
                .unwrap();
        assert_eq!(model.num_entries(), crate::lp_types::EntryIndex::new(0));
        assert_eq!(
            model.objective_coefficients()[ColIndex::new(0)].to_bits(),
            0
        );
        assert_eq!(
            model.constraint_lower_bounds()[RowIndex::new(0)].to_bits(),
            (-0.0_f64).to_bits()
        );
    }
}
