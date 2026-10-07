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
use crate::lp_types::{ColIndex, Fractional, INFINITY, RowIndex};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Section {
    None,
    ObjSense,
    ObjName,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowKind {
    Less,
    Equal,
    Greater,
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

/// Parses fixed- or free-field MPS text.
///
/// # Errors
///
/// Returns an error containing the source line for malformed or unsupported
/// input.
pub fn parse_mps(contents: &str) -> Result<LinearProgram, MpsError> {
    let mut parser = Parser::new();
    for (position, raw_line) in contents.lines().enumerate() {
        parser.parse_line(position + 1, raw_line)?;
    }
    parser.finish(contents.lines().count())
}

struct Parser {
    lp: LinearProgram,
    section: Section,
    objective_name: Option<String>,
    row_kinds: HashMap<String, (RowIndex, RowKind)>,
    rhs_name: Option<String>,
    range_name: Option<String>,
    bound_name: Option<String>,
    ended: bool,
    integer_mode: bool,
    last_column_name: Option<String>,
}

impl Parser {
    fn new() -> Self {
        Self {
            lp: LinearProgram::new(),
            section: Section::None,
            objective_name: None,
            row_kinds: HashMap::new(),
            rhs_name: None,
            range_name: None,
            bound_name: None,
            ended: false,
            integer_mode: false,
            last_column_name: None,
        }
    }

    fn parse_line(&mut self, line_number: usize, raw_line: &str) -> Result<(), MpsError> {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') || self.ended {
            return Ok(());
        }
        let whitespace_tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if whitespace_tokens.is_empty() {
            return Ok(());
        }

        let keyword = whitespace_tokens[0].to_ascii_uppercase();
        let is_header = keyword == "NAME"
            || keyword == "OBJSENSE"
            || keyword == "OBJNAME"
            || (whitespace_tokens.len() == 1
                && matches!(
                    keyword.as_str(),
                    "ROWS" | "COLUMNS" | "RHS" | "RANGES" | "BOUNDS" | "ENDATA"
                ));
        if is_header {
            return self.start_section(line_number, &whitespace_tokens);
        }

        let fixed_tokens = fixed_fields(raw_line, self.section);
        let tokens: Vec<&str> = fixed_tokens.as_ref().map_or(whitespace_tokens, |fields| {
            fields.iter().map(String::as_str).collect()
        });

        match self.section {
            Section::ObjSense => self.parse_objective_sense(line_number, &tokens),
            Section::ObjName => self.parse_objective_name(line_number, &tokens),
            Section::Rows => self.parse_row(line_number, &tokens),
            Section::Columns => self.parse_column(line_number, &tokens),
            Section::Rhs => self.parse_rhs(line_number, &tokens),
            Section::Ranges => self.parse_range(line_number, &tokens),
            Section::Bounds => self.parse_bound(line_number, &tokens),
            Section::None => Err(MpsError::new(line_number, "data before first section")),
        }
    }

    fn start_section(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        match tokens[0].to_ascii_uppercase().as_str() {
            "NAME" => {
                if let Some(name) = tokens.get(1) {
                    self.lp.set_name(*name);
                }
                Ok(())
            }
            "OBJSENSE" => {
                self.section = Section::ObjSense;
                if tokens.len() > 1 {
                    self.parse_objective_sense(line, &tokens[1..])?;
                }
                Ok(())
            }
            "OBJNAME" => {
                self.section = Section::ObjName;
                if tokens.len() > 1 {
                    self.parse_objective_name(line, &tokens[1..])?;
                }
                Ok(())
            }
            "ROWS" => {
                self.section = Section::Rows;
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
        match sense.to_ascii_uppercase().as_str() {
            "MIN" | "MINIMIZE" | "MINIMUM" => self.lp.set_maximization_problem(false),
            "MAX" | "MAXIMIZE" | "MAXIMUM" => self.lp.set_maximization_problem(true),
            _ => {
                return Err(MpsError::new(
                    line,
                    format!("invalid objective sense {sense}"),
                ));
            }
        }
        Ok(())
    }

    fn parse_objective_name(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        let name = tokens
            .first()
            .ok_or_else(|| MpsError::new(line, "missing objective row name"))?;
        self.objective_name = Some((*name).to_owned());
        Ok(())
    }

    fn parse_row(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        if tokens.len() < 2 {
            return Err(MpsError::new(line, "row requires a type and name"));
        }
        let name = tokens[1];
        let kind = match tokens[0].to_ascii_uppercase().as_str() {
            "N" => {
                if self.objective_name.is_none() {
                    self.objective_name = Some(name.to_owned());
                }
                return Ok(());
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
                "INTORG" => self.integer_mode = true,
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
        let column_name = if tokens[0].is_empty() {
            self.last_column_name
                .clone()
                .ok_or_else(|| MpsError::new(line, "column continuation without a column"))?
        } else {
            self.last_column_name = Some(tokens[0].to_owned());
            tokens[0].to_owned()
        };
        let column = self.lp.find_or_create_variable(&column_name);
        if self.integer_mode {
            self.lp
                .set_variable_type(column, ModelVariableType::Integer);
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
        if Some(row_name) == self.objective_name.as_deref() {
            self.lp.set_objective_coefficient(column, coefficient);
        } else {
            let &(row, _) = self
                .row_kinds
                .get(row_name)
                .ok_or_else(|| MpsError::new(line, format!("unknown row {row_name}")))?;
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
        let (name, pairs) = if tokens.len().is_multiple_of(2) {
            ("", tokens)
        } else {
            (tokens[0], &tokens[1..])
        };
        let selected_name = match vector {
            NamedVector::Rhs => &mut self.rhs_name,
            NamedVector::Range => &mut self.range_name,
        };
        if selected_name.is_none() {
            *selected_name = Some(name.to_owned());
        }
        if selected_name.as_deref() != Some(name) {
            return Ok(());
        }
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
        let &(row, kind) = self
            .row_kinds
            .get(row_name)
            .ok_or_else(|| MpsError::new(line, format!("unknown row {row_name}")))?;
        match kind {
            RowKind::Less => self.lp.set_constraint_bounds(row, -INFINITY, value),
            RowKind::Equal => self.lp.set_constraint_bounds(row, value, value),
            RowKind::Greater => self.lp.set_constraint_bounds(row, value, INFINITY),
        }
        Ok(())
    }

    fn store_range(&mut self, line: usize, row_name: &str, value: &str) -> Result<(), MpsError> {
        let value = parse_number(line, value)?;
        let &(row, kind) = self
            .row_kinds
            .get(row_name)
            .ok_or_else(|| MpsError::new(line, format!("unknown row {row_name}")))?;
        let lower = self.lp.constraint_lower_bounds()[row];
        let upper = self.lp.constraint_upper_bounds()[row];
        match kind {
            RowKind::Less => self
                .lp
                .set_constraint_bounds(row, upper - value.abs(), upper),
            RowKind::Greater => self
                .lp
                .set_constraint_bounds(row, lower, lower + value.abs()),
            RowKind::Equal if value >= 0.0 => {
                self.lp.set_constraint_bounds(row, lower, lower + value);
            }
            RowKind::Equal => self.lp.set_constraint_bounds(row, lower + value, lower),
        }
        Ok(())
    }

    fn parse_bound(&mut self, line: usize, tokens: &[&str]) -> Result<(), MpsError> {
        if tokens.len() < 3 {
            return Err(MpsError::new(
                line,
                "bound requires type, vector, and column",
            ));
        }
        if self.bound_name.is_none() {
            self.bound_name = Some(tokens[1].to_owned());
        }
        if self.bound_name.as_deref() != Some(tokens[1]) {
            return Ok(());
        }
        let column = self.lp.find_or_create_variable(tokens[2]);
        let lower = self.lp.variable_lower_bounds()[column];
        let upper = self.lp.variable_upper_bounds()[column];
        let value = if let Some(token) = tokens.get(3) {
            parse_number(line, token)?
        } else {
            0.0
        };
        match tokens[0].to_ascii_uppercase().as_str() {
            "LO" => self.lp.set_variable_bounds(column, value, upper),
            "UP" => self.lp.set_variable_bounds(column, lower, value),
            "FX" => self.lp.set_variable_bounds(column, value, value),
            "FR" => self.lp.set_variable_bounds(column, -INFINITY, INFINITY),
            "MI" => self.lp.set_variable_bounds(column, -INFINITY, upper),
            "PL" => self.lp.set_variable_bounds(column, lower, INFINITY),
            "BV" => {
                self.lp
                    .set_variable_type(column, ModelVariableType::Integer);
                self.lp.set_variable_bounds(column, 0.0, 1.0);
            }
            "LI" => {
                self.lp
                    .set_variable_type(column, ModelVariableType::Integer);
                self.lp.set_variable_bounds(column, value, upper);
            }
            "UI" => {
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

    fn finish(mut self, last_line: usize) -> Result<LinearProgram, MpsError> {
        if !self.ended {
            return Err(MpsError::new(last_line, "missing ENDATA"));
        }
        if self.objective_name.is_none() {
            return Err(MpsError::new(0, "missing objective row"));
        }
        self.lp.clean_up();
        self.lp
            .validate()
            .map_err(|message| MpsError::new(0, message))?;
        Ok(self.lp)
    }
}

#[derive(Clone, Copy)]
enum NamedVector {
    Rhs,
    Range,
}

fn parse_number(line: usize, token: &str) -> Result<Fractional, MpsError> {
    token
        .replace(['D', 'd'], "E")
        .parse()
        .map_err(|_| MpsError::new(line, format!("invalid number {token}")))
}

fn fixed_fields(line: &str, section: Section) -> Option<Vec<String>> {
    let bytes = line.as_bytes();
    let row_line = section == Section::Rows;
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
        Section::Rows => &[(1, 3), (4, 12)],
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
}
