//! Parser for the LP text emitted by [`LinearProgram::dump`](crate::lp_data::LinearProgram::dump).

use std::collections::HashSet;

use crate::lp_data::{LinearProgram, ModelVariableType};
use crate::lp_types::{ColIndex, Fractional, INFINITY, VectorIndex};

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedConstraint {
    pub name: String,
    pub variable_names: Vec<String>,
    pub coefficients: Vec<Fractional>,
    pub lower_bound: Fractional,
    pub upper_bound: Fractional,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LpParseError(pub String);

impl std::fmt::Display for LpParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for LpParseError {}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Error,
    End,
    Addend(Fractional, String),
    Value(Fractional),
    Infinity(Fractional),
    Name(String),
    LessEqual,
    Equal,
    GreaterEqual,
    Comma,
}

impl Token {
    const fn is_bound(&self) -> bool {
        matches!(self, Self::Value(_) | Self::Infinity(_))
    }
    const fn bound(&self) -> Fractional {
        match self {
            Self::Value(value) | Self::Infinity(value) => *value,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Copy)]
struct Lexer<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Lexer<'a> {
    const fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }
    fn remaining(self) -> &'a str {
        &self.input[self.position..]
    }
    fn skip_whitespace_from(&self, mut position: usize) -> usize {
        while self
            .input
            .as_bytes()
            .get(position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            position += 1;
        }
        position
    }
    #[allow(clippy::too_many_lines)] // Mirrors GLOP's ordered token recognizers.
    fn consume(&mut self) -> Token {
        let bytes = self.input.as_bytes();
        let start = self.skip_whitespace_from(self.position);
        if start == bytes.len() {
            self.position = start;
            return Token::End;
        }

        if is_word(bytes[start]) {
            let mut end = start + 1;
            while bytes
                .get(end)
                .is_some_and(|&byte| is_word(byte) || matches!(byte, b'[' | b']'))
            {
                end += 1;
            }
            if bytes.get(end) == Some(&b':') {
                self.position = end + 1;
                return Token::Name(self.input[start..end].to_owned());
            }
        }
        for keyword in ["int", "bin"] {
            if self.input[start..]
                .get(..keyword.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(keyword))
            {
                let mut end = self.skip_whitespace_from(start + keyword.len());
                if bytes.get(end) == Some(&b':') {
                    end += 1;
                }
                self.position = end;
                return Token::Name(keyword.to_owned());
            }
        }
        if bytes[start] == b'<' {
            self.position = start + 1 + usize::from(bytes.get(start + 1) == Some(&b'='));
            return Token::LessEqual;
        }
        if bytes[start] == b'=' {
            self.position = start + 1;
            return Token::Equal;
        }
        if bytes[start] == b'>' {
            self.position = start + 1 + usize::from(bytes.get(start + 1) == Some(&b'='));
            return Token::GreaterEqual;
        }
        if bytes[start] == b',' {
            self.position = start + 1;
            return Token::Comma;
        }

        let mut position = self.position;
        let mut minus_count = 0;
        loop {
            let sign = self.skip_whitespace_from(position);
            match bytes.get(sign) {
                Some(b'+') => position = sign + 1,
                Some(b'-') => {
                    minus_count += 1;
                    position = sign + 1;
                }
                _ => break,
            }
        }
        position = self.skip_whitespace_from(position);
        if self.input[position..]
            .get(..3)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("inf"))
        {
            self.position = position + 3;
            return Token::Infinity(if minus_count % 2 == 0 {
                INFINITY
            } else {
                -INFINITY
            });
        }

        let value_end = decimal_end(self.input, position);
        let has_value = value_end > position;
        let mut coefficient = if has_value {
            match self.input[position..value_end].parse::<f64>() {
                Ok(value) => value,
                Err(_) => return Token::Error,
            }
        } else {
            1.0
        };
        if minus_count % 2 == 1 {
            coefficient = -coefficient;
        }
        if has_value && !coefficient.is_finite() {
            self.position = value_end;
            return Token::Infinity(coefficient);
        }
        position = self.skip_whitespace_from(value_end.max(position));
        let multiplication = bytes.get(position) == Some(&b'*');
        if multiplication {
            position = self.skip_whitespace_from(position + 1);
        }
        if bytes
            .get(position)
            .is_some_and(|&byte| is_addend_initial(byte))
        {
            let name_start = position;
            position += 1;
            while bytes
                .get(position)
                .is_some_and(|&byte| is_word(byte) || matches!(byte, b'[' | b']' | b')'))
            {
                position += 1;
            }
            self.position = position;
            if multiplication && !has_value {
                return Token::Error;
            }
            return Token::Addend(coefficient, self.input[name_start..position].to_owned());
        }
        if has_value {
            self.position = value_end;
            Token::Value(coefficient)
        } else {
            Token::Error
        }
    }
}

const fn is_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}
const fn is_addend_initial(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b')')
}

fn decimal_end(input: &str, start: usize) -> usize {
    let bytes = input.as_bytes();
    let mut end = start;
    while matches!(bytes.get(end), Some(b'0'..=b'9')) {
        end += 1;
    }
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        let fraction_start = end;
        while matches!(bytes.get(end), Some(b'0'..=b'9')) {
            end += 1;
        }
        if end == fraction_start && fraction_start - 1 == start {
            return start;
        }
    }
    if end == start {
        return start;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let marker = end;
        end += 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let exponent_start = end;
        while matches!(bytes.get(end), Some(b'0'..=b'9')) {
            end += 1;
        }
        if end == exponent_start {
            end = marker;
        }
    }
    end
}

/// Parses one LP constraint with GLOP's exact structural rules.
///
/// # Errors
/// Returns the same first structural diagnostic as GLOP.
pub fn parse_constraint(input: &str) -> Result<ParsedConstraint, LpParseError> {
    let mut lexer = Lexer::new(input);
    let mut name = String::new();
    let checkpoint = lexer;
    if let Token::Name(parsed_name) = lexer.consume() {
        name = parsed_name;
    } else {
        lexer = checkpoint;
    }
    let mut left_bound = 0.0;
    let mut right_bound = 0.0;
    let mut left_sign = Token::End;
    let mut right_sign = Token::End;
    let mut token = lexer.consume();
    if token.is_bound() {
        left_bound = token.bound();
        left_sign = lexer.consume();
        if !matches!(
            left_sign,
            Token::LessEqual | Token::Equal | Token::GreaterEqual
        ) {
            return Err(error(
                "Expected an equality/inequality sign for the left bound.",
            ));
        }
        token = lexer.consume();
    }
    let mut variable_names = Vec::new();
    let mut coefficients = Vec::new();
    let mut used = HashSet::new();
    while let Token::Addend(coefficient, variable_name) = token {
        if !used.insert(variable_name.clone()) {
            return Err(error(format!("Duplicate variable name: {variable_name}")));
        }
        variable_names.push(variable_name);
        coefficients.push(coefficient);
        token = lexer.consume();
    }
    if left_sign == Token::Equal && token != Token::End {
        return Err(error("Equality constraints can have only one bound."));
    }
    if token != Token::End {
        right_sign = token;
        if !matches!(
            right_sign,
            Token::LessEqual | Token::Equal | Token::GreaterEqual
        ) {
            return Err(error(
                "Expected an equality/inequality sign for the right bound.",
            ));
        }
        if left_sign != Token::End && right_sign == Token::Equal {
            return Err(error("Equality constraints can have only one bound."));
        }
        let bound = lexer.consume();
        if !bound.is_bound() {
            return Err(error("Bound value was expected."));
        }
        right_bound = bound.bound();
        if lexer.consume() != Token::End {
            return Err(error(format!(
                "End of input was expected, found: {}",
                lexer.remaining()
            )));
        }
    }
    if left_sign == Token::End && right_sign == Token::End {
        return Err(error("The input constraint was empty."));
    }
    let mut lower_bound = -INFINITY;
    let mut upper_bound = INFINITY;
    if matches!(left_sign, Token::LessEqual | Token::Equal) {
        lower_bound = left_bound;
    }
    if matches!(left_sign, Token::GreaterEqual | Token::Equal) {
        upper_bound = left_bound;
    }
    if matches!(right_sign, Token::LessEqual | Token::Equal) {
        upper_bound = upper_bound.min(right_bound);
    }
    if matches!(right_sign, Token::GreaterEqual | Token::Equal) {
        lower_bound = lower_bound.max(right_bound);
    }
    Ok(ParsedConstraint {
        name,
        variable_names,
        coefficients,
        lower_bound,
        upper_bound,
    })
}

fn error(message: impl Into<String>) -> LpParseError {
    LpParseError(message.into())
}

/// Parses a complete semicolon-delimited LP model.
///
/// On failure the partially populated model is intentionally retained, as in GLOP.
pub fn parse_lp(model: &str, lp: &mut LinearProgram) -> bool {
    *lp = LinearProgram::new();
    let mut parser = ModelParser {
        lp,
        bounded: HashSet::new(),
    };
    let mut has_objective = false;
    for line in model.split(';').filter(|line| !line.is_empty()) {
        if !has_objective && parser.parse_objective(line) {
            has_objective = true;
        } else if !parser.parse_model_constraint(line)
            && !parser.parse_integer_list(line)
            && !is_empty(line)
        {
            return false;
        }
    }
    for column in 0..parser.lp.num_variables().to_usize() {
        let column = ColIndex::from_usize(column);
        if !parser.bounded.contains(&column) {
            parser.lp.set_variable_bounds(column, -INFINITY, INFINITY);
        }
    }
    parser.lp.clean_up();
    true
}

struct ModelParser<'a> {
    lp: &'a mut LinearProgram,
    bounded: HashSet<ColIndex>,
}

#[allow(clippy::float_cmp)] // Unit coefficients have structural meaning upstream.
impl ModelParser<'_> {
    fn parse_objective(&mut self, input: &str) -> bool {
        let mut lexer = Lexer::new(input);
        let Token::Name(direction) = lexer.consume() else {
            return false;
        };
        if direction.eq_ignore_ascii_case("min") {
            self.lp.set_maximization_problem(false);
        } else if direction.eq_ignore_ascii_case("max") {
            self.lp.set_maximization_problem(true);
        } else {
            return false;
        }
        let mut token = lexer.consume();
        if let Token::Value(offset) = token {
            self.lp.set_objective_offset(offset);
            token = lexer.consume();
        } else {
            self.lp.set_objective_offset(0.0);
        }
        while let Token::Addend(coefficient, name) = token {
            let column = self.lp.find_or_create_variable(&name);
            if self.lp.objective_coefficients()[column] != 0.0 {
                return false;
            }
            self.lp.set_objective_coefficient(column, coefficient);
            token = lexer.consume();
        }
        token == Token::End
    }
    fn parse_integer_list(&mut self, input: &str) -> bool {
        let mut lexer = Lexer::new(input);
        let Token::Name(keyword) = lexer.consume() else {
            return false;
        };
        let binary = keyword.eq_ignore_ascii_case("bin");
        if !binary && !keyword.eq_ignore_ascii_case("int") {
            return false;
        }
        let mut token = lexer.consume();
        while let Token::Addend(coefficient, name) = token {
            if coefficient != 1.0 {
                return false;
            }
            let column = self.lp.find_or_create_variable(&name);
            self.lp
                .set_variable_type(column, ModelVariableType::Integer);
            if binary && !self.set_variable_bounds(column, 0.0, 1.0) {
                return false;
            }
            token = lexer.consume();
            if token == Token::Comma {
                token = lexer.consume();
            }
        }
        token == Token::End
    }
    fn parse_model_constraint(&mut self, input: &str) -> bool {
        let Ok(parsed) = parse_constraint(input) else {
            return false;
        };
        if parsed.name.is_empty() && parsed.coefficients.len() == 1 && parsed.coefficients[0] == 1.0
        {
            let column = self.lp.find_or_create_variable(&parsed.variable_names[0]);
            return self.set_variable_bounds(column, parsed.lower_bound, parsed.upper_bound);
        }
        let before = self.lp.num_constraints();
        let row = if parsed.name.is_empty() {
            self.lp.create_new_constraint()
        } else {
            self.lp.find_or_create_constraint(&parsed.name)
        };
        if self.lp.num_constraints() == before
            || !bounds_valid(parsed.lower_bound, parsed.upper_bound)
        {
            return false;
        }
        self.lp
            .set_constraint_bounds(row, parsed.lower_bound, parsed.upper_bound);
        for (name, coefficient) in parsed.variable_names.iter().zip(&parsed.coefficients) {
            let column = self.lp.find_or_create_variable(name);
            self.lp.set_coefficient(row, column, *coefficient);
        }
        true
    }
    fn set_variable_bounds(&mut self, column: ColIndex, lower: f64, upper: f64) -> bool {
        if self.bounded.insert(column) {
            self.lp.set_variable_bounds(column, -INFINITY, INFINITY);
        }
        let lower = lower.max(self.lp.variable_lower_bounds()[column]);
        let upper = upper.min(self.lp.variable_upper_bounds()[column]);
        if !bounds_valid(lower, upper) {
            return false;
        }
        self.lp.set_variable_bounds(column, lower, upper);
        true
    }
}

fn is_empty(input: &str) -> bool {
    Lexer::new(input).consume() == Token::End
}
fn bounds_valid(lower: f64, upper: f64) -> bool {
    !(lower.is_nan()
        || upper.is_nan()
        || lower == INFINITY && upper == INFINITY
        || lower == -INFINITY && upper == -INFINITY)
        && lower <= upper
}

#[cfg(test)]
mod tests {
    use super::{parse_constraint, parse_lp};
    use crate::lp_data::LinearProgram;
    #[test]
    fn parses_dump_style_model() {
        let mut lp = LinearProgram::new();
        assert!(parse_lp(
            "min: 1 + x + 2*x2; 0 <= x <= 1; r: x-x2 >= -2; int x;",
            &mut lp
        ));
        assert_eq!(lp.num_variables().value(), 2);
        assert_eq!(lp.num_constraints().value(), 1);
    }
    #[test]
    fn rejects_duplicate_constraint_variables() {
        assert_eq!(
            parse_constraint("x + x <= 1").unwrap_err().0,
            "Duplicate variable name: x"
        );
    }
}
