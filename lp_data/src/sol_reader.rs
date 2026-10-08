//! Reader for the SOL format used by GLOP.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::lp_data::LinearProgram;
use crate::lp_types::{ColIndex, DenseRow, VectorIndex};

#[derive(Debug)]
pub enum SolReadError {
    Io(std::io::Error),
    InvalidArgument(String),
}

impl fmt::Display for SolReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::InvalidArgument(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SolReadError {}

/// Parses a SOL file for `model`.
///
/// # Errors
/// Returns an I/O error or the first malformed/unknown-variable diagnostic.
pub fn parse_sol_file(
    path: impl AsRef<Path>,
    model: &LinearProgram,
) -> Result<DenseRow, SolReadError> {
    let solution = std::fs::read_to_string(path).map_err(SolReadError::Io)?;
    parse_sol_string(&solution, model)
}

/// Parses a SOL string for `model`, matching GLOP's overwrite semantics.
///
/// # Errors
/// Returns the first malformed-line, malformed-value, or unknown-variable diagnostic.
pub fn parse_sol_string(solution: &str, model: &LinearProgram) -> Result<DenseRow, SolReadError> {
    let mut variable_by_name = HashMap::new();
    for column in 0..model.num_variables().to_usize() {
        let column = ColIndex::from_usize(column);
        variable_by_name.insert(model.variable_name(column), column);
    }
    let mut values = DenseRow::from_vec(vec![0.0; model.num_variables().to_usize()]);
    for line in solution.split('\n').filter(|line| !line.is_empty()) {
        let mut fields: Vec<&str> = line
            .split([' ', '\t'])
            .filter(|field| !field.is_empty())
            .collect();
        if let Some(comment) = fields.iter().position(|field| field.starts_with('#')) {
            fields.truncate(comment);
        }
        if fields.is_empty() {
            continue;
        }
        if fields.len() == 1 {
            return Err(invalid(format!("Found only one field on line '{line}'.")));
        }
        if fields.len() > 2 {
            return Err(invalid(format!(
                "Found more than two fields on line '{line}'."
            )));
        }
        let value = parse_leading_double(fields[1]).unwrap_or(f64::INFINITY);
        if value == f64::INFINITY {
            return Err(invalid(format!("Couldn't parse value on line '{line}'.")));
        }
        if fields[0] == "=obj=" {
            continue;
        }
        let Some(&column) = variable_by_name.get(fields[0]) else {
            return Err(invalid(format!(
                "Couldn't find variable named '{}' in the model.",
                fields[0]
            )));
        };
        values[column] = value;
    }
    Ok(values)
}

fn invalid(message: String) -> SolReadError {
    SolReadError::InvalidArgument(message)
}

fn parse_leading_double(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let tail = text.get(end..)?;
    let lowercase = tail.to_ascii_lowercase();
    if lowercase.starts_with("nan") {
        return Some(f64::NAN.copysign(if text.starts_with('-') { -1.0 } else { 1.0 }));
    }
    if lowercase.starts_with("infinity") || lowercase.starts_with("inf") {
        return Some(if text.starts_with('-') {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if tail.starts_with("0x") || tail.starts_with("0X") {
        return parse_leading_hex_double(text, end);
    }
    let mut digits = 0;
    while matches!(bytes.get(end), Some(b'0'..=b'9')) {
        end += 1;
        digits += 1;
    }
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        while matches!(bytes.get(end), Some(b'0'..=b'9')) {
            end += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let exponent = end;
        end += 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let start = end;
        while matches!(bytes.get(end), Some(b'0'..=b'9')) {
            end += 1;
        }
        if start == end {
            end = exponent;
        }
    }
    let value: f64 = text[..end].parse().ok()?;
    reject_range_error(value, &text[..end])
}

#[allow(clippy::too_many_lines)] // The bit-level conversion is kept in source order.
fn parse_leading_hex_double(text: &str, sign_end: usize) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut end = sign_end + 2;
    let mut digits = Vec::new();
    let mut digits_before_point = 0_i32;
    while let Some(digit) = bytes.get(end).and_then(|&byte| hex_digit(byte)) {
        digits.push(digit);
        end += 1;
        digits_before_point += 1;
    }
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        while let Some(digit) = bytes.get(end).and_then(|&byte| hex_digit(byte)) {
            digits.push(digit);
            end += 1;
        }
    }
    if digits.is_empty() {
        return Some(if text.starts_with('-') { -0.0 } else { 0.0 });
    }
    let mut parsed_exponent = 0_i32;
    if matches!(bytes.get(end), Some(b'p' | b'P')) {
        end += 1;
        let negative = bytes.get(end) == Some(&b'-');
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let exponent_start = end;
        let mut exponent = 0_i32;
        while let Some(byte @ b'0'..=b'9') = bytes.get(end).copied() {
            exponent = exponent
                .saturating_mul(10)
                .saturating_add(i32::from(byte - b'0'));
            end += 1;
        }
        if exponent_start != end && negative {
            parsed_exponent = -exponent;
        } else if exponent_start != end {
            parsed_exponent = exponent;
        }
    }

    let Some(first_nonzero) = digits.iter().position(|&digit| digit != 0) else {
        return Some(if text.starts_with('-') { -0.0 } else { 0.0 });
    };
    let leading_zero_bits = digits[first_nonzero].leading_zeros() - 4;
    let significant_bits = 4 * (digits.len() - first_nonzero) - leading_zero_bits as usize;
    let digits_before = digits_before_point;
    let first_digit = i32::try_from(first_nonzero).unwrap_or(i32::MAX);
    let exponent = parsed_exponent
        .saturating_add(4_i32.saturating_mul(digits_before.saturating_sub(first_digit + 1)))
        .saturating_add(3_i32.saturating_sub(i32::try_from(leading_zero_bits).unwrap_or(3)));
    if !(-1074..=1023).contains(&exponent) {
        return None;
    }

    let normal = exponent >= -1022;
    let retained_bits = if normal {
        53
    } else {
        usize::try_from(exponent + 1075).ok()?
    };

    let mut retained = 0_u64;
    let mut seen = 0_usize;
    let mut guard = false;
    let mut sticky = false;
    for (digit_index, &digit) in digits.iter().enumerate().skip(first_nonzero) {
        for bit_index in (0..4).rev() {
            let bit = (digit >> bit_index) & 1;
            if digit_index == first_nonzero && seen == 0 && bit == 0 {
                continue;
            }
            match seen.cmp(&retained_bits) {
                std::cmp::Ordering::Less => {
                    retained = (retained << 1) | u64::from(bit);
                }
                std::cmp::Ordering::Equal => guard = bit != 0,
                std::cmp::Ordering::Greater => sticky |= bit != 0,
            }
            seen += 1;
        }
    }
    if significant_bits < retained_bits {
        retained <<= retained_bits - significant_bits;
    } else if !normal && (guard || sticky) {
        // Apple's strtod reports ERANGE for an inexact subnormal conversion,
        // but accepts an exactly representable hexadecimal subnormal.
        return None;
    } else if normal && guard && (sticky || retained & 1 != 0) {
        retained += 1;
    }
    if !normal {
        let sign_bit = u64::from(text.starts_with('-')) << 63;
        return Some(f64::from_bits(sign_bit | retained));
    }
    let mut rounded_exponent = exponent;
    if retained == 1_u64 << 53 {
        retained >>= 1;
        rounded_exponent += 1;
        if rounded_exponent > 1023 {
            return None;
        }
    }
    let exponent_bits = u64::try_from(rounded_exponent + 1023).ok()? << 52;
    let fraction_bits = retained & ((1_u64 << 52) - 1);
    let sign_bit = u64::from(text.starts_with('-')) << 63;
    Some(f64::from_bits(sign_bit | exponent_bits | fraction_bits))
}

const fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn reject_range_error(value: f64, parsed: &str) -> Option<f64> {
    let unsigned = parsed.trim_start_matches(['+', '-']);
    let nonzero_source = if unsigned.starts_with("0x") || unsigned.starts_with("0X") {
        unsigned.split(['p', 'P']).next().is_some_and(|mantissa| {
            mantissa
                .bytes()
                .any(|byte| matches!(byte, b'1'..=b'9' | b'a'..=b'f' | b'A'..=b'F'))
        })
    } else {
        unsigned
            .split(['e', 'E'])
            .next()
            .is_some_and(|mantissa| mantissa.bytes().any(|byte| matches!(byte, b'1'..=b'9')))
    };
    if value.is_infinite() || (nonzero_source && (value == 0.0 || value.abs() < f64::MIN_POSITIVE))
    {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_sol_string;
    use crate::lp_data::LinearProgram;

    #[test]
    fn comments_objective_and_overwrites_match_sol_semantics() {
        let mut model = LinearProgram::new();
        model.find_or_create_variable("x");
        model.find_or_create_variable("y");
        let values = parse_sol_string("x 1\n=obj= 9\ny 2 # comment\nx 3junk\n", &model).unwrap();
        assert_eq!(values.as_slice(), &[3.0, 2.0]);
    }
}
