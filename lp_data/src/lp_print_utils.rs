//! LP text formatting from `ortools/lp_data/lp_print_utils`.

/// Formats a double like upstream's `Stringify(double)` (`%.16g`).
#[must_use]
pub fn stringify(value: f64) -> String {
    stringify_significant(value, 16)
}

/// Formats a double as Abseil's default string append used by `DumpSolution`.
#[must_use]
pub fn stringify_default(value: f64) -> String {
    stringify_significant(value, 6)
}

/// Formats a finite double as GLOP's continued-fraction approximation.
///
/// On the pinned Apple arm64 reference platform, C++ `long double` has the
/// same 53-bit mantissa and exponent range as `double`, so these operations
/// deliberately use `f64` in the same order.
///
/// # Panics
///
/// Panics for NaN; infinities use GLOP's explicit `inf` spellings.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
pub fn stringify_rational(value: f64, precision: f64) -> String {
    if value == f64::INFINITY {
        return "inf".to_owned();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_owned();
    }
    assert!(value.is_finite());

    let absolute = value.abs();
    let mut remainder = absolute;
    let mut previous_numerator = 0_i64;
    let mut previous_denominator = 1_i64;
    let mut numerator = 1_i64;
    let mut denominator = 0_i64;
    loop {
        #[allow(clippy::cast_possible_truncation)]
        let term = remainder.floor() as i64;
        let new_numerator = term
            .wrapping_mul(numerator)
            .wrapping_add(previous_numerator);
        let new_denominator = term
            .wrapping_mul(denominator)
            .wrapping_add(previous_denominator);
        if new_numerator < 0 || new_denominator < 0 {
            break;
        }
        previous_numerator = numerator;
        previous_denominator = denominator;
        numerator = new_numerator;
        denominator = new_denominator;
        let numerator_approximation = absolute * denominator as f64;
        if (numerator_approximation - numerator as f64).abs() <= precision * numerator_approximation
        {
            break;
        }
        remainder = 1.0 / (remainder - term as f64);
    }
    if value < 0.0 {
        numerator = -numerator;
    }
    if denominator == 1 {
        numerator.to_string()
    } else {
        format!("{numerator}/{denominator}")
    }
}

/// Selects GLOP's rational or decimal representation.
#[must_use]
pub fn stringify_as(value: f64, fraction: bool) -> String {
    if fraction {
        stringify_rational(value, f64::EPSILON)
    } else {
        stringify(value)
    }
}

fn stringify_significant(value: f64, precision: i32) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value == f64::INFINITY {
        return "inf".to_owned();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }

    let fractional_precision = usize::try_from(precision - 1).unwrap_or(0);
    let scientific = format!("{value:.fractional_precision$e}");
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    if !(-4..precision).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!("{mantissa}e{exponent:+03}");
    }

    let fractional_digits = usize::try_from(precision - 1 - exponent).unwrap_or(0);
    let decimal = format!("{value:.fractional_digits$}");
    if decimal.contains('.') {
        decimal
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    } else {
        decimal
    }
}

/// Pretty-prints a nonzero monomial with GLOP's spacing and sign rules.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn stringify_monomial(coefficient: f64, variable: &str) -> String {
    stringify_monomial_as(coefficient, variable, false)
}

/// Pretty-prints a monomial using either rational or decimal coefficients.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn stringify_monomial_as(coefficient: f64, variable: &str, fraction: bool) -> String {
    if coefficient == 0.0 {
        String::new()
    } else if coefficient > 0.0 {
        if coefficient == 1.0 {
            format!(" + {variable}")
        } else {
            format!(" + {} {variable}", stringify_as(coefficient, fraction))
        }
    } else if coefficient == -1.0 {
        format!(" - {variable}")
    } else {
        format!(" - {} {variable}", stringify_as(-coefficient, fraction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_format_and_monomials_match_printf_g_conventions() {
        assert_eq!(stringify(1.25), "1.25");
        assert_eq!(stringify(1e-4), "0.0001");
        assert_eq!(stringify(1e-5), "1e-05");
        assert_eq!(stringify(1e16), "1e+16");
        assert_eq!(stringify_monomial(-1.0, "x"), " - x");
        assert_eq!(stringify_monomial(2.5, "x"), " + 2.5 x");
        assert_eq!(stringify_rational(1.0 / 3.0, f64::EPSILON), "1/3");
        assert_eq!(stringify_monomial_as(-0.125, "x", true), " - 1/8 x");
    }
}
