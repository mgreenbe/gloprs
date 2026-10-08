//! Distribution statistics matching the non-timing surface of
//! `ortools/util/stats.{h,cc}`.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistributionKind {
    Ratio,
    Double,
    Integer,
}

#[derive(Clone, Debug)]
struct Distribution {
    name: String,
    kind: DistributionKind,
    sum: f64,
    average: f64,
    sum_squares_from_average: f64,
    minimum: f64,
    maximum: f64,
    count: i64,
}

impl Distribution {
    fn new(name: &str, kind: DistributionKind) -> Self {
        Self {
            name: name.to_owned(),
            kind,
            sum: 0.0,
            average: 0.0,
            sum_squares_from_average: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, value: f64) {
        if self.count == 0 {
            self.minimum = value;
            self.maximum = value;
            self.sum = value;
            self.average = value;
            self.count = 1;
            return;
        }
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.sum += value;
        self.count += 1;
        let delta = value - self.average;
        #[allow(clippy::cast_precision_loss)]
        let count = self.count as f64;
        self.average = self.sum / count;
        self.sum_squares_from_average += delta * (value - self.average);
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.average = 0.0;
        self.sum_squares_from_average = 0.0;
        self.minimum = 0.0;
        self.maximum = 0.0;
        self.count = 0;
    }

    fn standard_deviation(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let count = self.count as f64;
        (self.sum_squares_from_average / count).sqrt()
    }

    fn value_as_string(&self) -> String {
        match self.kind {
            DistributionKind::Ratio => format!(
                "{:8} [{:7.2}%, {:7.2}%] {:7.2}% {:7.2}%\n",
                self.count,
                100.0 * self.minimum,
                100.0 * self.maximum,
                100.0 * self.average,
                100.0 * self.standard_deviation()
            ),
            DistributionKind::Double => format!(
                "{:8} [{}, {}] {} {}\n",
                self.count,
                scientific(self.minimum),
                scientific(self.maximum),
                scientific(self.average),
                scientific(self.standard_deviation())
            ),
            DistributionKind::Integer => format!(
                "{:8} [{:8.0}, {:8.0}] {:8.2} {:8.2} {:8.0}\n",
                self.count,
                self.minimum,
                self.maximum,
                self.average,
                self.standard_deviation(),
                self.sum
            ),
        }
    }
}

fn scientific(value: f64) -> String {
    let formatted = format!("{value:.1e}");
    let Some((mantissa, exponent)) = formatted.split_once('e') else {
        return formatted;
    };
    let exponent: i32 = exponent.parse().expect("Rust emitted a valid exponent");
    format!("{:>8}", format!("{mantissa}e{exponent:+03}"))
}

#[derive(Clone, Debug)]
pub struct StatsGroup {
    name: String,
    distributions: Vec<Distribution>,
}

impl StatsGroup {
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            distributions: Vec::new(),
        }
    }

    /// Adds one observation to the named distribution.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if an existing name is reused with a different
    /// distribution kind.
    pub fn add(&mut self, name: &str, kind: DistributionKind, value: f64) {
        let distribution = if let Some(index) = self
            .distributions
            .iter()
            .position(|distribution| distribution.name == name)
        {
            &mut self.distributions[index]
        } else {
            self.distributions.push(Distribution::new(name, kind));
            self.distributions.last_mut().expect("just appended")
        };
        debug_assert_eq!(distribution.kind, kind);
        distribution.add(value);
    }

    pub fn reset(&mut self) {
        for distribution in &mut self.distributions {
            distribution.reset();
        }
    }

    /// Prepends an accumulated history to this group's single observations.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if this group contains more than the one new
    /// observation per distribution produced by one Markowitz factorization.
    pub(crate) fn prepend_history(&mut self, history: &Self) {
        let current = std::mem::replace(self, history.clone());
        for distribution in current.distributions {
            debug_assert!(distribution.count <= 1);
            if distribution.count == 1 {
                self.add(&distribution.name, distribution.kind, distribution.average);
            }
        }
    }

    #[must_use]
    pub fn stat_string(&self) -> String {
        let mut distributions: Vec<_> = self
            .distributions
            .iter()
            .filter(|distribution| distribution.count != 0)
            .collect();
        if distributions.is_empty() {
            return String::new();
        }
        distributions.sort_by(|left, right| {
            right
                .sum
                .total_cmp(&left.sum)
                .then_with(|| left.name.cmp(&right.name))
        });
        let longest_name = distributions
            .iter()
            .map(|distribution| distribution.name.chars().count())
            .max()
            .unwrap_or(0);
        let mut output = format!("{} {{\n", self.name);
        for distribution in distributions {
            output.push_str("  ");
            output.push_str(&distribution.name);
            output.push_str(&" ".repeat(longest_name - distribution.name.chars().count()));
            output.push_str(" : ");
            output.push_str(&distribution.value_as_string());
        }
        output.push_str("}\n");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_reset_groups_do_not_print() {
        let mut stats = StatsGroup::new("Stats");
        assert!(stats.stat_string().is_empty());
        stats.add("value", DistributionKind::Integer, 2.0);
        assert!(!stats.stat_string().is_empty());
        stats.reset();
        assert!(stats.stat_string().is_empty());
    }
}
