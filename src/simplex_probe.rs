//! Opt-in Matrix Market snapshots of bases and their direct solve inputs.
//!
//! This diagnostic has no role in the upstream algorithm. It records the
//! square basis at each successful LU construction and right-hand sides used
//! before the first rank-one update changes that basis.

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub enum SolveKind {
    Ftran,
    Btran,
}

#[derive(Debug)]
pub struct SimplexProbe {
    root: PathBuf,
    instance: String,
    pivot: u64,
    serial: usize,
    active: Option<(u64, usize)>,
    solve_basis_recorded: bool,
    ftran_count: usize,
    btran_count: usize,
    factorization_only: bool,
    source_recorded: bool,
}

impl SimplexProbe {
    /// Creates a recorder for one Netlib model.
    ///
    /// # Errors
    ///
    /// Rejects names that cannot safely form a single path component.
    pub fn new(root: impl Into<PathBuf>, instance: &str) -> io::Result<Self> {
        if instance.is_empty()
            || instance == "."
            || instance == ".."
            || !instance
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "instance name must be a safe path component",
            ));
        }
        Ok(Self {
            root: root.into(),
            instance: instance.to_owned(),
            pivot: 0,
            serial: 0,
            active: None,
            solve_basis_recorded: false,
            ftran_count: 0,
            btran_count: 0,
            factorization_only: false,
            source_recorded: false,
        })
    }

    /// Suppress solve-basis and RHS snapshots when only LU inputs are needed.
    #[must_use]
    pub fn factorizations_only(mut self) -> Self {
        self.factorization_only = true;
        self
    }

    pub fn set_pivot(&mut self, pivot: u64) {
        self.pivot = pivot;
    }

    #[must_use]
    pub const fn factorization_only(&self) -> bool {
        self.factorization_only
    }

    #[must_use]
    pub const fn source_recorded(&self) -> bool {
        self.source_recorded
    }

    /// Saves the immutable compact problem matrix once for selected-view benchmarks.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or rejects a conflicting preexisting source.
    pub fn record_source_matrix(
        &mut self,
        rows: usize,
        columns: usize,
        entries: &[(usize, usize, f64)],
    ) -> io::Result<()> {
        if !self.source_recorded {
            write_unchanged_or_new(
                &self.root.join(format!("{}-source.mtx", self.instance)),
                &matrix_market(rows, columns, entries),
            )?;
            self.source_recorded = true;
        }
        Ok(())
    }

    /// Saves zero-based source-column indices for the next basis snapshot.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or rejects conflicting preexisting indices.
    pub fn record_selection(&self, columns: &[usize]) -> io::Result<()> {
        let mut contents = format!("{}\n", columns.len());
        for &column in columns {
            writeln!(contents, "{column}").expect("writing to String cannot fail");
        }
        write_unchanged_or_new(
            &self
                .directory()
                .join(format!("selection-{:04}.txt", self.serial)),
            &contents,
        )
    }

    /// Records a newly constructed square basis in column/entry order.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or rejects a conflicting preexisting probe.
    pub fn record_basis(
        &mut self,
        dimension: usize,
        entries: &[(usize, usize, f64)],
    ) -> io::Result<()> {
        let serial = self.serial;
        let directory = self.directory();
        let contents = matrix_market(dimension, dimension, entries);
        write_unchanged_or_new(
            &directory.join(format!("factorization-{serial:04}.mtx")),
            &contents,
        )?;
        self.active = Some((self.pivot, serial));
        self.solve_basis_recorded = false;
        self.serial += 1;
        self.ftran_count = 0;
        self.btran_count = 0;
        Ok(())
    }

    #[must_use]
    #[inline]
    pub fn should_capture_rhs(&self, basis_has_updates: bool) -> bool {
        !self.factorization_only
            && matches!(self.active, Some((pivot, _)) if pivot == self.pivot)
            && !basis_has_updates
    }

    #[must_use]
    pub const fn solve_basis_recorded(&self) -> bool {
        self.solve_basis_recorded
    }

    /// Records the current solve basis, which may have an absorbed column
    /// permutation relative to the factorization input.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or rejects a conflicting preexisting probe.
    pub fn ensure_solve_basis(
        &mut self,
        dimension: usize,
        entries: &[(usize, usize, f64)],
    ) -> io::Result<()> {
        if self.solve_basis_recorded {
            return Ok(());
        }
        if let Some((pivot, serial)) = self.active
            && pivot == self.pivot
        {
            write_unchanged_or_new(
                &self.directory().join(format!("basis-{serial:04}.mtx")),
                &matrix_market(dimension, dimension, entries),
            )?;
            self.solve_basis_recorded = true;
        }
        Ok(())
    }

    /// Records a right-hand side only while the saved basis remains current.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or rejects a conflicting preexisting probe.
    pub fn record_rhs(
        &mut self,
        kind: SolveKind,
        rhs: &[f64],
        basis_has_updates: bool,
    ) -> io::Result<()> {
        let Some((basis_pivot, serial)) = self.active else {
            return Ok(());
        };
        if self.pivot != basis_pivot || basis_has_updates {
            return Ok(());
        }
        let directory = self.directory();
        let (label, count) = match kind {
            SolveKind::Ftran => ("ftran", &mut self.ftran_count),
            SolveKind::Btran => ("btran", &mut self.btran_count),
        };
        let entries: Vec<_> = rhs
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0 || value.is_sign_negative())
            .map(|(row, &value)| (row, 0, value))
            .collect();
        let contents = matrix_market(rhs.len(), 1, &entries);
        write_unchanged_or_new(
            &directory.join(format!("{label}-{serial:04}-{:06}.mtx", *count)),
            &contents,
        )?;
        *count += 1;
        Ok(())
    }

    fn directory(&self) -> PathBuf {
        self.root.join(format!("{}-{}", self.instance, self.pivot))
    }
}

fn matrix_market(rows: usize, columns: usize, entries: &[(usize, usize, f64)]) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    output.push_str("%%MatrixMarket matrix coordinate real general\n");
    writeln!(&mut output, "{rows} {columns} {}", entries.len()).unwrap();
    for &(row, column, value) in entries {
        writeln!(&mut output, "{} {} {value:.17e}", row + 1, column + 1).unwrap();
    }
    output
}

fn write_unchanged_or_new(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => file.write_all(contents.as_bytes()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if fs::read_to_string(path)? == contents {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("conflicting existing probe: {}", path.display()),
                ))
            }
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::{SimplexProbe, SolveKind, matrix_market};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn coordinate_matrix_market_preserves_order_and_signed_zero() {
        assert_eq!(
            matrix_market(2, 2, &[(1, 0, -0.0), (0, 1, 2.5)]),
            "%%MatrixMarket matrix coordinate real general\n2 2 2\n2 1 -0.00000000000000000e0\n1 2 2.50000000000000000e0\n"
        );
    }

    #[test]
    fn unsafe_instance_names_are_rejected() {
        assert!(SimplexProbe::new("ignored", "../escape").is_err());
        assert!(SimplexProbe::new("ignored", "pilot.ja").is_ok());
    }

    #[test]
    fn factorization_only_disables_rhs_capture() {
        let mut probe = SimplexProbe::new("ignored", "case")
            .unwrap()
            .factorizations_only();
        probe.active = Some((0, 0));
        assert!(!probe.should_capture_rhs(false));
    }

    #[test]
    fn selected_view_snapshots_keep_source_and_basis_order() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gloprs-selected-probe-{}-{nonce}",
            std::process::id()
        ));
        let source = [(0, 0, 2.0), (1, 1, 3.0), (0, 2, 5.0)];
        let mut probe = SimplexProbe::new(&root, "case")
            .unwrap()
            .factorizations_only();
        probe.record_source_matrix(2, 3, &source).unwrap();
        probe.record_selection(&[2, 1]).unwrap();
        probe.record_basis(2, &[(0, 0, 5.0), (1, 1, 3.0)]).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("case-0/selection-0000.txt")).unwrap(),
            "2\n2\n1\n"
        );
        assert!(
            fs::read_to_string(root.join("case-source.mtx"))
                .unwrap()
                .contains("2 3 3")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshots_are_pivot_scoped_rerunnable_and_keep_both_basis_orders() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("gloprs-probe-{}-{nonce}", std::process::id()));
        let factorization = [(0, 1, 2.0), (1, 0, 3.0)];
        let solve_basis = [(0, 0, 2.0), (1, 1, 3.0)];
        for _ in 0..2 {
            let mut probe = SimplexProbe::new(&root, "case").unwrap();
            probe.record_basis(2, &factorization).unwrap();
            probe.ensure_solve_basis(2, &solve_basis).unwrap();
            probe
                .record_rhs(SolveKind::Ftran, &[4.0, -0.0], false)
                .unwrap();
            assert!(!probe.should_capture_rhs(true));
            probe.set_pivot(1);
            assert!(!probe.should_capture_rhs(false));
            probe.record_basis(2, &solve_basis).unwrap();
            probe.ensure_solve_basis(2, &solve_basis).unwrap();
            probe
                .record_rhs(SolveKind::Btran, &[0.0, 5.0], false)
                .unwrap();
        }
        assert_ne!(
            fs::read(root.join("case-0/factorization-0000.mtx")).unwrap(),
            fs::read(root.join("case-0/basis-0000.mtx")).unwrap()
        );
        assert_eq!(
            fs::read_to_string(root.join("case-0/ftran-0000-000000.mtx"))
                .unwrap()
                .lines()
                .nth(1),
            Some("2 1 2")
        );
        fs::remove_dir_all(&root).unwrap();
    }
}
