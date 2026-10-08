#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::format_collect,
    clippy::too_many_lines
)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use glop::basis_representation::BasisRepresentation;
use glop::entering_variable::EnteringVariable;
use glop::initial_basis::InitialBasis;
use glop::parameters::GlopParameters;
use glop::pricing::DynamicMaximum;
use glop::primal_ratio_test::{LeavingChoice, choose_leaving_variable_row};
use glop::reduced_costs::{PrimalPrices, ReducedCosts};
use glop::update_row::UpdateRow;
use glop::variable_values::VariableValues;
use glop::variables_info::VariablesInfo;
use lp_data::lp_types::{
    ColIndex, DenseRow, INVALID_COL, RowIndex, RowToColMapping, VariableStatus, VariableType,
    VariableTypeRow, VectorIndex,
};
use lp_data::scattered_vector::ScatteredColumn;
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

fn sparse_matrix(values: &[Vec<f64>]) -> SparseMatrix {
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(
        RowIndex::from_usize(values.len()),
        ColIndex::from_usize(values[0].len()),
    );
    for (row, entries) in values.iter().enumerate() {
        for (column, &value) in entries.iter().enumerate() {
            if value != 0.0 {
                matrix
                    .mutable_column(ColIndex::from_usize(column))
                    .add_entry(RowIndex::from_usize(row), value);
            }
        }
    }
    matrix.clean_up();
    matrix
}

#[test]
fn reduced_costs_values_and_prices_match_their_definitions() {
    // Two structural columns followed by GLOP's trailing identity slacks.
    let full = sparse_matrix(&[vec![2.0, -1.0, 1.0, 0.0], vec![1.0, 3.0, 0.0, 1.0]]);
    let compact = CompactSparseMatrix::from_sparse(&full);
    let basis = RowToColMapping::from_vec(vec![ColIndex::new(2), ColIndex::new(3)]);
    let identity = sparse_matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let factorization = BasisRepresentation::new(identity, 0.1, 20).unwrap();
    let mut info = VariablesInfo::new(&full);
    info.load_bounds_and_return_true_if_unchanged(
        &[0.0, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    );
    info.initialize_to_default_status();
    info.update_to_basic_status(ColIndex::new(2));
    info.update_to_basic_status(ColIndex::new(3));
    let objective = DenseRow::from_vec(vec![-4.0, 2.0, 0.5, -1.5]);
    let mut reduced = ReducedCosts::new(&compact, &objective, &basis, &info, &factorization, 1);
    let values = reduced.reduced_costs().unwrap();
    // y = c_B because B = I; r = c - A^T y.
    assert_eq!(values, &[-3.5, 7.0, 0.0, 0.0]);
    assert_eq!(reduced.dual_values().unwrap(), &[0.5, -1.5]);
    assert_eq!(reduced.compute_maximum_dual_residual().unwrap(), 0.0);

    let mut variable_values = VariableValues::new(
        &GlopParameters::default(),
        &compact,
        &basis,
        &info,
        &factorization,
    );
    variable_values.reset_all_non_basic_variable_values(&DenseRow::new());
    let mut dual_prices = DynamicMaximum::new(1);
    variable_values
        .recompute_basic_variable_values(&mut dual_prices)
        .unwrap();
    assert_eq!(variable_values.dense_row().as_slice(), &[0.0; 4]);
    assert_eq!(variable_values.compute_maximum_primal_residual(), 0.0);

    // Column 1 cannot decrease from its lower bound, so column 0 enters.
    let mut norms = glop::primal_edge_norms::PrimalEdgeNorms::new(&full);
    let mut prices = PrimalPrices::new(1);
    let reduced_values = reduced.reduced_costs().unwrap().to_vec();
    let squared_norms = norms
        .squared_norms(&factorization, info.relevance())
        .unwrap()
        .to_vec();
    assert_eq!(
        prices.best_entering_column_from_values(&info, &reduced_values, &squared_norms, 1e-8,),
        Some(ColIndex::new(0))
    );
    prices.force_recomputation();
    assert_eq!(
        prices
            .best_entering_column(&info, &factorization, &mut norms, &mut reduced)
            .unwrap(),
        Some(ColIndex::new(0))
    );
}

#[test]
fn primal_harris_ratio_test_covers_pivot_flip_and_refactorization() {
    let basis = RowToColMapping::from_vec(vec![ColIndex::new(2), ColIndex::new(3)]);
    let mut direction = ScatteredColumn::new(RowIndex::new(2));
    direction.set(RowIndex::new(0), 1.0);
    direction.set(RowIndex::new(1), 0.5);
    let lower = [0.0, 0.0, 0.0, 0.0];
    let upper = [10.0, 10.0, 10.0, 10.0];
    let values = [0.0, 0.0, 2.0, 3.0];
    let parameters = GlopParameters::default();
    assert_eq!(
        choose_leaving_variable_row(
            ColIndex::new(0),
            -1.0,
            &direction,
            1.0,
            &values,
            &lower,
            &upper,
            &basis,
            true,
            &parameters,
        ),
        LeavingChoice::Pivot {
            row: RowIndex::new(0),
            step: 2.0,
            target_bound: 0.0,
        }
    );
    let mut bounded_values = values;
    bounded_values[0] = 9.5;
    assert_eq!(
        choose_leaving_variable_row(
            ColIndex::new(0),
            -1.0,
            &direction,
            1.0,
            &bounded_values,
            &lower,
            &upper,
            &basis,
            true,
            &parameters,
        ),
        LeavingChoice::BoundFlip { step: 0.5 }
    );
    let mut tiny = ScatteredColumn::new(RowIndex::new(2));
    tiny.set(RowIndex::new(0), 1e-8);
    let mut permissive = parameters;
    permissive.ratio_test_zero_threshold = 1e-12;
    let mut distant_upper = upper;
    distant_upper[0] = 1e12;
    assert_eq!(
        choose_leaving_variable_row(
            ColIndex::new(0),
            -1.0,
            &tiny,
            1.0,
            &values,
            &lower,
            &distant_upper,
            &basis,
            false,
            &permissive,
        ),
        LeavingChoice::Refactorize
    );
}

#[derive(Clone)]
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn usize(&mut self, limit: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(limit).unwrap()).unwrap()
    }
}

#[test]
fn initial_basis_crashes_agree_with_native_glop() {
    let adapter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/native/initial_basis_reference_adapter");
    if !adapter.exists() {
        eprintln!("skipping native initial-basis differential test: adapter not built");
        return;
    }
    let mut generator = Generator(0x811c_9dc5_2437_18af);
    for case in 0..75 {
        let rows = 2 + generator.usize(7);
        let structural = 2 + generator.usize(9);
        let columns = structural + rows;
        let mut values = vec![vec![0.0; columns]; rows];
        for column in 0..structural {
            let mut maximum = 0.0_f64;
            for row_values in values.iter_mut().take(rows) {
                if generator.usize(4) != 0 {
                    let value = f64::from(i32::try_from(generator.usize(17)).unwrap() - 8);
                    row_values[column] = value;
                    maximum = maximum.max(value.abs());
                }
            }
            if maximum != 0.0 {
                for row_values in values.iter_mut().take(rows) {
                    row_values[column] /= maximum;
                }
            }
        }
        for row in 0..rows {
            values[row][structural + row] = 1.0;
        }
        let matrix = sparse_matrix(&values);
        let compact = CompactSparseMatrix::from_sparse(&matrix);
        let objective = DenseRow::from_vec(
            (0..columns)
                .map(|column| {
                    if column >= structural || (case + column) % 4 == 0 {
                        0.0
                    } else {
                        (column + 1) as f64 / 17.0
                    }
                })
                .collect(),
        );
        let lower = DenseRow::from_vec(
            (0..columns)
                .map(|column| -3.0 + (column % 3) as f64)
                .collect(),
        );
        let upper = DenseRow::from_vec(
            (0..columns)
                .map(|column| 2.0 + (column % 5) as f64)
                .collect(),
        );
        let types = VariableTypeRow::from_vec(
            (0..columns)
                .map(|column| match (column + case) % 5 {
                    0 => VariableType::Unconstrained,
                    1 => VariableType::LowerBounded,
                    2 => VariableType::UpperBounded,
                    3 => VariableType::UpperAndLowerBounded,
                    _ => VariableType::FixedVariable,
                })
                .collect(),
        );
        for mode in 0..5 {
            let candidate_columns = if mode == 0 { structural } else { columns };
            let initial = RowToColMapping::filled(RowIndex::from_usize(rows), INVALID_COL);
            let mut rust_basis = initial.clone();
            let mut crash = InitialBasis::new(&compact, &objective, &lower, &upper, &types);
            match mode {
                0 => crash
                    .complete_bixby_basis(ColIndex::from_usize(candidate_columns), &mut rust_basis),
                1 => crash.complete_triangular_primal_basis(
                    ColIndex::from_usize(candidate_columns),
                    &mut rust_basis,
                ),
                2 => crash.complete_triangular_dual_basis(
                    ColIndex::from_usize(candidate_columns),
                    &mut rust_basis,
                ),
                3 => crash.get_primal_maros_basis(
                    ColIndex::from_usize(candidate_columns),
                    &mut rust_basis,
                ),
                _ => crash
                    .get_dual_maros_basis(ColIndex::from_usize(candidate_columns), &mut rust_basis),
            }
            let mut input = format!(
                "{mode} {rows} {columns} {} {candidate_columns}\n",
                matrix.num_entries().value()
            );
            for column in 0..columns {
                for entry in matrix.column(ColIndex::from_usize(column)) {
                    writeln!(
                        input,
                        "{} {column} {:.17}",
                        entry.index().value(),
                        entry.coefficient()
                    )
                    .unwrap();
                }
            }
            for column in 0..columns {
                writeln!(
                    input,
                    "{:.17} {:.17} {:.17} {}",
                    objective[ColIndex::from_usize(column)],
                    lower[ColIndex::from_usize(column)],
                    upper[ColIndex::from_usize(column)],
                    types[ColIndex::from_usize(column)] as i8
                )
                .unwrap();
            }
            for column in initial.as_slice() {
                write!(input, "{} ", column.value()).unwrap();
            }
            input.push('\n');
            let mut child = Command::new(&adapter)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(input.as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(output.status.success());
            let native = String::from_utf8(output.stdout).unwrap();
            let expected = format!(
                "basis{}\n",
                rust_basis
                    .as_slice()
                    .iter()
                    .map(|column| format!(" {}", column.value()))
                    .collect::<String>()
            );
            assert_eq!(native, expected, "case {case}, mode {mode}\n{input}");
        }
    }
}

#[test]
fn simplex_state_agrees_with_native_glop_on_generated_fixtures() {
    let adapter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/native/simplex_state_reference_adapter");
    if !adapter.exists() {
        eprintln!("skipping native simplex-state differential test: adapter not built");
        return;
    }
    let mut generator = Generator(0xd6e8_feb8_6659_fd93);
    for case in 0..100 {
        let rows = 1 + generator.usize(7);
        let structural = 1 + generator.usize(10);
        let columns = structural + rows;
        let mut dense = vec![vec![0.0; columns]; rows];
        for column in 0..structural {
            for row_values in &mut dense {
                if generator.usize(3) != 0 {
                    row_values[column] =
                        f64::from(i32::try_from(generator.usize(19)).unwrap() - 9) / 7.0;
                }
            }
        }
        for row in 0..rows {
            dense[row][structural + row] = 1.0;
        }
        let matrix = sparse_matrix(&dense);
        let compact = CompactSparseMatrix::from_sparse(&matrix);
        let objective = DenseRow::from_vec(
            (0..columns)
                .map(|column| ((case + 3 * column) % 13) as f64 / 5.0 - 1.0)
                .collect(),
        );
        let lower_values = (0..columns)
            .map(|column| -3.0 + ((case + column) % 4) as f64)
            .collect::<Vec<_>>();
        let upper_values = (0..columns)
            .map(|column| {
                if (case + column) % 7 == 0 {
                    lower_values[column]
                } else {
                    2.0 + ((case + 2 * column) % 5) as f64
                }
            })
            .collect::<Vec<_>>();
        let mut info = VariablesInfo::new(&matrix);
        info.load_bounds_and_return_true_if_unchanged(&lower_values, &upper_values);
        info.initialize_to_default_status();
        let basis = RowToColMapping::from_vec(
            (0..rows)
                .map(|row| ColIndex::from_usize(structural + row))
                .collect(),
        );
        for &column in basis.as_slice() {
            info.update_to_basic_status(column);
        }
        let identity = sparse_matrix(
            &(0..rows)
                .map(|row| {
                    (0..rows)
                        .map(|column| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect::<Vec<Vec<_>>>(),
        );
        let factorization = BasisRepresentation::new(identity, 0.1, 20).unwrap();
        let mut reduced = ReducedCosts::new(&compact, &objective, &basis, &info, &factorization, 1);
        let rust_reduced = reduced.reduced_costs().unwrap().to_vec();
        let rust_dual = reduced.dual_values().unwrap().to_vec();
        let mut values = VariableValues::new(
            &GlopParameters::default(),
            &compact,
            &basis,
            &info,
            &factorization,
        );
        values.reset_all_non_basic_variable_values(&DenseRow::new());
        values
            .recompute_basic_variable_values(&mut DynamicMaximum::new(1))
            .unwrap();
        let rust_values = values.dense_row().as_slice().to_vec();
        let rust_scalars = [
            values.compute_maximum_primal_residual(),
            values.compute_maximum_primal_infeasibility(),
            values.compute_sum_of_primal_infeasibilities(),
        ];

        let structural_entries = matrix
            .num_entries()
            .value()
            .saturating_sub(i64::try_from(rows).unwrap());
        let mut input = format!("{rows} {structural} {structural_entries}\n");
        for column in 0..structural {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                writeln!(
                    input,
                    "{} {column} {:.17}",
                    entry.index().value(),
                    entry.coefficient()
                )
                .unwrap();
            }
        }
        for column in 0..columns {
            writeln!(
                input,
                "{:.17} {:.17} {:.17}",
                objective[ColIndex::from_usize(column)],
                lower_values[column],
                upper_values[column]
            )
            .unwrap();
        }
        let mut child = Command::new(&adapter)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        let output = String::from_utf8(output.stdout).unwrap();
        let lines = output.lines().collect::<Vec<_>>();
        let parse = |line: &str| {
            line.split_whitespace()
                .skip(1)
                .map(|token| token.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        for (rust, native) in [
            (&rust_reduced, parse(lines[0])),
            (&rust_dual, parse(lines[1])),
            (&rust_values, parse(lines[2])),
        ] {
            assert_eq!(rust.len(), native.len());
            for (&left, right) in rust.iter().zip(native) {
                assert!(
                    (left - right).abs() <= 2e-14 * (1.0 + left.abs()),
                    "case {case}: {left} != {right}"
                );
            }
        }
        for (&left, line) in rust_scalars.iter().zip(&lines[3..]) {
            let right = parse(line)[0];
            assert!(
                (left - right).abs() <= 2e-14 * (1.0 + left.abs()),
                "case {case}: {left} != {right}"
            );
        }
    }
}

#[test]
fn dual_ratio_tests_agree_with_native_glop() {
    let adapter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/native/entering_variable_reference_adapter");
    if !adapter.exists() {
        eprintln!("skipping native entering-variable differential test: adapter not built");
        return;
    }
    let mut generator = Generator(0xa076_1d64_78bd_642f);
    for case in 0..100 {
        let rows = 2 + generator.usize(6);
        let structural = 2 + generator.usize(10);
        let columns = structural + rows;
        let leaving_row = generator.usize(rows);
        let cost_variation = if case % 2 == 0 { 2.5 } else { -3.25 };
        let mut dense = vec![vec![0.0; columns]; rows];
        for column in 0..structural {
            for row_values in &mut dense {
                if generator.usize(3) != 0 {
                    row_values[column] =
                        f64::from(i32::try_from(generator.usize(21)).unwrap() - 10) / 9.0;
                }
            }
        }
        for row in 0..rows {
            dense[row][structural + row] = 1.0;
        }
        let matrix = sparse_matrix(&dense);
        let compact = CompactSparseMatrix::from_sparse(&matrix);
        let objective = DenseRow::from_vec(
            (0..columns)
                .map(|column| {
                    if column < structural {
                        (column + 1) as f64 * 0.137 - 0.71
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let mut lower = vec![f64::NEG_INFINITY; columns];
        let mut upper = vec![f64::INFINITY; columns];
        for column in 0..structural {
            match (case + column) % 3 {
                0 => lower[column] = -1.0 - column as f64 / 10.0,
                1 => upper[column] = 1.0 + column as f64 / 11.0,
                _ => {}
            }
        }
        let mut info = VariablesInfo::new(&matrix);
        info.load_bounds_and_return_true_if_unchanged(&lower, &upper);
        info.initialize_to_default_status();
        let basis = RowToColMapping::from_vec(
            (0..rows)
                .map(|row| ColIndex::from_usize(structural + row))
                .collect(),
        );
        for &column in basis.as_slice() {
            info.update_to_basic_status(column);
        }
        let identity = sparse_matrix(
            &(0..rows)
                .map(|row| {
                    (0..rows)
                        .map(|column| if row == column { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect::<Vec<Vec<_>>>(),
        );
        let factorization = BasisRepresentation::new(identity, 0.1, 20).unwrap();
        let mut update = UpdateRow::new(&matrix);
        update
            .compute_update_row(&factorization, &matrix, info.relevance(), leaving_row)
            .unwrap();
        let mut reduced = ReducedCosts::new(&compact, &objective, &basis, &info, &factorization, 1);
        let mut entering = EnteringVariable::new(1);
        let mut flips = Vec::new();
        let phase_two = entering
            .dual_choose_entering_column(
                true,
                &update,
                cost_variation,
                &info,
                &mut reduced,
                &mut flips,
            )
            .unwrap();
        let phase_one = entering
            .dual_phase_one_choose_entering_column(
                true,
                &update,
                cost_variation,
                &info,
                &mut reduced,
            )
            .unwrap();

        let structural_entries = matrix
            .num_entries()
            .value()
            .saturating_sub(i64::try_from(rows).unwrap());
        let mut input = format!(
            "{rows} {structural} {structural_entries} {leaving_row} {cost_variation:.17}\n"
        );
        for column in 0..structural {
            for entry in matrix.column(ColIndex::from_usize(column)) {
                writeln!(
                    input,
                    "{} {column} {:.17}",
                    entry.index().value(),
                    entry.coefficient()
                )
                .unwrap();
            }
        }
        let token = |value: f64| {
            if value == f64::INFINITY {
                "inf".to_owned()
            } else if value == f64::NEG_INFINITY {
                "-inf".to_owned()
            } else {
                format!("{value:.17}")
            }
        };
        for column in 0..columns {
            writeln!(
                input,
                "{} {} {}",
                token(objective[ColIndex::from_usize(column)]),
                token(lower[column]),
                token(upper[column])
            )
            .unwrap();
        }
        let mut child = Command::new(&adapter)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "case {case}");
        let output = String::from_utf8(output.stdout).unwrap();
        let lines = output.lines().collect::<Vec<_>>();
        let native_phase_two = lines[1]
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let native_flips = lines[2]
            .split_whitespace()
            .skip(1)
            .map(|value| value.parse::<i32>().unwrap())
            .collect::<Vec<_>>();
        let native_phase_one = lines[3]
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        assert_eq!(
            phase_two.map_or(-1, ColIndex::value),
            native_phase_two,
            "case {case}"
        );
        assert_eq!(
            flips
                .iter()
                .map(|column| column.value())
                .collect::<Vec<_>>(),
            native_flips,
            "case {case}"
        );
        assert_eq!(
            phase_one.map_or(-1, ColIndex::value),
            native_phase_one,
            "case {case}"
        );
    }
}

#[test]
fn dynamic_pricing_agrees_with_native_glop() {
    let adapter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/native/pricing_reference_adapter");
    if !adapter.exists() {
        eprintln!("skipping native pricing differential test: adapter not built");
        return;
    }
    let cases = 500;
    let mut input = format!("{cases}\n");
    let mut expected = String::new();
    let mut generator = Generator(0xe703_7ed1_a0b4_28db);
    for case in 0..cases {
        let size = 1 + generator.usize(150);
        let mut operations = Vec::new();
        let mut maximum = DynamicMaximum::new(1);
        maximum.clear_and_resize(size);
        for position in 0..size {
            let value = (position * 1009 + case) as f64;
            operations.push(format!("0 {position} {value:.17}"));
            maximum.add_or_update(position, value);
        }
        operations.push("4".to_owned());
        let mut answers = vec![maximum.get_maximum().map_or(-1, |value| value as i32)];
        for update in 0..20 {
            let position = generator.usize(size);
            if update % 5 == 0 {
                operations.push(format!("1 {position}"));
                maximum.remove(position);
            } else {
                let value = -((update * 997 + position) as f64);
                operations.push(format!("0 {position} {value:.17}"));
                maximum.add_or_update(position, value);
            }
        }
        operations.push("4".to_owned());
        answers.push(maximum.get_maximum().map_or(-1, |value| value as i32));
        operations.push("2".to_owned());
        maximum.start_dense_updates();
        for position in 0..size {
            let value = -((position * 1013 + case) as f64);
            operations.push(format!("3 {position} {value:.17}"));
            maximum.dense_add_or_update(position, value);
        }
        operations.push("4".to_owned());
        answers.push(maximum.get_maximum().map_or(-1, |value| value as i32));
        writeln!(input, "{size} {}", operations.len()).unwrap();
        for operation in operations {
            writeln!(input, "{operation}").unwrap();
        }
        expected.push_str("case");
        for answer in answers {
            write!(expected, " {answer}").unwrap();
        }
        expected.push('\n');
    }
    let mut child = Command::new(&adapter)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
}

#[test]
fn controlled_multi_pivot_traces_agree_with_native_glop() {
    let adapter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/native/primal_pivot_trace_reference_adapter");
    if !adapter.exists() {
        eprintln!("skipping native controlled-pivot trace: adapter not built");
        return;
    }
    for case in 0..20 {
        let structural_values = vec![
            vec![1.011 + f64::from(case) / 101.0, 2.023, -0.971, 0.537],
            vec![2.041, -1.053, 1.067 + f64::from(case) / 127.0, 1.579],
            vec![-0.917, 1.089, 2.097, 1.113 + f64::from(case) / 149.0],
        ];
        let rows = structural_values.len();
        let structural = structural_values[0].len();
        let mut full_values = structural_values.clone();
        for (row, values) in full_values.iter_mut().enumerate() {
            values.extend((0..rows).map(|column| if row == column { 1.0 } else { 0.0 }));
        }
        let full = sparse_matrix(&full_values);
        let compact = CompactSparseMatrix::from_sparse(&full);
        let objective = DenseRow::from_vec(vec![-5.0, -4.1, -2.9, -1.7, 0.0, 0.0, 0.0]);
        let lower = vec![0.0, 0.0, 0.0, 0.0, -5.17, -4.31, -6.43];
        let upper = vec![8.19, 7.23, 6.29, 5.37, 4.41, 5.47, 3.53];
        let mut info = VariablesInfo::new(&full);
        info.load_bounds_and_return_true_if_unchanged(&lower, &upper);
        info.initialize_to_default_status();
        let mut basis =
            RowToColMapping::from_vec(vec![ColIndex::new(4), ColIndex::new(5), ColIndex::new(6)]);
        for &column in basis.as_slice() {
            info.update_to_basic_status(column);
        }
        let parameters = GlopParameters {
            use_scaling: false,
            initial_basis: glop::parameters::InitialBasisHeuristic::None,
            exploit_singleton_column_in_initial_basis: false,
            optimization_rule: glop::parameters::PricingRule::Dantzig,
            ..GlopParameters::default()
        };
        let mut iterations = 0_i64;
        for _ in 0..3 {
            let basis_matrix = basis_matrix(&full, basis.as_slice());
            let factorization =
                BasisRepresentation::new_with_parameters(basis_matrix, &parameters).unwrap();
            let mut values =
                VariableValues::new(&parameters, &compact, &basis, &info, &factorization);
            values.reset_all_non_basic_variable_values(&DenseRow::new());
            values
                .recompute_basic_variable_values(&mut DynamicMaximum::new(1))
                .unwrap();
            let current_values = values.dense_row().as_slice().to_vec();
            let mut reduced =
                ReducedCosts::new(&compact, &objective, &basis, &info, &factorization, 1);
            reduced.set_parameters(&parameters);
            let reduced_values = reduced.reduced_costs().unwrap().to_vec();
            let mut best = None;
            let mut best_price = f64::NEG_INFINITY;
            for column in info.relevance().iter_ones() {
                let rc = reduced_values[column.to_usize()];
                let valid = (info.can_increase().contains(column)
                    && rc < -reduced.dual_feasibility_tolerance())
                    || (info.can_decrease().contains(column)
                        && rc > reduced.dual_feasibility_tolerance());
                if !valid {
                    continue;
                }
                let norm = full
                    .column(column)
                    .into_iter()
                    .map(|entry| entry.coefficient() * entry.coefficient())
                    .sum::<f64>();
                let price = rc * rc / norm;
                if price > best_price {
                    best_price = price;
                    best = Some(column);
                }
            }
            let Some(entering) = best else { break };
            let mut rhs = vec![0.0; rows];
            for entry in full.column(entering) {
                rhs[entry.index().to_usize()] = entry.coefficient();
            }
            let dense_direction = factorization.solve(&rhs).unwrap();
            let mut direction = ScatteredColumn::new(RowIndex::from_usize(rows));
            for (row, &value) in dense_direction.iter().enumerate() {
                if value != 0.0 {
                    direction.set(RowIndex::from_usize(row), value);
                }
            }
            let direction_norm = dense_direction
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            match choose_leaving_variable_row(
                entering,
                reduced_values[entering.to_usize()],
                &direction,
                direction_norm,
                &current_values,
                &lower,
                &upper,
                &basis,
                factorization.is_refactorized(),
                &parameters,
            ) {
                LeavingChoice::BoundFlip { .. } => {
                    let status = match info.variable_statuses()[entering] {
                        VariableStatus::AtLowerBound => VariableStatus::AtUpperBound,
                        VariableStatus::AtUpperBound => VariableStatus::AtLowerBound,
                        status => panic!("non-boxed bound flip for {status:?}"),
                    };
                    info.update_to_nonbasic_status(entering, status);
                }
                LeavingChoice::Pivot {
                    row, target_bound, ..
                } => {
                    let leaving = basis[row];
                    let leaving_status = if target_bound == lower[leaving.to_usize()] {
                        VariableStatus::AtLowerBound
                    } else {
                        VariableStatus::AtUpperBound
                    };
                    info.update_to_nonbasic_status(leaving, leaving_status);
                    info.update_to_basic_status(entering);
                    basis[row] = entering;
                }
                LeavingChoice::Refactorize => {
                    panic!("fresh factorization requested refactorization")
                }
            }
            iterations += 1;
        }
        let basis_matrix = basis_matrix(&full, basis.as_slice());
        let factorization =
            BasisRepresentation::new_with_parameters(basis_matrix, &parameters).unwrap();
        let mut final_values =
            VariableValues::new(&parameters, &compact, &basis, &info, &factorization);
        final_values.reset_all_non_basic_variable_values(&DenseRow::new());
        final_values
            .recompute_basic_variable_values(&mut DynamicMaximum::new(1))
            .unwrap();
        let mut final_reduced =
            ReducedCosts::new(&compact, &objective, &basis, &info, &factorization, 1);
        final_reduced.set_parameters(&parameters);
        let rust_reduced = final_reduced.reduced_costs().unwrap().to_vec();

        let mut input = format!(
            "{rows} {structural} {} 3\n",
            full.num_entries().value() - rows as i64
        );
        for column in 0..structural {
            for entry in full.column(ColIndex::from_usize(column)) {
                writeln!(
                    input,
                    "{} {column} {:.17}",
                    entry.index().value(),
                    entry.coefficient()
                )
                .unwrap();
            }
        }
        for column in 0..structural {
            writeln!(
                input,
                "{:.17} {:.17} {:.17}",
                objective[ColIndex::from_usize(column)],
                lower[column],
                upper[column]
            )
            .unwrap();
        }
        for row in 0..rows {
            writeln!(
                input,
                "{} {}",
                -upper[structural + row],
                -lower[structural + row]
            )
            .unwrap();
        }
        let output = run_adapter(&adapter, &input);
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], format!("iterations {iterations}"), "case {case}");
        let mut native_basis = lines[1]
            .split_whitespace()
            .skip(1)
            .map(|value| value.parse::<i32>().unwrap())
            .collect::<Vec<_>>();
        let mut rust_basis = basis
            .as_slice()
            .iter()
            .map(|column| column.value())
            .collect::<Vec<_>>();
        native_basis.sort_unstable();
        rust_basis.sort_unstable();
        assert_eq!(native_basis, rust_basis, "case {case}");
        let native_values = parse_values(lines[2]);
        let native_reduced = parse_values(lines[3]);
        for (&left, right) in final_values
            .dense_row()
            .as_slice()
            .iter()
            .zip(native_values)
        {
            assert!(
                (left - right).abs() <= 1e-12 * (1.0 + left.abs()),
                "case {case}: {left} != {right}"
            );
        }
        for (&left, right) in rust_reduced.iter().zip(native_reduced) {
            assert!(
                (left - right).abs() <= 1e-12 * (1.0 + left.abs()),
                "case {case}: {left} != {right}"
            );
        }
    }
}

fn basis_matrix(matrix: &SparseMatrix, basis: &[ColIndex]) -> SparseMatrix {
    let mut result = SparseMatrix::new();
    result.populate_from_zero(matrix.num_rows(), ColIndex::from_usize(basis.len()));
    for (destination, &source) in basis.iter().enumerate() {
        for entry in matrix.column(source) {
            result
                .mutable_column(ColIndex::from_usize(destination))
                .add_entry(entry.index(), entry.coefficient());
        }
    }
    result.clean_up();
    result
}

fn run_adapter(adapter: &PathBuf, input: &str) -> String {
    let mut child = Command::new(adapter)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

fn parse_values(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .skip(1)
        .map(|token| token.parse().unwrap())
        .collect()
}
