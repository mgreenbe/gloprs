use std::env;
use std::process::ExitCode;

use gloprs::{LPSolver, LinearProgram, ProblemStatus, parse_mps_file};

fn usage(program: &std::ffi::OsStr) -> ExitCode {
    eprintln!(
        "usage: {} [--raw] [--dual] [--machine] MODEL.mps\n       {} inspect [--tsv] MODEL.mps",
        program.to_string_lossy(),
        program.to_string_lossy()
    );
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(mut command) = arguments.next() else {
        return usage(&program);
    };
    let mut path = if command == "solve" || command == "inspect" {
        let Some(path) = arguments.next() else {
            return usage(&program);
        };
        path
    } else {
        std::mem::replace(&mut command, "solve".into())
    };
    let mut raw = false;
    let mut dual_simplex = false;
    let mut machine = false;
    while command == "solve"
        && matches!(
            path.to_str(),
            Some("--raw" | "--dual" | "--default" | "--machine")
        )
    {
        if path == "--raw" {
            raw = true;
        } else if path == "--dual" {
            dual_simplex = true;
        } else if path == "--machine" {
            machine = true;
        }
        let Some(actual_path) = arguments.next() else {
            return usage(&program);
        };
        path = actual_path;
    }
    let tab_separated = command == "inspect" && path == "--tsv";
    if tab_separated {
        let Some(actual_path) = arguments.next() else {
            return usage(&program);
        };
        path = actual_path;
    }
    if arguments.next().is_some() {
        return usage(&program);
    }
    let model = match parse_mps_file(&path) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if command == "inspect" {
        if tab_separated {
            let summary = model.summary();
            println!(
                "{}\t{}\t{}\t{}\t{:016x}",
                summary.name,
                summary.rows,
                summary.columns,
                summary.nonzeros,
                model.data_fingerprint()
            );
        } else {
            print!("{}", model.pretty_problem_stats());
        }
        return ExitCode::SUCCESS;
    }
    if command == "solve" {
        let mut solver = LPSolver::new();
        if raw {
            solver.parameters_mut().use_preprocessing = false;
            solver.parameters_mut().use_scaling = false;
        }
        solver.parameters_mut().use_dual_simplex = dual_simplex;
        let status = solver.solve_status(&model);
        if machine {
            println!(
                "status={} objective={:.17e} iterations={} primal_infeasibility={:.17e} dual_infeasibility={:.17e}",
                status,
                solver.objective_value(),
                solver.number_of_simplex_iterations(),
                solver.maximum_primal_infeasibility(),
                solver.maximum_dual_infeasibility()
            );
        } else {
            print_solution(&model, &solver, status, &path);
        }
        if let Some(error) = solver.last_error() {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }
    usage(&program)
}

// Match upstream printf("%.15e"), including signed, two-digit exponents.
fn scientific(value: f64) -> String {
    let formatted = format!("{value:.15e}");
    let Some((mantissa, exponent)) = formatted.split_once('e') else {
        return formatted;
    };
    let exponent: i32 = exponent.parse().expect("formatted floating-point exponent");
    format!("{mantissa}e{exponent:+03}")
}

fn print_solution(
    model: &LinearProgram,
    solver: &LPSolver,
    status: ProblemStatus,
    path: &std::ffi::OsStr,
) {
    let summary = model.summary();
    println!("{:<12}: '{}'", "File", path.to_string_lossy());
    println!(
        "{:<12}: {} x {}",
        "Dimension", summary.rows, summary.columns
    );
    println!("{:<12}: {status}", "Status");
    println!(
        "{:<12}: {}",
        "Objective",
        scientific(solver.objective_value())
    );
    println!(
        "{:<12}: {}",
        "Iterations",
        solver.number_of_simplex_iterations()
    );
    println!(
        "{:<12}: {}",
        "Primal error",
        scientific(solver.maximum_primal_infeasibility())
    );
    println!(
        "{:<12}: {}",
        "Dual error",
        scientific(solver.maximum_dual_infeasibility())
    );
}
