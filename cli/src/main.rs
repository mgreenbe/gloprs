use std::env;
use std::process::ExitCode;

use glop::lp_solver::LPSolver;
use lp_data::mps_reader::parse_mps_file;

fn usage(program: &std::ffi::OsStr) -> ExitCode {
    eprintln!(
        "usage: {} inspect [--tsv] MODEL.mps\n       {} solve [--dual] MODEL.mps",
        program.to_string_lossy(),
        program.to_string_lossy()
    );
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(command) = arguments.next() else {
        return usage(&program);
    };
    let Some(mut path) = arguments.next() else {
        return usage(&program);
    };
    let tab_separated = command == "inspect" && path == "--tsv";
    let dual_simplex = command == "solve" && path == "--dual";
    if tab_separated || dual_simplex {
        let Some(actual_path) = arguments.next() else {
            return usage(&program);
        };
        path = actual_path;
    }
    if arguments.next().is_some() {
        return usage(&program);
    }
    let model = match parse_mps_file(path) {
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
            println!("{}", model.summary());
        }
        return ExitCode::SUCCESS;
    }
    if command == "solve" {
        let mut solver = LPSolver::new();
        solver.parameters_mut().use_preprocessing = false;
        solver.parameters_mut().use_scaling = false;
        solver.parameters_mut().use_dual_simplex = dual_simplex;
        let status = solver.solve(&model);
        println!(
            "status={} objective={:.17e} iterations={} primal_infeasibility={:.17e} dual_infeasibility={:.17e}",
            status,
            solver.objective_value(),
            solver.number_of_simplex_iterations(),
            solver.maximum_primal_infeasibility(),
            solver.maximum_dual_infeasibility()
        );
        return ExitCode::SUCCESS;
    }
    usage(&program)
}
