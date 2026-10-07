use std::env;
use std::process::ExitCode;

use lp_data::mps_reader::parse_mps_file;

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(command) = arguments.next() else {
        eprintln!(
            "usage: {} inspect [--tsv] MODEL.mps",
            program.to_string_lossy()
        );
        return ExitCode::FAILURE;
    };
    let Some(mut path) = arguments.next() else {
        eprintln!(
            "usage: {} inspect [--tsv] MODEL.mps",
            program.to_string_lossy()
        );
        return ExitCode::FAILURE;
    };
    let tab_separated = path == "--tsv";
    if tab_separated {
        let Some(actual_path) = arguments.next() else {
            eprintln!(
                "usage: {} inspect [--tsv] MODEL.mps",
                program.to_string_lossy()
            );
            return ExitCode::FAILURE;
        };
        path = actual_path;
    }
    if command != "inspect" || arguments.next().is_some() {
        eprintln!(
            "usage: {} inspect [--tsv] MODEL.mps",
            program.to_string_lossy()
        );
        return ExitCode::FAILURE;
    }
    match parse_mps_file(path) {
        Ok(model) => {
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
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
