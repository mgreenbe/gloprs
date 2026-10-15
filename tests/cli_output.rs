//! Compare CLI output byte-for-byte with fixtures captured from pinned native GLOP.
use std::path::Path;
use std::process::Command;

fn run(directory: &Path, arguments: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_gloprs"))
        .current_dir(directory)
        .args(arguments)
        .output()
        .expect("run CLI");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout).expect("UTF-8 output")
}

#[test]
fn formatted_solve_and_inspect_match_native() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli");
    let expected = include_str!("fixtures/cli/model.solve.txt");
    assert_eq!(run(&directory, &["model.mps"]), expected);
    assert_eq!(
        run(&directory, &["solve", "--default", "model.mps"]),
        expected
    );
    assert_eq!(
        run(&directory, &["inspect", "model.mps"]),
        include_str!("fixtures/cli/model.inspect.txt")
    );
}

#[test]
fn machine_and_tsv_output_remain_explicit() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli");
    let machine = run(&directory, &["--machine", "model.mps"]);
    assert_eq!(machine.lines().count(), 1);
    assert!(machine.starts_with("status=OPTIMAL objective="));
    assert!(machine.contains(" primal_infeasibility="));
    let tsv = run(&directory, &["inspect", "--tsv", "model.mps"]);
    assert_eq!(tsv.lines().count(), 1);
    assert_eq!(tsv.trim_end().split('\t').count(), 5);
}

#[test]
#[ignore = "requires shared ../datasets/netlib/mps/afiro.mps"]
fn afiro_readme_output_matches_native() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../datasets/netlib/mps");
    assert_eq!(
        run(&directory, &["afiro.mps"]),
        include_str!("fixtures/cli/afiro.solve.txt")
    );
    assert_eq!(
        run(&directory, &["inspect", "afiro.mps"]),
        include_str!("fixtures/cli/afiro.inspect.txt")
    );
}

#[test]
fn readme_examples_contain_native_captured_output() {
    let readme = include_str!("../README.md");
    for expected in [
        include_str!("fixtures/cli/afiro.solve.txt"),
        include_str!("fixtures/cli/afiro.inspect.txt"),
    ] {
        assert!(readme.contains(&format!("```text\n{expected}```")));
    }
}
