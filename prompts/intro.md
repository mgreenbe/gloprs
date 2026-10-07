# New project: `gloprs`

## Overview

My goal is to deeply understand the theory behind and the implementation of the GLOP (Google Linear Optimization Package) package within Google's OR-TOOLS suite.

I'm not fluent in C++, so I want you to port GLOP to Rust as a project called `gloprs`.

`gloprs` should, as much as possible, be a faithful, file-by-file, idiom-by-idiom, line-by-line translation of GLOP into Rust.
A notable exception to this guideline is that you should remove Google-specific infrastructure: bazel, protobuf and abseil, etc.

I want a working port that *prioritizes performance parity*.
For the purposes of regression testing, you should download the Netlib LP dataset (`.mps` files) into the `datasets` directory, adding a manifest consisting of relevant metadata.
You should also construct extracts of this dataset (10 smallest, 25 smallest, etc.) to use as smoke-testing datasets. As your port progresses, write testing harnesses as appropriate to validate the implementation to diagnose and fix performance regression.

You can (should?) fork the [OR-tools repo](https://github.com/google/or-tools) into `sparse/or-tools` so you can build it and directly compare performance between GLOP and `gloprs`.

The focus should be on GLOP, but you may need to include a few other parts of OR-TOOLS like
`or-tools/ortools/lp_data`. You can mirror the directory structure of `or-tools/ortools` inside `gloprs`, including crates like `gloprs/glop` (most of the action will be here) and `gloprs/lp_data` under the `gloprs` project within `sparse`.

## Instructions

Your first concrete tasks are:
- Create a file `gloprs/AGENTS.md` as a touchpoint for agents interacting with this project.
- Formulate a plan for accomplishing the goals described above in `gloprs/PLAN.md`.