//! Linear-program models and sparse data structures used by GLOP.

#![forbid(unsafe_code)]

pub mod lp_data;
pub mod lp_data_utils;
pub mod lp_decomposer;
pub mod lp_parser;
pub mod lp_print_utils;
pub mod lp_types;
pub mod lp_utils;
pub mod matrix_scaler;
pub mod matrix_utils;
pub mod mps_reader;
pub mod permutation;
pub mod scattered_vector;
pub mod sol_reader;
pub mod sparse;
pub mod sparse_row;
pub mod sparse_vector;
pub mod triangular_matrix;
