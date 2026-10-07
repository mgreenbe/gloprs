//! Linear-program models and sparse data structures used by GLOP.

#![forbid(unsafe_code)]

pub mod lp_data;
pub mod lp_types;
pub mod mps_reader;
pub mod permutation;
pub mod scattered_vector;
pub mod sparse;
pub mod sparse_row;
pub mod sparse_vector;
pub mod triangular_matrix;
