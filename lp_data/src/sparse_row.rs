//! Row-specialized sparse vectors from upstream `sparse_row.h`.

use crate::lp_types::ColIndex;
use crate::sparse_vector::SparseVector;

pub type SparseRow = SparseVector<ColIndex>;
