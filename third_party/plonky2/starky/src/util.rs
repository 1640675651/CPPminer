//! Utility module providing some helper functions.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2_maybe_rayon::*;

/// A helper function to transpose a row-wise trace and put it in the format that `prove` expects.
pub fn trace_rows_to_poly_values<F: Field, const COLUMNS: usize>(
    trace_rows: Vec<[F; COLUMNS]>,
) -> Vec<PolynomialValues<F>> {
    // transposing
    let trace_columns = (0..COLUMNS)
        .into_par_iter()
        .map(|i| PolynomialValues::new(trace_rows.iter().map(|row| row[i]).collect()))
        .collect::<Vec<_>>();
    trace_columns
}
