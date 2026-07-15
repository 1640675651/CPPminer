use alloc::vec::Vec;

use plonky2::hash::hash_types::RichField;

use crate::{PAIR_COLUMNS, PAIR_PUBLIC_INPUTS};

pub fn generate_trace_rows<F: RichField>(
    num_rows: usize,
) -> (Vec<[F; PAIR_COLUMNS]>, [F; PAIR_PUBLIC_INPUTS]) {
    let mut trace_rows: Vec<[F; PAIR_COLUMNS]> = Vec::with_capacity(num_rows);

    let mut row = [F::ZERO; PAIR_COLUMNS];
    row[3] = F::ONE;
    trace_rows.push(row);

    for row_iter in 1..num_rows {
        let prev_row = trace_rows[row_iter - 1];
        let mut row = [F::ZERO; PAIR_COLUMNS];

        row[0] = F::from_canonical_u64((row_iter % 2) as u64);
        row[1] = F::from_canonical_u64(row_iter as u64);
        row[2] = prev_row[3];
        if row_iter.is_multiple_of(2) {
            row[3] = prev_row[2] + prev_row[3] + row[1];
        } else {
            row[3] = prev_row[2] * prev_row[3] + row[1];
        }
        trace_rows.push(row);
    }

    let public_inputs = [
        trace_rows[0][2],
        trace_rows[0][3],
        trace_rows[num_rows - 1][3],
    ];
    (trace_rows, public_inputs)
}
