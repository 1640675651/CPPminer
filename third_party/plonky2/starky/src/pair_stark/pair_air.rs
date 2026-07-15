use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use starky::constraint_consumer::ConstraintConsumer;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::{PAIR_COLUMNS, PAIR_PUBLIC_INPUTS};

pub fn air_eval_packed<F, FE, P, const D: usize, const D2: usize>(
    vars: &StarkFrame<P, P::Scalar, PAIR_COLUMNS, PAIR_PUBLIC_INPUTS>,
    yield_constr: &mut ConstraintConsumer<P>,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();

    yield_constr.constraint_first_row(local_values[2] - public_inputs[0]);
    yield_constr.constraint_first_row(local_values[3] - public_inputs[1]);

    yield_constr.constraint_transition(next_values[2] - local_values[3]);

    // main operation constraint
    let add_case = next_values[3] - (local_values[2] + local_values[3] + next_values[1]);
    let mul_case = next_values[3] - (local_values[2] * local_values[3] + next_values[1]);
    let s = next_values[0];

    yield_constr.constraint_transition((P::ONES - s) * add_case + s * mul_case);

    yield_constr.constraint_last_row(local_values[3] - public_inputs[2]);
}
