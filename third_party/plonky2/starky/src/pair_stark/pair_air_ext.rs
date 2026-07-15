use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::RecursiveConstraintConsumer;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::{PAIR_COLUMNS, PAIR_PUBLIC_INPUTS};

pub fn eval_ext_circuit<F, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    vars: &StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, PAIR_COLUMNS, PAIR_PUBLIC_INPUTS>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) where
    F: RichField + Extendable<D>,
{
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();
    let one = builder.one_extension();

    // First row constraints
    let check_init_values_1 = builder.sub_extension(local_values[2], public_inputs[0]);
    let check_init_values_2 = builder.sub_extension(local_values[3], public_inputs[1]);
    yield_constr.constraint_first_row(builder, check_init_values_1);
    yield_constr.constraint_first_row(builder, check_init_values_2);

    // Transition constraints
    let copy_constraint_to_next_row = builder.sub_extension(next_values[2], local_values[3]);
    yield_constr.constraint_transition(builder, copy_constraint_to_next_row);

    // main operation constraint
    let add_sum = builder.add_many_extension([local_values[2], local_values[3], next_values[1]]);
    let add_case = builder.sub_extension(next_values[3], add_sum);

    let mul_val = builder.mul_extension(local_values[2], local_values[3]);
    let mul_sum = builder.add_extension(mul_val, next_values[1]);
    let mul_case = builder.sub_extension(next_values[3], mul_sum);

    let s = next_values[0];
    let one_minus_s = builder.sub_extension(one, s);
    let recurrence_add_case = builder.mul_extension(one_minus_s, add_case);
    let recurrence_mul_case = builder.mul_extension(s, mul_case);
    let recurrence = builder.add_extension(recurrence_add_case, recurrence_mul_case);
    yield_constr.constraint_transition(builder, recurrence);

    let check_output_value = builder.sub_extension(local_values[3], public_inputs[2]);
    yield_constr.constraint_last_row(builder, check_output_value);
}
