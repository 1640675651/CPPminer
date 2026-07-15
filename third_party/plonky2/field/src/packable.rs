use crate::packed::PackedField;

/// Points us to the default packing for a particular field. There may be multiple choices of
/// PackedField for a particular Field (e.g. every Field is also a PackedField), but this is the
/// recommended one. The recommended packing varies by target_arch and target_feature.
pub trait Packable {
    type Packing: PackedField<Scalar = Self>;
}

// GoldilocksField packing type, selected at compile time based on available SIMD features.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512f",
        target_feature = "avx512vl"
    ))
))]
type GoldilocksPacking = crate::arch::x86_64::avx2_goldilocks_field::Avx2GoldilocksField;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl"
))]
type GoldilocksPacking = crate::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2",),
    all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512cd",
        target_feature = "avx512dq",
        target_feature = "avx512f",
        target_feature = "avx512vl",
    ),
)))]
type GoldilocksPacking = crate::goldilocks_field::GoldilocksField;

impl Packable for crate::goldilocks_field::GoldilocksField {
    type Packing = GoldilocksPacking;
}

impl Packable for crate::secp256k1_base::Secp256K1Base {
    type Packing = Self;
}

impl Packable for crate::secp256k1_scalar::Secp256K1Scalar {
    type Packing = Self;
}

impl<F: crate::extension::Extendable<2>> Packable
    for crate::extension::quadratic::QuadraticExtension<F>
{
    type Packing = Self;
}

impl<F: crate::extension::Extendable<4>> Packable
    for crate::extension::quartic::QuarticExtension<F>
{
    type Packing = Self;
}

impl<F: crate::extension::Extendable<5>> Packable
    for crate::extension::quintic::QuinticExtension<F>
{
    type Packing = Self;
}
