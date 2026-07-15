#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::iop::wire::Wire;

/// Disjoint Set Forest data-structure following <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
pub struct Forest {
    /// A map of parent pointers, stored as indices.
    pub(crate) parents: Vec<usize>,

    num_wires: usize,
    num_routed_wires: usize,
    degree: usize,
}

impl Forest {
    pub fn new(
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
    ) -> Self {
        let capacity = num_wires * degree + num_virtual_targets;
        Self {
            parents: (0..capacity).collect(),
            num_wires,
            num_routed_wires,
            degree,
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    /// Add a new partition with a single member.
    #[allow(unused)]
    pub fn add(&mut self, t: Target) {
        let index = self.parents.len();
        debug_assert_eq!(self.target_index(t), index);
        self.parents.push(index);
    }

    /// Path compression method, see <https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives>.
    pub fn find(&mut self, mut x_index: usize) -> usize {
        // Note: We avoid recursion here since the chains can be long, causing stack overflows.

        // First, find the representative of the set containing `x_index`.
        let mut representative = x_index;
        while self.parents[representative] != representative {
            representative = self.parents[representative];
        }

        // Then, update each node in this chain to point directly to the representative.
        while self.parents[x_index] != x_index {
            let old_parent = self.parents[x_index];
            self.parents[x_index] = representative;
            x_index = old_parent;
        }

        representative
    }

    /// Merge two sets.
    pub fn merge(&mut self, tx: Target, ty: Target) {
        let x_index = self.find(self.target_index(tx));
        let y_index = self.find(self.target_index(ty));

        if x_index == y_index {
            return;
        }

        self.parents[y_index] = x_index;
    }

    /// Compress all paths. After calling this, every `parent` value will point to the node's
    /// representative.
    pub(crate) fn compress_paths(&mut self) {
        for i in 0..self.parents.len() {
            self.find(i);
        }
    }

    /// Assumes `compress_paths` has already been called.
    pub fn get_sigma_polys<F: Field>(
        &mut self,
        k_is: &[F],
        subgroup: &[F],
    ) -> Vec<PolynomialValues<F>> {
        let degree = self.degree;
        assert!(self.num_routed_wires * degree <= u32::MAX as usize);
        assert!(self.parents.len() < u32::MAX as usize);
        let mut sigma: Vec<_> = (0..(self.num_routed_wires * degree) as u32).collect();
        let mut last_in_cycle = vec![u32::MAX; self.parents.len()];
        for row in 0..degree {
            for column in 0..self.num_routed_wires {
                let parent = self.parents[self.target_index(Target::Wire(Wire { row, column }))];
                let wire_index = column * degree + row;
                let swap_with = last_in_cycle[parent];
                if swap_with != u32::MAX {
                    sigma.swap(wire_index, swap_with as usize);
                }
                last_in_cycle[parent] = wire_index as u32;
            }
        }

        // Step 6: Compute polynomial values from sigma
        let result = sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .par_iter()
                    .map(|&x| k_is[x as usize / degree] * subgroup[x as usize % degree])
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect();

        rayon::spawn(move || {
            drop(last_in_cycle);
            drop(sigma);
        }); // Frees in parallel to the main job

        result
    }
}
