/*******************************************************************************
 * Case 5 gemmstone generator TU: all generator pieces + explicit HW instancing.
 * Lives in-repo (not third_party) so Gen12LP + XeHPG symbols always link even
 * when prepare_case5_deps skips re-applying patches to vendored gemmstone.
 ******************************************************************************/

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pieces/address_setup.cxx"
#include "pieces/asm_helpers.cxx"
#include "pieces/atomic_fusions.cxx"
#include "pieces/c_update.cxx"
#include "pieces/common.cxx"
#include "pieces/copy.cxx"
#include "pieces/driver_info.cxx"
#include "pieces/emulation.cxx"
#include "pieces/gemm_microkernel.cxx"
#include "pieces/gemm_setup.cxx"
#include "pieces/gemm.cxx"
#include "pieces/case5_tile_xor.cxx"
#include "pieces/case5_blake3.cxx"
#include "pieces/k_loop_setup.cxx"
#include "pieces/k_loop.cxx"
#include "pieces/layout_setup.cxx"
#include "pieces/l3_prefetch.cxx"
#include "pieces/masks.cxx"
#include "pieces/math_helpers.cxx"
#include "pieces/matrix_access.cxx"
#include "pieces/matrix_multiply.cxx"
#include "pieces/monolithic_k_loop_dpasw.cxx"
#include "pieces/post_ops.cxx"
#include "pieces/register_allocation.cxx"
#include "pieces/remask.cxx"
#include "pieces/row_column_sums.cxx"
#include "pieces/state_utils.cxx"
#include "pieces/stream_k.cxx"
#include "pieces/tlb_warmup.cxx"
#include "pieces/walk_orders.cxx"
#include "pieces/quantization.cxx"

GEMMSTONE_NAMESPACE_START

template <HW hw>
constexpr typename Generator<hw>::status_stream::Endl Generator<hw>::status_stream::endl;

#ifdef _MSC_VER
#pragma warning(disable : 4661)
#endif

template class Generator<HW::Gen12LP>;
template class Generator<HW::XeHPG>;

GEMMSTONE_NAMESPACE_END
