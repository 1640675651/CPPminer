/*******************************************************************************
 * Case 5 gemmstone generator TU: all generator pieces + explicit HW instancing.
 * The gemmstone pieces (with the Case 5 modifications) come from the case5 oneDNN fork and are
 * vendored into third_party/gemmstone by prepare_onednn_deps.bat (also invoked from CMake).
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
template class Generator<HW::Xe2>;      // Battlemage / Lunar Lake (64-byte GRFs, 16-channel dpas, LSC only)

GEMMSTONE_NAMESPACE_END
