// oneDNN kernel.db from third_party/onednn-src (see prepare_onednn_deps.bat).

#include "gemmstone/kernel_catalog.hpp"

namespace gemmstone {
namespace case5_catalog_detail {
#define _CATALOG_ gemm_catalog
#include "selector/db/kernel.db"
#undef _CATALOG_
} // namespace case5_catalog_detail

const kcatalog::Catalog &case5_catalog() {
    static const kcatalog::Catalog catalog(case5_catalog_detail::gemm_catalog);
    return catalog;
}

} // namespace gemmstone
