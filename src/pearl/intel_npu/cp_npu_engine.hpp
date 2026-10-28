#pragma once
/*
 * OpenVINO side of the Intel NPU backend: the patched in-plugin compiler's switches, the model cache,
 * the compiled launch graph, its infer requests and NPU-visible host tensors.
 */

#include <string>
#include <vector>

#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>

#include "cp_npu_graph.hpp"

namespace cp_npu {

class Engine {
public:
    /* Compiles (or loads from the cache) the launch graph and creates `requests` infer requests.
     * kernel_dir holds tile_xor_acc.3720xx*.elf (and tile_jackpot.3720xx*.elf); cache_root gets one
     * subdirectory per kernel and compiler build. Returns 0, or -1 with a message in *err. */
    int init(const LaunchShape& shape, const std::string& kernel_dir, const std::string& cache_root,
             int requests, std::string* err, const ov::AnyMap& extra_config = {});
    void shutdown();
    bool ready() const { return m_ready; }

    const LaunchShape& shape() const { return m_shape; }
    const std::vector<int>& word_index() const { return m_word_index; }
    int requests() const { return int(m_requests.size()); }
    ov::InferRequest& request(int i) { return m_requests[size_t(i)]; }
    /* Binds request i's inputs: the A panels, and the B^T panels to every B input of the graph. */
    void set_inputs(int i, const ov::Tensor& a, const ov::Tensor& bt);
    /* fused_jackpot: the [chunk_m/16, 16] int32 key and bound rows, bound to every request at init;
     * write them while no launch is in flight. */
    ov::Tensor& jackpot_params() { return m_jackpot; }

    /* A host tensor for input `index` (0: A panels, 1: B^T panels) that the NPU reads in place
     * (the plugin's host allocation); a plain tensor if the plugin offers none. */
    ov::Tensor host_tensor(int index);
    bool zero_copy() const { return m_zero_copy; }

    std::string device_name();
    double compile_sec() const { return m_compile_sec; }
    const std::string& cache_dir() const { return m_cache_dir; }

private:
    ov::Core m_core;
    ov::CompiledModel m_compiled;
    std::vector<ov::InferRequest> m_requests;
    ov::Tensor m_jackpot;
    LaunchShape m_shape;
    std::vector<int> m_word_index;
    std::string m_cache_dir;
    double m_compile_sec = 0.0;
    bool m_ready = false;
    bool m_zero_copy = true;
};

/* Sets the patched compiler's environment switches (int32 DPU output, kernel directory) unless the
 * caller already set them. Must run before the first compile. */
void set_compiler_environment(const std::string& kernel_dir);

/* Kernel directory next to the executable (kernels/intel_npu), else the source tree's copy. */
std::string default_kernel_dir();
/* Model cache root next to the executable (npu_cache). */
std::string default_cache_root();

} // namespace cp_npu
