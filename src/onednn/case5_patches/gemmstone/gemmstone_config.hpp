/*******************************************************************************
 * Case5 standalone gemmstone config (no oneDNN dependency).
 *******************************************************************************/
#ifndef GEMMSTONE_CONFIG_HPP
#define GEMMSTONE_CONFIG_HPP

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ngen_register_allocator.hpp"

#ifndef OPENCL_OUTPUT
#define OPENCL_OUTPUT
#endif
#ifndef GEMMSTONE_WITH_OPENCL_RUNTIME
#define GEMMSTONE_WITH_OPENCL_RUNTIME
#endif
// Keep assertions off: copy_plan.hpp includes <iostream> inside the gemmstone
// namespace when DUMP is enabled, which breaks MSVC if iostream wasn't seen first.
#define GEMMSTONE_ASSERTIONS 0
#define GEMMSTONE_ENABLE_COPY_PLAN_DUMP 0

#define GENERATOR_SUPER(hw) ngen::OpenCLCodeGenerator<hw>
#define GENERATOR_BASE(hw) ngen::OpenCLCodeGenerator<hw>
#define FORWARD(hw) NGEN_FORWARD_OPENCL(hw)
#define GENERATOR_DEBUGINFO \
    ngen::DebugConfig { __FILE__, __LINE__ }

namespace gemmstone {

inline int getEnv(const char *, int def) { return def; }

enum class GEMMVerbose { DebugInfo };
inline int getVerbose(GEMMVerbose) { return 0; }

template <typename... Args>
inline void verbosePrintf(const char *, Args...) {}

struct SerializationStream {
    template <typename... Args>
    void append(Args &&...) {}
};

// Defined in gemmstone/problem.hpp.
enum class BinaryOp;

struct PostOpEntry {
    bool is_binary() const { return false; }
    bool is_sum() const { return false; }
    void set_scale(float) {}
};

struct PostOps {
    using entry_t = PostOpEntry;
    bool empty() const { return ops_.empty(); }
    size_t len() const { return ops_.size(); }
    entry_t &operator[](size_t i) { return ops_[i]; }
    const entry_t &operator[](size_t i) const { return ops_[i]; }
    entry_t &back() { return ops_.back(); }
    const entry_t &back() const { return ops_.back(); }
    void pop_back() { ops_.pop_back(); }
    auto begin() { return ops_.begin(); }
    auto end() { return ops_.end(); }
    auto begin() const { return ops_.begin(); }
    auto end() const { return ops_.end(); }
    std::vector<entry_t> ops_;
};

struct PostOpsProblem {
    static const int maxPostOps = 32;

    PostOpsProblem() = default;
    PostOpsProblem(PostOps &&o) : ops(std::move(o)) {}
    PostOpsProblem &operator=(PostOps &&o) {
        ops = std::move(o);
        return *this;
    }
    PostOpsProblem &operator=(const PostOps &o) {
        ops = o;
        return *this;
    }

    template <ngen::HW hw>
    struct Injector {
        template <typename... Args>
        explicit Injector(Args &&...) {
            throw std::runtime_error("PostOps injector not available in Case5");
        }
        int preferred_scratch_regs() const { return 0; }
        int min_scratch_regs() const { return 0; }
        void set_scratch(ngen::GRFRange) {}
        template <typename... Args>
        void compute(Args &&...) {}
    };

    static BinaryOp toBinaryOp(const PostOps::entry_t &e);

    bool empty() const { return ops.empty(); }
    size_t len() const { return ops.len(); }
    PostOps::entry_t &operator[](size_t idx) { return ops[idx]; }
    const PostOps::entry_t &operator[](size_t idx) const { return ops[idx]; }

    void transpose() {
        std::swap(binaryRow, binaryCol);
        binaryTrans.flip();
    }

    void serialize(SerializationStream &) const {}

    template <typename Gen>
    void injectNonBinaryPostOps(const PostOps::entry_t &, Gen *,
                                ngen::RegisterAllocator, int *, int) const {
        throw std::runtime_error("injectNonBinaryPostOps not available");
    }

    template <typename Gen>
    void injectStochasticRound(Gen *, ngen::RegisterAllocator, int *, int,
                               const ngen::Subregister &, ngen::DataType) const {
        throw std::runtime_error("injectStochasticRound not available");
    }

    template <typename Gen>
    void injectMXScale(Gen *, ngen::RegisterAllocator, int *, int,
                       const ngen::Subregister &, ngen::DataType, int) const {
        throw std::runtime_error("injectMXScale not available");
    }

    PostOps ops;
    std::bitset<maxPostOps> binaryRow;
    std::bitset<maxPostOps> binaryCol;
    std::bitset<maxPostOps> binaryBatch;
    std::bitset<maxPostOps> binaryTrans;
    bool fwd = true;
    bool cStochasticRound = false;
};

} // namespace gemmstone

#endif
