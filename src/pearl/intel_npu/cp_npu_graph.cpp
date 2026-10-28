#include "cp_npu_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include <openvino/core/attribute_visitor.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>

namespace cp_npu {
namespace {

using Out = ov::Output<ov::Node>;
using ov::op::v0::Constant;

/*
 * The patched compiler's custom op, serialized as type "TileXorAcc", version "extension", with the
 * bool attributes has_acc and dep_acc (the compiler deserializes it with its own copy of this class,
 * vpux/compiler/frontend/tile_xor_acc.hpp). Inputs int32 [1, 1, H, W]: with has_acc the first is the
 * running accumulator, the others are partial sums. Outputs: the updated accumulator and the tile
 * words [1, 1, H/kTileRows, G * W/16] (partial-major within a tile row); the compiler takes the tile
 * height from NPU_EXPERIMENTAL_TILE_XOR_ROWS.
 */
class TileXorAcc : public ov::op::Op {
public:
    OPENVINO_OP("TileXorAcc", "extension", ov::op::Op);

    TileXorAcc() = default;
    TileXorAcc(const ov::OutputVector& inputs, bool has_acc)
        : ov::op::Op(inputs), m_has_acc(has_acc) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto dims = get_input_partial_shape(0).to_shape();
        const auto partials = static_cast<int64_t>(get_input_size()) - (m_has_acc ? 1 : 0);
        set_output_type(0, ov::element::i32, get_input_partial_shape(0));
        set_output_type(1, ov::element::i32,
                        ov::PartialShape{1, 1, static_cast<int64_t>(dims[2]) / kTileRows,
                                         partials * (static_cast<int64_t>(dims[3]) / kTileCols)});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<TileXorAcc>(args, m_has_acc);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("has_acc", m_has_acc);
        visitor.on_attribute("dep_acc", m_dep_acc);
        return true;
    }

private:
    bool m_has_acc = false;
    bool m_dep_acc = false;
};

Out i64s(const std::vector<int64_t>& v) {
    return Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
}

Out f16_scalar(float v) {
    return Constant::create(ov::element::f16, ov::Shape{}, {v});
}

/* u8 holding value + 128 -> int8 for the DPU. The compiler folds this Convert + FakeQuantize pair
 * into a cast, so no device work runs for it. */
Out quantize_u8(const Out& node) {
    auto converted = std::make_shared<ov::op::v0::Convert>(node, ov::element::f16);
    return std::make_shared<ov::op::v0::FakeQuantize>(converted, f16_scalar(0.0f), f16_scalar(255.0f),
                                                      f16_scalar(-128.0f), f16_scalar(127.0f), 256);
}

Out slice(const Out& node, const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
          const std::vector<int64_t>& axes) {
    std::vector<int64_t> step(start.size(), 1);
    return std::make_shared<ov::op::v8::Slice>(node, i64s(start), i64s(stop), i64s(step), i64s(axes));
}

Out reshape(const Out& node, const std::vector<int64_t>& shape) {
    return std::make_shared<ov::op::v1::Reshape>(node, i64s(shape), false);
}

Out concat(const ov::OutputVector& nodes, int64_t axis) {
    return nodes.size() > 1 ? Out(std::make_shared<ov::op::v0::Concat>(nodes, axis)) : nodes[0];
}

/* Panel p of a [K/128, X, 128] input as [X, 128]. */
Out panel(const Out& input, int64_t p) {
    return std::make_shared<ov::op::v0::Squeeze>(slice(input, {p}, {p + 1}, {0}), i64s({0}));
}

} // namespace

const char* launch_shape_error(const LaunchShape& s) {
    if (s.k % kKr)
        return "K must be a multiple of 128";
    if (s.chunk_m % (2 * kTileRows) || s.rows % s.chunk_m)
        return "rows must be a multiple of the row block (chunk_m, itself a multiple of two tile rows)";
    if (s.chunk_n % kTileCols || s.cols % s.chunk_n)
        return "cols must be a multiple of the column chain (chunk_n, itself a multiple of 16)";
    if (s.cols / s.chunk_n > 8)
        return "at most 8 column chains (cols / chunk_n): lockstep keeps all of them live at once";
    /* The kernel's record scan needs at least 4 partials per op (3 with an accumulator), so the last
     * group of panels must not be shorter. */
    const int tail = s.milestones() % s.group;
    if (s.group < 4 || s.group > 8 || (tail && tail < 3))
        return "group must be 4..8 and leave a last group of at least 3 panels";
    return nullptr;
}

std::shared_ptr<ov::Model> build_launch_model(const LaunchShape& s, std::vector<int>* word_index) {
    if (const char* why = launch_shape_error(s))
        throw std::invalid_argument(std::string("cp_npu: unsupported launch shape: ") + why);
    const int64_t ms = s.milestones(), cols = s.cols, chunk_m = s.chunk_m, chunk_n = s.chunk_n;
    const int64_t blocks = s.row_blocks(), chains = cols / chunk_n, words_per_chain = chunk_n / kTileCols;

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{size_t(ms), size_t(s.rows), size_t(kKr)});
    a->set_friendly_name("a_panels");
    ov::ParameterVector params{a};
    for (int i = 0; i < s.b_inputs(); ++i) {
        auto bt = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{size_t(ms), size_t(cols), size_t(kKr)});
        bt->set_friendly_name(i ? "bt_panels_" + std::to_string(i) : std::string("bt_panels"));
        params.push_back(bt);
    }

    ov::OutputVector block_words;
    for (int64_t rb = 0; rb < blocks; ++rb) {
        const Out& bt = params[size_t(s.split_b ? 1 + rb : 1)];
        /* A panel p of the row block is shared by its chains; each B block (panel p, chain c) feeds one
         * GEMM of this row block. Rows are cut before the cast: a cast feeding the MatMul directly is
         * what the compiler's int32-output path matches. */
        std::vector<Out> aq(ms);
        for (int64_t p = 0; p < ms; ++p)
            aq[p] = quantize_u8(blocks == 1 ? panel(a, p) : slice(panel(a, p), {rb * chunk_m}, {(rb + 1) * chunk_m}, {0}));
        auto gemm = [&](int64_t p, int64_t c) {
            Out bq = quantize_u8(slice(panel(bt, p), {c * chunk_n}, {(c + 1) * chunk_n}, {0}));
            /* B as rows of B^T (transpose_b): the form the int32-output DPU path handles exactly. */
            auto mm = std::make_shared<ov::op::v0::MatMul>(aq[p], bq, false, true);
            return reshape(std::make_shared<ov::op::v0::Convert>(mm, ov::element::i32), {1, 1, chunk_m, chunk_n});
        };

        /* Per chain: ops over groups of panels, carrying the accumulator. */
        struct Step {
            int64_t ms0, count;
            Out words;
        };
        std::vector<std::vector<Step>> steps(chains);
        for (int64_t c = 0; c < chains; ++c) {
            Out acc;
            for (int64_t p0 = 0; p0 < ms; p0 += s.group) {
                const int64_t p1 = std::min<int64_t>(p0 + s.group, ms);
                ov::OutputVector inputs;
                if (p0 > 0)
                    inputs.push_back(acc);
                for (int64_t p = p0; p < p1; ++p)
                    inputs.push_back(gemm(p, c));
                auto node = std::make_shared<TileXorAcc>(inputs, p0 > 0);
                acc = node->output(0);
                steps[c].push_back({p0, p1 - p0, node->output(1)});
            }
        }

        /* Lockstep order: step s of every chain, then step s + 1. OpenVINO's topological order visits a
         * node's inputs in order and the compiler's scheduler follows it, so each A panel is used by all
         * chains in quick succession and freed instead of staying in CMX for the whole row block. Every
         * row block has the same word order. */
        ov::OutputVector words;
        word_index->assign(size_t(s.tile_cols() * ms), 0);
        int offset = 0;
        for (size_t step = 0; step < steps[0].size(); ++step)
            for (int64_t c = 0; c < chains; ++c) {
                const Step& st = steps[c][step];
                words.push_back(st.words);
                for (int64_t i = 0; i < st.count; ++i)
                    for (int64_t t = 0; t < words_per_chain; ++t)
                        (*word_index)[size_t((c * words_per_chain + t) * ms + st.ms0 + i)] =
                            offset + int(i * words_per_chain + t);
                offset += int(st.count * words_per_chain);
            }
        block_words.push_back(concat(words, 3)); /* [1, 1, chunk_m/8, ms * cols/16] */
    }
    Out out = reshape(concat(block_words, 2), {s.rows / kTileRows, ms * (cols / kTileCols)});
    return std::make_shared<ov::Model>(ov::OutputVector{out}, params, "pearl_npu_launch");
}

} // namespace cp_npu
