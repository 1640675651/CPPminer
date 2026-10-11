// Standalone integration harness: separate contexts on one GPU, or --physical on two.
// Including the implementation keeps the test-only logical-context setup out of the miner.
#include "../src/cuda/cp_gpu.cu"
#include <vector>

static void require(bool ok, const char* what)
{
    if(!ok){ fprintf(stderr, "FAIL: %s\n", what); exit(2); }
}

static void check_proof(const uint8_t header[76], int m, int n, int row, int col,
                        int layout, const std::vector<int8_t>& signal,
                        const uint32_t target[8])
{
    CpShareWitness* w = nullptr;
    require(cp_gpu_fetch_share_witness(row, col, layout, &w) == 0, "winner witness fetch");
    uint8_t root[32];
    pearl_keyed_digest_int8(signal.data(), signal.size(), g_zero_b.job_key, root);
    require(!memcmp(root, w->a_root, 32), "winner root vs independent CPU BLAKE3");
    CpMatrixWitness a{w->a_subroots, w->a_num_subroots, w->a_blocks, w->a_block_idx,
                      w->a_num_blocks, w->a_root};
    CpMatrixWitness b{w->bt_subroots, w->bt_num_subroots, nullptr, nullptr, 0, w->bt_root};
    std::vector<char> proof(2 * 1024 * 1024), reference(proof.size());
    const uint8_t* config = layout == CP_TILE_LAYOUT_CUTLASS ?
                            PEARL_CUTLASS_CONFIG : PEARL_SCATTERED_CONFIG;
    char err[1024] = {};
    require(cp_proof_build_witness(header, 76, config, 52, &a, &b, m, n, K_DIM,
                R_RANK, row, col, layout, proof.data(), proof.size(), err, sizeof(err)) == 0, err);
    require(cp_proof_build(header, 76, config, 52, signal.data(), nullptr, m, n,
                K_DIM, R_RANK, row, col, layout, reference.data(), reference.size(),
                err, sizeof(err)) == 0, err);
    require(!strcmp(proof.data(), reference.data()), "device witness vs full host proof");
    uint8_t target_be[32];
    for(int i = 0; i < 8; i++) for(int j = 0; j < 4; j++)
        target_be[i * 4 + j] = (uint8_t)(target[7 - i] >> (24 - 8 * j));
    require(cp_proof_verify(header, 76, (uint8_t*)proof.data(), strlen(proof.data()),
                           target_be, 3, err, sizeof(err)) == 0, err);
    // Mixing a different device's root must be rejected by the real Rust builder.
    a.root = g_attempts[0].root;
    require(cp_proof_build_witness(header, 76, config, 52, &a, &b, m, n, K_DIM,
                R_RANK, row, col, layout, reference.data(), reference.size(),
                err, sizeof(err)) != 0, "cross-device root mix must fail");
    cp_share_witness_free(w);
}

int main(int argc, char** argv)
{
    bool physical = false, fused = true;
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--physical")) physical = true;
        else if(!strcmp(argv[i], "--scalar")){ fused = false; g_period_gemm = 0; }
        else if(!strcmp(argv[i], "--period")){ fused = false; g_period_gemm = 1; }
        else require(false, "unknown test argument");
    }
    int devices[2] = {0, 1};
    g_cutlass_fused = fused ? 1 : 0;
    pearl_set_cutlass_fused(g_cutlass_fused);
    cp_gpu_set_row_period_batch(2);
    cp_gpu_set_col_period_batch(2);
    g_m_active = g_n_active = 1024;
    cp_gpu_init(devices, physical ? 2 : 1);
    if(!physical){
        // Two independent allocations on CMP50 exercise ownership and accounting;
        // they cannot validate cross-device copies or hardware concurrency.
        g_ngpu = 2;
        GpuCtx* g = &g_gpus[1];
        g->dev = 0; g->use_cutlass_fused = g_cutlass_fused;
        CU_CHECK(cudaMalloc(&g->d_found, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_rows, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_out_t_cols, sizeof(int)));
        CU_CHECK(cudaMalloc(&g->d_a_key8, 32));
    }
    uint8_t header[76] = {}, key[32];
    uint32_t impossible[8] = {};
    int row = -1, col = -1;
    uint64_t tiles = 0;
    const uint64_t expected = (uint64_t)cp_pp_num_row_parts(1024, 0) *
                                       cp_pp_num_col_parts(1024, 0);
    for(int job = 0; job < 2; job++){
        header[36] = (uint8_t)(job + 1);
        pearl_job_key(header, 76, key);
        cp_gpu_begin_job(key, 1024, 1024, 3);
        for(int attempt = 0; attempt < 3; attempt++){
            require(cp_gpu_mine_attempt(nullptr, 0, key, impossible, 1024, 1024, 0,
                        nullptr, nullptr, nullptr, nullptr, nullptr, &row, &col, &tiles) == 0,
                    "independent no-hit scan");
            require(tiles == expected * 2, "two independent scans counted");
            require(memcmp(g_attempts[0].root, g_attempts[1].root, 32) != 0,
                    "distinct A commitments on every attempt and job");
            uint8_t keys[2][32];
            for(int i = 0; i < 2; i++){
                CU_CHECK(cudaSetDevice(g_gpus[i].dev));
                CU_CHECK(cudaMemcpy(keys[i], g_gpus[i].d_a_key8, 32, cudaMemcpyDeviceToHost));
            }
            require(memcmp(keys[0], keys[1], 32) != 0, "distinct jackpot keys");
        }
        // A real jackpot on GPU1 with no GPU0 hit. Scan one CTA per device; repeat
        // independent attempts until this case occurs (bounded to 64 trials).
        uint32_t small_target[8] = {};
        small_target[7] = fused ? 32 : 16; // Roughly one hit per 512 tiles after scaling.
        bool second_hit = false;
        for(int trial = 0; trial < 64 && !second_hit; trial++){
            for(int i = 0; i < 2; i++){
                GpuCtx* g = &g_gpus[i];
                uint8_t device_key[32];
                require(gpu_prepare_attempt_a(g, cp_gpu_fresh_rng_seed(), key,
                        1024, 1024, device_key) == 0, "fresh real-hit trial");
                CU_CHECK(cudaMemcpy(g->d_a_key8, device_key, 32, cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemset(g->d_found, 0, sizeof(int)));
            }
            const int found = gpu_scan_device(nullptr, small_target, 128, fused ? 128 : 256,
                                               &row, &col, &tiles);
            second_hit = found == 1 && g_share_gpu == 1;
        }
        require(second_hit, "real jackpot selected second device within 64 trials");
        GpuCtx* winner = &g_gpus[1];
        CU_CHECK(cudaSetDevice(winner->dev));
        std::vector<int8_t> signal((size_t)1024 * K_DIM), actual(signal.size());
        CU_CHECK(cudaMemcpy(actual.data(), winner->d_A_sig, actual.size(), cudaMemcpyDeviceToHost));
        require(cp_gpu_fetch_share_signals(signal.data(), nullptr) == 0 && signal == actual,
                "fetch selects second device's A");
        check_proof(header, 1024, 1024, row, col,
                    fused ? CP_TILE_LAYOUT_CUTLASS : CP_TILE_LAYOUT_SCATTERED,
                    signal, small_target);
    }
    // Shared CPU inputs: each batch has exactly one owner.
    g_shared_a = 1;
    for(int r = 0; r < 5; r++) for(int c = 0; c < 7; c++)
        require(gpu_owns_batch(0, r, c, 7) != gpu_owns_batch(1, r, c, 7),
                "shared-input disjoint batch ownership");
    std::vector<int8_t> shared_a((size_t)1024 * K_DIM, 1), shared_b(shared_a.size(), 2);
    require(cp_gpu_mine_attempt(nullptr, 0, key, impossible, 1024, 1024, 1,
            shared_a.data(), shared_b.data(), key, nullptr, nullptr,
            &row, &col, &tiles) == 0 && tiles == expected, "CPU-upload scan counted once");
    require(cp_gpu_mine_plain_proof(shared_a.data(), shared_b.data(), key, impossible,
            1024, 1024, &row, &col, &tiles) == 0 && tiles == expected,
            "explicit-input scan counted once");
    cp_job_mine_begin("multi-gpu-cancel");
    cp_job_request_cancel();
    require(cp_gpu_mine_attempt(nullptr, 0, key, impossible, 1024, 1024, 1,
            shared_a.data(), shared_b.data(), key, nullptr, nullptr,
            &row, &col, &tiles) == -1 && tiles == 0 && g_share_gpu == -1,
            "cancelled shared scan reports no work or winner");
    cp_job_mine_end();
    cp_gpu_shutdown();
    require(g_ngpu == 0 && g_share_gpu == -1, "shutdown clears ownership");
    puts(physical ? "PASS: physical multi-GPU ownership and proofs" :
                    "PASS: logical multi-GPU ownership and proofs (one physical GPU)");
}
