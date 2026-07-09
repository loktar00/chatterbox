#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int HIDDEN = 1024;
constexpr int N_HEAD = 16;
constexpr int HEAD_DIM = 64;
#ifndef T3_GGML_PAST_LEN
#define T3_GGML_PAST_LEN 16
#endif
constexpr int PAST_LEN = T3_GGML_PAST_LEN;
constexpr int SEQ_LEN = PAST_LEN + 1;
constexpr int FF = 4096;
constexpr float LN_EPS = 1e-5f;
constexpr float ATTN_SCALE = 0.125f;

std::string fixture_dir() {
    const char * env = std::getenv("GGML_T3_BLOCK0_FIXTURE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    if (PAST_LEN == 16) {
        return "/root/chatterbox/exports/ggml_t3_block0_fixture";
    }
    return "/root/chatterbox/exports/ggml_t3_block0_fixture_p" + std::to_string(PAST_LEN);
}

std::string output_json() {
    return "/root/chatterbox/exports/benchmarks/ggml_t3_block0_correctness_p" + std::to_string(PAST_LEN) + "_2026-07-08.json";
}

std::string output_md() {
    return "/root/chatterbox/exports/benchmarks/ggml_t3_block0_correctness_p" + std::to_string(PAST_LEN) + "_2026-07-08.md";
}

using clock_type = std::chrono::steady_clock;

struct Result {
    std::string backend;
    std::string device;
    bool ok = false;
    double mean_ms = 0.0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rms_error = 0.0;
    bool allclose_1e_4 = false;
    bool allclose_1e_3 = false;
    std::string error;
};

struct BlockGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;

    ggml_tensor * input = nullptr;
    ggml_tensor * past_k = nullptr;
    ggml_tensor * past_v = nullptr;

    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    ggml_tensor * c_attn_w = nullptr;
    ggml_tensor * c_attn_b = nullptr;
    ggml_tensor * c_proj_w = nullptr;
    ggml_tensor * c_proj_b = nullptr;
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    ggml_tensor * c_fc_w = nullptr;
    ggml_tensor * c_fc_b = nullptr;
    ggml_tensor * mlp_proj_w = nullptr;
    ggml_tensor * mlp_proj_b = nullptr;

    ggml_tensor * output = nullptr;
};

std::string path_for(const std::string & name) {
    return fixture_dir() + "/" + name + ".f32";
}

std::vector<float> read_f32(const std::string & path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }
    std::vector<float> data(count);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("short read from " + path);
    }
    return data;
}

void set_tensor_file(ggml_tensor * t, const std::string & name, size_t count) {
    std::vector<float> data = read_f32(path_for(name), count);
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

ggml_tensor * repeat_to(ggml_context * ctx, ggml_tensor * src, ggml_tensor * dst_like) {
    return ggml_repeat(ctx, src, dst_like);
}

ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(ctx, x, repeat_to(ctx, bias, x));
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x, ggml_tensor * bias) {
    return add_bias(ctx, ggml_mul_mat(ctx, weight, x), bias);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * norm = ggml_norm(ctx, x, LN_EPS);
    ggml_tensor * scaled = ggml_mul(ctx, norm, repeat_to(ctx, weight, norm));
    return ggml_add(ctx, scaled, repeat_to(ctx, bias, norm));
}

BlockGraph build_graph() {
    ggml_init_params params = {
        /* .mem_size   = */ 96ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    BlockGraph g;
    g.ctx = ctx;

    g.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, 1);
    g.past_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, PAST_LEN, N_HEAD);
    g.past_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PAST_LEN, HEAD_DIM, N_HEAD);

    g.ln1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.c_attn_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, 3 * HIDDEN);
    g.c_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * HIDDEN);
    g.c_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, HIDDEN);
    g.c_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.c_fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, FF);
    g.c_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FF);
    g.mlp_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FF, HIDDEN);
    g.mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);

    ggml_tensor * ln1 = layer_norm(ctx, g.input, g.ln1_w, g.ln1_b);
    ggml_tensor * qkv = linear(ctx, g.c_attn_w, ln1, g.c_attn_b);

    ggml_tensor * q_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 0);
    ggml_tensor * k_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], HIDDEN * sizeof(float));
    ggml_tensor * v_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 2 * HIDDEN * sizeof(float));

    ggml_tensor * q_3d = ggml_reshape_3d(ctx, q_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * k_new_3d = ggml_reshape_3d(ctx, k_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * v_new_3d = ggml_reshape_3d(ctx, v_flat, HEAD_DIM, N_HEAD, 1);

    ggml_tensor * q = ggml_permute(ctx, q_3d, 0, 2, 1, 3);          // [64, 1, 16]
    ggml_tensor * k_new = ggml_permute(ctx, k_new_3d, 0, 2, 1, 3);  // [64, 1, 16]
    ggml_tensor * v_new = ggml_permute(ctx, v_new_3d, 1, 2, 0, 3);  // [1, 64, 16]

    ggml_tensor * k_all = ggml_concat(ctx, g.past_k, k_new, 1);     // [64, 17, 16]
    ggml_tensor * v_all = ggml_concat(ctx, g.past_v, v_new, 0);     // [17, 64, 16]

    ggml_tensor * kq = ggml_mul_mat(ctx, k_all, q);                 // [17, 1, 16]
    ggml_tensor * probs = ggml_soft_max_ext(ctx, kq, nullptr, ATTN_SCALE, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_all, probs);            // [64, 1, 16]
    ggml_tensor * merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);      // [64, 16, 1]
    ggml_tensor * attn_out = ggml_cont_2d(ctx, merged, HIDDEN, 1);
    ggml_tensor * attn_proj = linear(ctx, g.c_proj_w, attn_out, g.c_proj_b);
    ggml_tensor * resid1 = ggml_add(ctx, g.input, attn_proj);

    ggml_tensor * ln2 = layer_norm(ctx, resid1, g.ln2_w, g.ln2_b);
    ggml_tensor * fc = linear(ctx, g.c_fc_w, ln2, g.c_fc_b);
    ggml_tensor * act = ggml_gelu(ctx, fc);
    ggml_tensor * mlp = linear(ctx, g.mlp_proj_w, act, g.mlp_proj_b);
    g.output = ggml_add(ctx, resid1, mlp);

    g.graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(g.graph, g.output);
    return g;
}

void load_fixture(BlockGraph & g) {
    set_tensor_file(g.input, "input_hidden", HIDDEN);
    set_tensor_file(g.past_k, "past_k", HEAD_DIM * PAST_LEN * N_HEAD);
    set_tensor_file(g.past_v, "past_v", PAST_LEN * HEAD_DIM * N_HEAD);
    set_tensor_file(g.ln1_w, "ln1_weight", HIDDEN);
    set_tensor_file(g.ln1_b, "ln1_bias", HIDDEN);
    set_tensor_file(g.c_attn_w, "attn_c_attn_weight", HIDDEN * 3 * HIDDEN);
    set_tensor_file(g.c_attn_b, "attn_c_attn_bias", 3 * HIDDEN);
    set_tensor_file(g.c_proj_w, "attn_c_proj_weight", HIDDEN * HIDDEN);
    set_tensor_file(g.c_proj_b, "attn_c_proj_bias", HIDDEN);
    set_tensor_file(g.ln2_w, "ln2_weight", HIDDEN);
    set_tensor_file(g.ln2_b, "ln2_bias", HIDDEN);
    set_tensor_file(g.c_fc_w, "mlp_c_fc_weight", HIDDEN * FF);
    set_tensor_file(g.c_fc_b, "mlp_c_fc_bias", FF);
    set_tensor_file(g.mlp_proj_w, "mlp_c_proj_weight", FF * HIDDEN);
    set_tensor_file(g.mlp_proj_b, "mlp_c_proj_bias", HIDDEN);
}

void compare(const std::vector<float> & actual, const std::vector<float> & expected, Result & result) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    bool all_1e_4 = true;
    bool all_1e_3 = true;

    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double tol_1e_4 = 1e-4 + 1e-4 * std::abs(static_cast<double>(expected[i]));
        const double tol_1e_3 = 1e-3 + 1e-3 * std::abs(static_cast<double>(expected[i]));
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        all_1e_4 = all_1e_4 && diff <= tol_1e_4;
        all_1e_3 = all_1e_3 && diff <= tol_1e_3;
    }

    result.max_abs_error = max_abs;
    result.mean_abs_error = sum_abs / static_cast<double>(actual.size());
    result.rms_error = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    result.allclose_1e_4 = all_1e_4;
    result.allclose_1e_3 = all_1e_3;
}

Result run_backend(const std::string & label, ggml_backend_t backend, const std::vector<float> & expected) {
    Result result;
    result.backend = label;
    result.device = ggml_backend_dev_description(ggml_backend_get_device(backend));

    try {
        BlockGraph g = build_graph();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate backend tensor buffer");
        }
        if (label == "vulkan") {
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        load_fixture(g);
        ggml_backend_synchronize(backend);

        for (int i = 0; i < 3; ++i) {
            ggml_status status = ggml_backend_graph_compute(backend, g.graph);
            ggml_backend_synchronize(backend);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("warmup failed: ") + ggml_status_to_string(status));
            }
        }

        constexpr int repeats = 20;
        double elapsed_ms = 0.0;
        for (int i = 0; i < repeats; ++i) {
            const auto t0 = clock_type::now();
            ggml_status status = ggml_backend_graph_compute(backend, g.graph);
            ggml_backend_synchronize(backend);
            const auto t1 = clock_type::now();
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
            }
            elapsed_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        result.mean_ms = elapsed_ms / repeats;

        std::vector<float> actual(HIDDEN);
        ggml_backend_tensor_get(g.output, actual.data(), 0, actual.size() * sizeof(float));
        compare(actual, expected, result);
        result.ok = result.allclose_1e_3;

        ggml_backend_buffer_free(buffer);
        ggml_free(g.ctx);
    } catch (const std::exception & e) {
        result.ok = false;
        result.error = e.what();
    }

    return result;
}

std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void write_outputs(const std::vector<Result> & results) {
    {
        std::ofstream f(output_json());
        f << "{\n";
        f << "  \"description\": \"Real Chatterbox T3 GPT-2 block-0 cached one-token ggml correctness probe\",\n";
        f << "  \"fixture_manifest\": \"" << fixture_dir() << "/manifest.json\",\n";
        f << "  \"hidden_size\": " << HIDDEN << ",\n";
        f << "  \"n_head\": " << N_HEAD << ",\n";
        f << "  \"head_dim\": " << HEAD_DIM << ",\n";
        f << "  \"past_len\": " << PAST_LEN << ",\n";
        f << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            f << "    {\n";
            f << "      \"backend\": \"" << json_escape(r.backend) << "\",\n";
            f << "      \"device\": \"" << json_escape(r.device) << "\",\n";
            f << "      \"ok\": " << (r.ok ? "true" : "false") << ",\n";
            f << "      \"mean_ms\": " << r.mean_ms << ",\n";
            f << "      \"max_abs_error\": " << r.max_abs_error << ",\n";
            f << "      \"mean_abs_error\": " << r.mean_abs_error << ",\n";
            f << "      \"rms_error\": " << r.rms_error << ",\n";
            f << "      \"allclose_1e_4\": " << (r.allclose_1e_4 ? "true" : "false") << ",\n";
            f << "      \"allclose_1e_3\": " << (r.allclose_1e_3 ? "true" : "false") << ",\n";
            f << "      \"error\": \"" << json_escape(r.error) << "\"\n";
            f << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }

    {
        std::ofstream f(output_md());
        f << "# ggml Real T3 Block-0 Correctness Probe\n\n";
        f << "Fixture: `" << fixture_dir() << "/manifest.json`\n\n";
        f << "Cache length: `" << PAST_LEN << "` past tokens, `" << SEQ_LEN << "` tokens after update.\n\n";
        f << "| Backend | Device | Mean ms | Max abs error | Mean abs error | RMS error | allclose 1e-4 | allclose 1e-3 | Status |\n";
        f << "|---|---|---:|---:|---:|---:|---:|---:|---|\n";
        for (const auto & r : results) {
            f << "| " << r.backend
              << " | " << r.device
              << " | " << r.mean_ms
              << " | " << r.max_abs_error
              << " | " << r.mean_abs_error
              << " | " << r.rms_error
              << " | " << (r.allclose_1e_4 ? "true" : "false")
              << " | " << (r.allclose_1e_3 ? "true" : "false")
              << " | " << (r.ok ? "ok" : r.error)
              << " |\n";
        }
    }
}

} // namespace

int main() {
    ggml_backend_load_all_from_path("/root/llama.cpp/build-vulkan/bin");
    ggml_backend_register(ggml_backend_vk_reg());

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (cpu == nullptr || vk == nullptr) {
        std::cerr << "failed to initialize required backends\n";
        return 2;
    }

    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    if (set_threads != nullptr) {
        set_threads(cpu, 2);
    }

    std::vector<float> expected = read_f32(path_for("ref_output_hidden"), HIDDEN);
    std::vector<Result> results;
    results.push_back(run_backend("cpu", cpu, expected));
    results.push_back(run_backend("vulkan", vk, expected));
    write_outputs(results);

    std::cout << "Wrote " << output_json() << "\n";
    std::cout << "Wrote " << output_md() << "\n\n";
    for (const auto & r : results) {
        std::cout << r.backend
                  << " device=\"" << r.device << "\""
                  << " mean_ms=" << r.mean_ms
                  << " max_abs_error=" << r.max_abs_error
                  << " allclose_1e_4=" << (r.allclose_1e_4 ? "true" : "false")
                  << " allclose_1e_3=" << (r.allclose_1e_3 ? "true" : "false")
                  << " status=" << (r.ok ? "ok" : r.error)
                  << "\n";
    }

    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    return std::all_of(results.begin(), results.end(), [](const Result & r) { return r.ok; }) ? 0 : 1;
}
