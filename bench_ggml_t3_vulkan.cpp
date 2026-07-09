#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct BenchResult {
    std::string backend;
    std::string device;
    std::string name;
    std::string weight_type;
    int repeats = 0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    bool ok = false;
    std::string error;
};

const char * ggml_type_label(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        default: return "other";
    }
}

std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void set_cpu_threads_if_supported(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads != nullptr) {
        set_threads(backend, n_threads);
    }
}

void init_tensor(ggml_tensor * t) {
    if (t->op != GGML_OP_NONE || ggml_nbytes(t) == 0) {
        return;
    }

    const size_t n = ggml_nelements(t);
    std::mt19937 rng(12345u + static_cast<unsigned>(n) + static_cast<unsigned>(t->ne[0] * 17));
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);

    if (t->type == GGML_TYPE_F32) {
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(rng);
        }
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<float> data_f32(n);
        std::vector<ggml_fp16_t> data_f16(n);
        for (size_t i = 0; i < n; ++i) {
            data_f32[i] = dist(rng);
        }
        ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), static_cast<int64_t>(n));
        ggml_backend_tensor_set(t, data_f16.data(), 0, data_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
}

void init_leaf_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor(t);
    }
}

struct BuiltGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> roots;
};

BuiltGraph build_matvec(ggml_type weight_type, int in_dim, int out_dim) {
    ggml_init_params params = {
        /* .mem_size   = */ 32ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, weight_type, in_dim, out_dim);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);
    ggml_set_name(w, "w");
    ggml_set_name(x, "x");
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_set_name(out, "out");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    return {ctx, graph, {out}};
}

BuiltGraph build_layer_dense(ggml_type weight_type) {
    ggml_init_params params = {
        /* .mem_size   = */ 64ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int n_embd = 1024;
    constexpr int n_ff = 4096;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_embd * 3);
    ggml_tensor * wproj = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_embd);
    ggml_tensor * wfc = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff);
    ggml_tensor * wdown = ggml_new_tensor_2d(ctx, weight_type, n_ff, n_embd);

    ggml_tensor * norm1 = ggml_norm(ctx, x, 1e-5f);
    ggml_tensor * qkv = ggml_mul_mat(ctx, wqkv, norm1);
    ggml_tensor * proj = ggml_mul_mat(ctx, wproj, norm1);
    ggml_tensor * resid1 = ggml_add(ctx, x, proj);
    ggml_tensor * norm2 = ggml_norm(ctx, resid1, 1e-5f);
    ggml_tensor * fc = ggml_mul_mat(ctx, wfc, norm2);
    ggml_tensor * act = ggml_gelu(ctx, fc);
    ggml_tensor * down = ggml_mul_mat(ctx, wdown, act);
    ggml_tensor * out = ggml_add(ctx, resid1, down);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, qkv);
    ggml_build_forward_expand(graph, out);
    return {ctx, graph, {qkv, out}};
}

BuiltGraph build_attention_935(ggml_type weight_type) {
    ggml_init_params params = {
        /* .mem_size   = */ 32ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int head_dim = 64;
    constexpr int n_ctx = 935;
    constexpr int n_head = 16;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, weight_type, head_dim, n_ctx, n_head);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, weight_type, n_ctx, head_dim, n_head);

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, probs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    return {ctx, graph, {out}};
}

BenchResult run_bench(
        const std::string & backend_label,
        const std::string & device_label,
        ggml_backend_t backend,
        const std::string & name,
        ggml_type weight_type,
        int repeats,
        BuiltGraph (*builder)(ggml_type)) {
    BenchResult result;
    result.backend = backend_label;
    result.device = device_label;
    result.name = name;
    result.weight_type = ggml_type_label(weight_type);
    result.repeats = repeats;

    BuiltGraph built = builder(weight_type);
    if (built.ctx == nullptr || built.graph == nullptr) {
        result.error = "failed to build graph";
        return result;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(built.ctx, backend);
    if (buffer == nullptr) {
        result.error = "failed to allocate backend tensor buffer";
        ggml_free(built.ctx);
        return result;
    }

    if (backend_label == "vulkan") {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    init_leaf_tensors(built.ctx);
    ggml_backend_synchronize(backend);

    for (int i = 0; i < 3; ++i) {
        ggml_status status = ggml_backend_graph_compute(backend, built.graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            result.error = std::string("warmup failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(buffer);
            ggml_free(built.ctx);
            return result;
        }
    }

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = clock_type::now();
        ggml_status status = ggml_backend_graph_compute(backend, built.graph);
        ggml_backend_synchronize(backend);
        const auto t1 = clock_type::now();
        if (status != GGML_STATUS_SUCCESS) {
            result.error = std::string("compute failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(buffer);
            ggml_free(built.ctx);
            return result;
        }
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    result.ok = true;
    result.min_ms = *std::min_element(samples.begin(), samples.end());
    result.max_ms = *std::max_element(samples.begin(), samples.end());
    double total = 0.0;
    for (double x : samples) {
        total += x;
    }
    result.mean_ms = total / static_cast<double>(samples.size());

    ggml_backend_buffer_free(buffer);
    ggml_free(built.ctx);
    return result;
}

BenchResult run_matvec_bench(
        const std::string & backend_label,
        const std::string & device_label,
        ggml_backend_t backend,
        const std::string & name,
        ggml_type weight_type,
        int in_dim,
        int out_dim,
        int repeats) {
    auto builder = [in_dim, out_dim](ggml_type type) {
        return build_matvec(type, in_dim, out_dim);
    };

    BenchResult result;
    result.backend = backend_label;
    result.device = device_label;
    result.name = name;
    result.weight_type = ggml_type_label(weight_type);
    result.repeats = repeats;

    BuiltGraph built = builder(weight_type);
    if (built.ctx == nullptr || built.graph == nullptr) {
        result.error = "failed to build graph";
        return result;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(built.ctx, backend);
    if (buffer == nullptr) {
        result.error = "failed to allocate backend tensor buffer";
        ggml_free(built.ctx);
        return result;
    }

    if (backend_label == "vulkan") {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    init_leaf_tensors(built.ctx);
    ggml_backend_synchronize(backend);

    for (int i = 0; i < 3; ++i) {
        ggml_status status = ggml_backend_graph_compute(backend, built.graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            result.error = std::string("warmup failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(buffer);
            ggml_free(built.ctx);
            return result;
        }
    }

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = clock_type::now();
        ggml_status status = ggml_backend_graph_compute(backend, built.graph);
        ggml_backend_synchronize(backend);
        const auto t1 = clock_type::now();
        if (status != GGML_STATUS_SUCCESS) {
            result.error = std::string("compute failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(buffer);
            ggml_free(built.ctx);
            return result;
        }
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    result.ok = true;
    result.min_ms = *std::min_element(samples.begin(), samples.end());
    result.max_ms = *std::max_element(samples.begin(), samples.end());
    double total = 0.0;
    for (double x : samples) {
        total += x;
    }
    result.mean_ms = total / static_cast<double>(samples.size());

    ggml_backend_buffer_free(buffer);
    ggml_free(built.ctx);
    return result;
}

void write_outputs(const std::vector<BenchResult> & results, const std::string & json_path, const std::string & md_path) {
    {
        std::ofstream f(json_path);
        f << "{\n";
        f << "  \"description\": \"ggml Vulkan vs CPU primitive benchmarks for Chatterbox T3-sized operations on BC-250\",\n";
        f << "  \"baseline_note\": \"Existing Chatterbox baseline: CPU 270-char full API 62.809s, RTX 5090 3.495s, corrected IREE/Vulkan p1024 projection 81.956s, real-prefill bridge projection 121.172s.\",\n";
        f << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            f << "    {\n";
            f << "      \"backend\": \"" << json_escape(r.backend) << "\",\n";
            f << "      \"device\": \"" << json_escape(r.device) << "\",\n";
            f << "      \"name\": \"" << json_escape(r.name) << "\",\n";
            f << "      \"weight_type\": \"" << json_escape(r.weight_type) << "\",\n";
            f << "      \"repeats\": " << r.repeats << ",\n";
            f << "      \"ok\": " << (r.ok ? "true" : "false") << ",\n";
            f << "      \"mean_ms\": " << r.mean_ms << ",\n";
            f << "      \"min_ms\": " << r.min_ms << ",\n";
            f << "      \"max_ms\": " << r.max_ms << ",\n";
            f << "      \"error\": \"" << json_escape(r.error) << "\"\n";
            f << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }

    {
        std::ofstream f(md_path);
        f << "# ggml Vulkan T3 Primitive Benchmark\n\n";
        f << "Baseline saved here for comparison:\n\n";
        f << "- Chatterbox CPU full API, 270 chars: 62.809 s\n";
        f << "- RTX 5090 full API, 270 chars: 3.495 s\n";
        f << "- Corrected IREE/Vulkan p1024 projection: 81.956 s\n";
        f << "- Real-prefill IREE/Vulkan bridge projection: 121.172 s\n\n";
        f << "| Backend | Device | Op | Weights | Mean ms | Min ms | Max ms | Status |\n";
        f << "|---|---|---|---:|---:|---:|---:|---|\n";
        for (const auto & r : results) {
            f << "| " << r.backend
              << " | " << r.device
              << " | " << r.name
              << " | " << r.weight_type
              << " | " << r.mean_ms
              << " | " << r.min_ms
              << " | " << r.max_ms
              << " | " << (r.ok ? "ok" : r.error)
              << " |\n";
        }
    }
}

} // namespace

int main() {
    const char * lib_dir = "/root/llama.cpp/build-vulkan/bin";
    ggml_backend_load_all_from_path(lib_dir);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_register(ggml_backend_vk_reg());
    ggml_backend_t gpu = ggml_backend_vk_init(0);

    if (cpu == nullptr) {
        std::fprintf(stderr, "failed to initialize CPU backend\n");
        return 2;
    }
    if (gpu == nullptr) {
        std::fprintf(stderr, "failed to initialize GPU backend\n");
        ggml_backend_free(cpu);
        return 3;
    }

    set_cpu_threads_if_supported(cpu, 2);

    const std::string cpu_device = ggml_backend_dev_description(ggml_backend_get_device(cpu));
    const std::string gpu_device = ggml_backend_dev_description(ggml_backend_get_device(gpu));

    std::vector<BenchResult> results;
    const int repeats_small = 30;
    const int repeats_layer = 20;

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        results.push_back(run_matvec_bench("cpu", cpu_device, cpu, "matvec_1024_to_1024", type, 1024, 1024, repeats_small));
        results.push_back(run_matvec_bench("vulkan", gpu_device, gpu, "matvec_1024_to_1024", type, 1024, 1024, repeats_small));
        results.push_back(run_matvec_bench("cpu", cpu_device, cpu, "qkv_1024_to_3072", type, 1024, 3072, repeats_small));
        results.push_back(run_matvec_bench("vulkan", gpu_device, gpu, "qkv_1024_to_3072", type, 1024, 3072, repeats_small));
        results.push_back(run_matvec_bench("cpu", cpu_device, cpu, "mlp_up_1024_to_4096", type, 1024, 4096, repeats_small));
        results.push_back(run_matvec_bench("vulkan", gpu_device, gpu, "mlp_up_1024_to_4096", type, 1024, 4096, repeats_small));
        results.push_back(run_matvec_bench("cpu", cpu_device, cpu, "mlp_down_4096_to_1024", type, 4096, 1024, repeats_small));
        results.push_back(run_matvec_bench("vulkan", gpu_device, gpu, "mlp_down_4096_to_1024", type, 4096, 1024, repeats_small));
        results.push_back(run_matvec_bench("cpu", cpu_device, cpu, "head_1024_to_8192", type, 1024, 8192, repeats_small));
        results.push_back(run_matvec_bench("vulkan", gpu_device, gpu, "head_1024_to_8192", type, 1024, 8192, repeats_small));
        results.push_back(run_bench("cpu", cpu_device, cpu, "attention_ctx935_16h64", type, repeats_small, build_attention_935));
        results.push_back(run_bench("vulkan", gpu_device, gpu, "attention_ctx935_16h64", type, repeats_small, build_attention_935));
        results.push_back(run_bench("cpu", cpu_device, cpu, "dense_layer_no_kv", type, repeats_layer, build_layer_dense));
        results.push_back(run_bench("vulkan", gpu_device, gpu, "dense_layer_no_kv", type, repeats_layer, build_layer_dense));
    }

    const std::string json_path = "/root/chatterbox/exports/benchmarks/ggml_t3_vulkan_primitives_2026-07-08.json";
    const std::string md_path = "/root/chatterbox/exports/benchmarks/ggml_t3_vulkan_primitives_2026-07-08.md";
    write_outputs(results, json_path, md_path);

    std::cout << "Wrote " << json_path << "\n";
    std::cout << "Wrote " << md_path << "\n\n";
    std::cout << "| Backend | Device | Op | Weights | Mean ms | Status |\n";
    std::cout << "|---|---|---|---:|---:|---|\n";
    for (const auto & r : results) {
        std::cout << "| " << r.backend
                  << " | " << r.device
                  << " | " << r.name
                  << " | " << r.weight_type
                  << " | " << r.mean_ms
                  << " | " << (r.ok ? "ok" : r.error)
                  << " |\n";
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
