#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int HIDDEN = 256;
constexpr int TOKENS = 16;
constexpr float EPS = 1e-5f;
using clock_type = std::chrono::steady_clock;

std::string fixture_dir() {
    return "/root/chatterbox/exports/ggml_s3_layernorm_mid_t16";
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

std::vector<float> read_fixture(const std::string & name, size_t count) {
    return read_f32(fixture_dir() + "/" + name, count);
}

void set_tensor(ggml_tensor * t, const std::vector<float> & data) {
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

struct Metrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    bool allclose_1e_4 = false;
    bool allclose_1e_3 = false;
};

Metrics compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    Metrics m;
    double sum = 0.0;
    bool close_1e4 = true;
    bool close_1e3 = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double ref = std::abs(static_cast<double>(expected[i]));
        m.max_abs = std::max(m.max_abs, diff);
        sum += diff;
        close_1e4 = close_1e4 && diff <= (1e-4 + 1e-4 * ref);
        close_1e3 = close_1e3 && diff <= (1e-3 + 1e-3 * ref);
    }
    m.mean_abs = sum / static_cast<double>(actual.size());
    m.allclose_1e_4 = close_1e4;
    m.allclose_1e_3 = close_1e3;
    return m;
}

struct Result {
    std::string backend_name;
    std::string device;
    double mean_ms = 0.0;
    Metrics metrics;
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * output = nullptr;
};

Graph build_graph() {
    ggml_init_params params = { 4ull * 1024ull * 1024ull, nullptr, true };
    Graph g;
    g.ctx = ggml_init(params);
    if (!g.ctx) {
        throw std::runtime_error("ggml_init failed");
    }
    g.input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, HIDDEN, TOKENS);
    g.weight = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, HIDDEN);
    g.bias = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, HIDDEN);
    ggml_tensor * norm = ggml_norm(g.ctx, g.input, EPS);
    ggml_tensor * scaled = ggml_mul(g.ctx, norm, ggml_repeat(g.ctx, g.weight, norm));
    g.output = ggml_add(g.ctx, scaled, ggml_repeat(g.ctx, g.bias, norm));
    g.graph = ggml_new_graph_custom(g.ctx, 64, false);
    ggml_build_forward_expand(g.graph, g.output);
    return g;
}

Result run_backend(const std::string & name, ggml_backend_t backend, int repeats) {
    auto input = read_fixture("input_t_h.f32", HIDDEN * TOKENS);
    auto weight = read_fixture("weight_h.f32", HIDDEN);
    auto bias = read_fixture("bias_h.f32", HIDDEN);
    auto reference = read_fixture("reference_t_h.f32", HIDDEN * TOKENS);

    Graph g = build_graph();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (buffer == nullptr) {
        throw std::runtime_error("failed to allocate tensors for " + name);
    }
    set_tensor(g.input, input);
    set_tensor(g.weight, weight);
    set_tensor(g.bias, bias);

    // Warm up once so shader/module setup is not included.
    ggml_status status = ggml_backend_graph_compute(backend, g.graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("warmup failed: ") + ggml_status_to_string(status));
    }

    double total_ms = 0.0;
    for (int i = 0; i < repeats; ++i) {
        ggml_backend_synchronize(backend);
        const auto t0 = clock_type::now();
        status = ggml_backend_graph_compute(backend, g.graph);
        ggml_backend_synchronize(backend);
        const auto t1 = clock_type::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
        }
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::vector<float> output(HIDDEN * TOKENS);
    ggml_backend_tensor_get(g.output, output.data(), 0, output.size() * sizeof(float));

    Result r;
    r.backend_name = name;
    r.device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    r.mean_ms = total_ms / static_cast<double>(repeats);
    r.metrics = compare(output, reference);

    ggml_backend_buffer_free(buffer);
    ggml_free(g.ctx);
    return r;
}

std::string json_bool(bool value) {
    return value ? "true" : "false";
}

void write_results(const std::vector<Result> & results) {
    const std::string json_path = "/root/chatterbox/exports/benchmarks/ggml_s3_layernorm_mid_t16_2026-07-08.json";
    const std::string md_path = "/root/chatterbox/exports/benchmarks/ggml_s3_layernorm_mid_t16_2026-07-08.md";
    {
        std::ofstream f(json_path);
        f << "{\n";
        f << "  \"description\": \"ggml S3 mid LayerNorm repro on CPU and Vulkan\",\n";
        f << "  \"fixture_manifest\": \"" << fixture_dir() << "/manifest.json\",\n";
        f << "  \"shape\": [1, " << TOKENS << ", " << HIDDEN << "],\n";
        f << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            f << "    {\"backend\": \"" << r.backend_name << "\", \"device\": \"" << r.device
              << "\", \"mean_ms\": " << r.mean_ms
              << ", \"max_abs_error\": " << r.metrics.max_abs
              << ", \"mean_abs_error\": " << r.metrics.mean_abs
              << ", \"allclose_1e_4\": " << json_bool(r.metrics.allclose_1e_4)
              << ", \"allclose_1e_3\": " << json_bool(r.metrics.allclose_1e_3) << "}"
              << (i + 1 == results.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }
    {
        std::ofstream f(md_path);
        f << "# ggml S3 LayerNorm Probe\n\n";
        f << "Fixture: `" << fixture_dir() << "/manifest.json`\n\n";
        f << "| Backend | Device | Mean ms | Max abs | Mean abs | allclose 1e-4 | Status |\n";
        f << "|---|---|---:|---:|---:|---:|---|\n";
        for (const auto & r : results) {
            f << "| " << r.backend_name << " | " << r.device << " | " << r.mean_ms
              << " | " << r.metrics.max_abs << " | " << r.metrics.mean_abs
              << " | " << json_bool(r.metrics.allclose_1e_4)
              << " | " << (r.metrics.allclose_1e_4 ? "ok" : "fail") << " |\n";
        }
    }
    std::cout << "wrote=" << json_path << "\n";
    std::cout << "wrote=" << md_path << "\n";
}

} // namespace

int main() {
    try {
        ggml_backend_load_all_from_path("/root/llama.cpp/build-vulkan/bin");
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        ggml_backend_register(ggml_backend_vk_reg());
        ggml_backend_t vk = ggml_backend_vk_init(0);
        if (!vk) {
            throw std::runtime_error("failed to initialize Vulkan backend");
        }

        std::vector<Result> results;
        results.push_back(run_backend("cpu", cpu, 50));
        results.push_back(run_backend("vulkan", vk, 200));
        write_results(results);

        for (const auto & r : results) {
            std::cout << r.backend_name << " mean_ms=" << r.mean_ms
                      << " max_abs=" << r.metrics.max_abs
                      << " mean_abs=" << r.metrics.mean_abs
                      << " allclose_1e_4=" << (r.metrics.allclose_1e_4 ? "true" : "false")
                      << "\n";
        }

        ggml_backend_free(cpu);
        ggml_backend_free(vk);
        return std::all_of(results.begin(), results.end(), [](const Result & r) {
            return r.metrics.allclose_1e_4;
        }) ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
