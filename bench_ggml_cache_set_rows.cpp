#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int HEAD_DIM = 64;
constexpr int MAX_LEN = 935;
constexpr int N_HEAD = 16;
constexpr int SLOT = 423;
constexpr int N_ELEM = HEAD_DIM * MAX_LEN * N_HEAD;
using clock_type = std::chrono::steady_clock;

struct Result {
    std::string backend;
    std::string device;
    bool ok = false;
    double mean_ms = 0.0;
    double max_abs_error = 0.0;
    int bad_count = 0;
    std::string error;
};

std::vector<float> make_cache() {
    std::vector<float> data(N_ELEM);
    for (int h = 0; h < N_HEAD; ++h) {
        for (int t = 0; t < MAX_LEN; ++t) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                const int i = d + HEAD_DIM * (t + MAX_LEN * h);
                data[i] = 0.001f * static_cast<float>(d) + 0.00001f * static_cast<float>(t) + 0.01f * static_cast<float>(h);
            }
        }
    }
    return data;
}

std::vector<float> make_slot() {
    std::vector<float> data(HEAD_DIM * N_HEAD);
    for (int h = 0; h < N_HEAD; ++h) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            const int i = d + HEAD_DIM * h;
            data[i] = 10.0f + 0.1f * static_cast<float>(h) + 0.001f * static_cast<float>(d);
        }
    }
    return data;
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * cache = nullptr;
    ggml_tensor * slot = nullptr;
    ggml_tensor * index = nullptr;
    ggml_tensor * updated = nullptr;
};

Graph build_graph() {
    ggml_init_params params = {
        /* .mem_size   = */ 8ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    Graph g;
    g.ctx = ctx;
    g.cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, MAX_LEN, N_HEAD);
    g.slot = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, 1, N_HEAD);
    g.index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    g.updated = ggml_set_rows(ctx, g.cache, g.slot, g.index);
    g.graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(g.graph, g.updated);
    return g;
}

Result run_backend(const std::string & label, ggml_backend_t backend) {
    Result r;
    r.backend = label;
    r.device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    try {
        Graph g = build_graph();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate backend tensor buffer");
        }
        auto cache = make_cache();
        auto slot = make_slot();
        int32_t index = SLOT;
        ggml_backend_tensor_set(g.cache, cache.data(), 0, cache.size() * sizeof(float));
        ggml_backend_tensor_set(g.slot, slot.data(), 0, slot.size() * sizeof(float));
        ggml_backend_tensor_set(g.index, &index, 0, sizeof(index));
        ggml_backend_synchronize(backend);

        for (int i = 0; i < 3; ++i) {
            ggml_status status = ggml_backend_graph_compute(backend, g.graph);
            ggml_backend_synchronize(backend);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("warmup failed: ") + ggml_status_to_string(status));
            }
        }

        constexpr int repeats = 50;
        double total_ms = 0.0;
        for (int i = 0; i < repeats; ++i) {
            ggml_backend_tensor_set(g.cache, cache.data(), 0, cache.size() * sizeof(float));
            ggml_backend_tensor_set(g.index, &index, 0, sizeof(index));
            ggml_backend_synchronize(backend);
            const auto t0 = clock_type::now();
            ggml_status status = ggml_backend_graph_compute(backend, g.graph);
            ggml_backend_synchronize(backend);
            const auto t1 = clock_type::now();
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
            }
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        r.mean_ms = total_ms / static_cast<double>(repeats);

        std::vector<float> actual(N_ELEM);
        ggml_backend_tensor_get(g.cache, actual.data(), 0, actual.size() * sizeof(float));
        auto expected = cache;
        for (int h = 0; h < N_HEAD; ++h) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                expected[d + HEAD_DIM * (SLOT + MAX_LEN * h)] = slot[d + HEAD_DIM * h];
            }
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            r.max_abs_error = std::max(r.max_abs_error, diff);
            if (diff > 1e-6) {
                ++r.bad_count;
            }
        }
        r.ok = r.bad_count == 0;
        ggml_backend_buffer_free(buffer);
        ggml_free(g.ctx);
    } catch (const std::exception & e) {
        r.ok = false;
        r.error = e.what();
    }
    return r;
}

std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '\\') out << "\\\\";
        else if (c == '"') out << "\\\"";
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

void write_outputs(const std::vector<Result> & results) {
    const std::string json_path = "/root/chatterbox/exports/benchmarks/ggml_cache_set_rows_p935_2026-07-08.json";
    const std::string md_path = "/root/chatterbox/exports/benchmarks/ggml_cache_set_rows_p935_2026-07-08.md";
    {
        std::ofstream f(json_path);
        f << "{\n";
        f << "  \"description\": \"ggml set_rows p935 cache slot update probe\",\n";
        f << "  \"slot\": " << SLOT << ",\n";
        f << "  \"shape\": [64, 935, 16],\n";
        f << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            f << "    {\"backend\": \"" << json_escape(r.backend)
              << "\", \"device\": \"" << json_escape(r.device)
              << "\", \"ok\": " << (r.ok ? "true" : "false")
              << ", \"mean_ms\": " << r.mean_ms
              << ", \"max_abs_error\": " << r.max_abs_error
              << ", \"bad_count\": " << r.bad_count
              << ", \"error\": \"" << json_escape(r.error) << "\"}"
              << (i + 1 == results.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }
    {
        std::ofstream f(md_path);
        f << "# ggml set_rows Cache Update Probe\n\n";
        f << "Shape: `[64, 935, 16]`; updated slot: `" << SLOT << "`.\n\n";
        f << "| Backend | Device | Mean ms | Max abs error | Bad count | Status |\n";
        f << "|---|---|---:|---:|---:|---|\n";
        for (const auto & r : results) {
            f << "| " << r.backend << " | " << r.device << " | " << r.mean_ms << " | "
              << r.max_abs_error << " | " << r.bad_count << " | " << (r.ok ? "ok" : r.error) << " |\n";
        }
    }
    std::cout << "Wrote " << json_path << "\n";
    std::cout << "Wrote " << md_path << "\n";
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
    std::vector<Result> results;
    results.push_back(run_backend("cpu", cpu));
    results.push_back(run_backend("vulkan", vk));
    write_outputs(results);
    for (const auto & r : results) {
        std::cout << r.backend << " mean_ms=" << r.mean_ms << " bad_count=" << r.bad_count
                  << " status=" << (r.ok ? "ok" : r.error) << "\n";
    }
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    return std::all_of(results.begin(), results.end(), [](const Result & r) { return r.ok; }) ? 0 : 1;
}
