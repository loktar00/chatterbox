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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int HIDDEN = 1024;
constexpr int N_LAYER = 24;
constexpr int N_HEAD = 16;
constexpr int HEAD_DIM = 64;
constexpr int PAST_LEN = 935;
constexpr int SEQ_LEN = PAST_LEN + 1;
constexpr int FF = 4096;
constexpr int SPEECH_VOCAB = 6563;
constexpr int STEPS = 4;
constexpr float LN_EPS = 1e-5f;
constexpr float ATTN_SCALE = 0.125f;

using clock_type = std::chrono::steady_clock;

std::string fixture_dir() {
    const char * env = std::getenv("GGML_T3_REAL_PROMPT_FIXTURE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return "/root/chatterbox/exports/ggml_t3_real_prompt_multistep_hello_s4_p935";
}

std::string output_stem() {
    const char * env = std::getenv("GGML_T3_REAL_PROMPT_OUTPUT_STEM");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return "hello_s4_p935";
}

std::string output_json() {
    return "/root/chatterbox/exports/benchmarks/ggml_t3_real_prompt_multistep_" + output_stem() + "_2026-07-08.json";
}

std::string output_md() {
    return "/root/chatterbox/exports/benchmarks/ggml_t3_real_prompt_multistep_" + output_stem() + "_2026-07-08.md";
}

struct Comparison {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rms_error = 0.0;
    bool allclose_1e_4 = false;
    bool allclose_1e_3 = false;
    bool allclose_1e_2 = false;
};

struct StepResult {
    int step = 0;
    double ms = 0.0;
    Comparison logits;
    Comparison final_hidden;
    int expected_argmax = -1;
    int actual_argmax = -1;
    int top10_overlap = 0;
    bool ok = false;
};

struct BackendResult {
    std::string backend;
    std::string device;
    bool ok = false;
    double mean_ms = 0.0;
    std::vector<StepResult> steps;
    std::string error;
};

struct LayerTensors {
    ggml_tensor * input_past_k = nullptr;
    ggml_tensor * input_past_v = nullptr;
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
};

struct StackGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * attn_mask = nullptr;
    std::vector<LayerTensors> layers;
    ggml_tensor * ln_f_w = nullptr;
    ggml_tensor * ln_f_b = nullptr;
    ggml_tensor * speech_head_w = nullptr;
    ggml_tensor * speech_head_b = nullptr;
    ggml_tensor * final_hidden = nullptr;
    ggml_tensor * logits = nullptr;
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

ggml_tensor * build_layer(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * attn_mask, const LayerTensors & l) {
    ggml_tensor * ln1 = layer_norm(ctx, cur, l.ln1_w, l.ln1_b);
    ggml_tensor * qkv = linear(ctx, l.c_attn_w, ln1, l.c_attn_b);

    ggml_tensor * q_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 0);
    ggml_tensor * k_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], HIDDEN * sizeof(float));
    ggml_tensor * v_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 2 * HIDDEN * sizeof(float));

    ggml_tensor * q_3d = ggml_reshape_3d(ctx, q_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * k_new_3d = ggml_reshape_3d(ctx, k_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * v_new_3d = ggml_reshape_3d(ctx, v_flat, HEAD_DIM, N_HEAD, 1);

    ggml_tensor * q = ggml_permute(ctx, q_3d, 0, 2, 1, 3);
    ggml_tensor * k_new = ggml_permute(ctx, k_new_3d, 0, 2, 1, 3);
    ggml_tensor * v_new = ggml_permute(ctx, v_new_3d, 1, 2, 0, 3);
    ggml_tensor * k_all = ggml_concat(ctx, l.input_past_k, k_new, 1);
    ggml_tensor * v_all = ggml_concat(ctx, l.input_past_v, v_new, 0);

    ggml_tensor * kq = ggml_mul_mat(ctx, k_all, q);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, kq, attn_mask, ATTN_SCALE, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_all, probs);
    ggml_tensor * merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor * attn_out = ggml_cont_2d(ctx, merged, HIDDEN, 1);
    ggml_tensor * attn_proj = linear(ctx, l.c_proj_w, attn_out, l.c_proj_b);
    ggml_tensor * resid1 = ggml_add(ctx, cur, attn_proj);

    ggml_tensor * ln2 = layer_norm(ctx, resid1, l.ln2_w, l.ln2_b);
    ggml_tensor * fc = linear(ctx, l.c_fc_w, ln2, l.c_fc_b);
    ggml_tensor * act = ggml_gelu(ctx, fc);
    ggml_tensor * mlp = linear(ctx, l.mlp_proj_w, act, l.mlp_proj_b);
    return ggml_add(ctx, resid1, mlp);
}

LayerTensors new_layer_tensors(ggml_context * ctx) {
    LayerTensors l;
    l.input_past_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, PAST_LEN, N_HEAD);
    l.input_past_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PAST_LEN, HEAD_DIM, N_HEAD);
    l.ln1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.c_attn_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, 3 * HIDDEN);
    l.c_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * HIDDEN);
    l.c_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, HIDDEN);
    l.c_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.c_fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, FF);
    l.c_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FF);
    l.mlp_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FF, HIDDEN);
    l.mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    return l;
}

StackGraph build_graph() {
    ggml_init_params params = {
        /* .mem_size   = */ 256ull * 1024ull * 1024ull,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    StackGraph g;
    g.ctx = ctx;
    g.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, 1);
    g.attn_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, SEQ_LEN, 1, 1);
    g.layers.reserve(N_LAYER);
    ggml_tensor * cur = g.input;
    for (int i = 0; i < N_LAYER; ++i) {
        g.layers.push_back(new_layer_tensors(ctx));
        cur = build_layer(ctx, cur, g.attn_mask, g.layers.back());
    }

    g.ln_f_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    g.final_hidden = layer_norm(ctx, cur, g.ln_f_w, g.ln_f_b);
    g.speech_head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, SPEECH_VOCAB);
    g.speech_head_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SPEECH_VOCAB);
    g.logits = linear(ctx, g.speech_head_w, g.final_hidden, g.speech_head_b);
    g.graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(g.graph, g.logits);
    return g;
}

void load_layer_weights(LayerTensors & l, int i) {
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "layer_%02d", i);
    const std::string p(prefix);
    set_tensor_file(l.ln1_w, p + "_ln1_weight", HIDDEN);
    set_tensor_file(l.ln1_b, p + "_ln1_bias", HIDDEN);
    set_tensor_file(l.c_attn_w, p + "_attn_c_attn_weight", HIDDEN * 3 * HIDDEN);
    set_tensor_file(l.c_attn_b, p + "_attn_c_attn_bias", 3 * HIDDEN);
    set_tensor_file(l.c_proj_w, p + "_attn_c_proj_weight", HIDDEN * HIDDEN);
    set_tensor_file(l.c_proj_b, p + "_attn_c_proj_bias", HIDDEN);
    set_tensor_file(l.ln2_w, p + "_ln2_weight", HIDDEN);
    set_tensor_file(l.ln2_b, p + "_ln2_bias", HIDDEN);
    set_tensor_file(l.c_fc_w, p + "_mlp_c_fc_weight", HIDDEN * FF);
    set_tensor_file(l.c_fc_b, p + "_mlp_c_fc_bias", FF);
    set_tensor_file(l.mlp_proj_w, p + "_mlp_c_proj_weight", FF * HIDDEN);
    set_tensor_file(l.mlp_proj_b, p + "_mlp_c_proj_bias", HIDDEN);
}

void load_static_weights(StackGraph & g) {
    for (int i = 0; i < N_LAYER; ++i) {
        load_layer_weights(g.layers[i], i);
    }
    set_tensor_file(g.ln_f_w, "ln_f_weight", HIDDEN);
    set_tensor_file(g.ln_f_b, "ln_f_bias", HIDDEN);
    set_tensor_file(g.speech_head_w, "speech_head_weight_t", HIDDEN * SPEECH_VOCAB);
    set_tensor_file(g.speech_head_b, "speech_head_bias", SPEECH_VOCAB);
}

std::string step_name(int step, const std::string & suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "step_%02d_%s", step, suffix.c_str());
    return buf;
}

void load_step_inputs(StackGraph & g, int step) {
    set_tensor_file(g.input, step_name(step, "input_hidden"), HIDDEN);
    set_tensor_file(g.attn_mask, step_name(step, "attn_mask"), SEQ_LEN);
    for (int i = 0; i < N_LAYER; ++i) {
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "step_%02d_layer_%02d", step, i);
        const std::string p(prefix);
        set_tensor_file(g.layers[i].input_past_k, p + "_past_k", HEAD_DIM * PAST_LEN * N_HEAD);
        set_tensor_file(g.layers[i].input_past_v, p + "_past_v", PAST_LEN * HEAD_DIM * N_HEAD);
    }
}

Comparison compare_arrays(const std::vector<float> & actual, const std::vector<float> & expected) {
    Comparison c;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    bool all_1e_4 = true;
    bool all_1e_3 = true;
    bool all_1e_2 = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double ref = std::abs(static_cast<double>(expected[i]));
        c.max_abs_error = std::max(c.max_abs_error, diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        all_1e_4 = all_1e_4 && diff <= (1e-4 + 1e-4 * ref);
        all_1e_3 = all_1e_3 && diff <= (1e-3 + 1e-3 * ref);
        all_1e_2 = all_1e_2 && diff <= (1e-2 + 1e-2 * ref);
    }
    c.mean_abs_error = sum_abs / static_cast<double>(actual.size());
    c.rms_error = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    c.allclose_1e_4 = all_1e_4;
    c.allclose_1e_3 = all_1e_3;
    c.allclose_1e_2 = all_1e_2;
    return c;
}

int argmax(const std::vector<float> & values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

std::vector<int> topk(const std::vector<float> & values, int k) {
    std::vector<int> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return values[a] > values[b];
    });
    idx.resize(k);
    return idx;
}

int overlap(const std::vector<int> & a, const std::vector<int> & b) {
    int n = 0;
    for (int x : a) {
        if (std::find(b.begin(), b.end(), x) != b.end()) {
            ++n;
        }
    }
    return n;
}

StepResult run_step(ggml_backend_t backend, StackGraph & g, int step) {
    StepResult r;
    r.step = step;
    std::vector<float> ref_final = read_f32(path_for(step_name(step, "ref_final_hidden")), HIDDEN);
    std::vector<float> ref_logits = read_f32(path_for(step_name(step, "ref_logits")), SPEECH_VOCAB);
    r.expected_argmax = argmax(ref_logits);

    load_step_inputs(g, step);
    ggml_backend_synchronize(backend);
    const auto t0 = clock_type::now();
    ggml_status status = ggml_backend_graph_compute(backend, g.graph);
    ggml_backend_synchronize(backend);
    const auto t1 = clock_type::now();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
    }
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<float> actual_final(HIDDEN);
    std::vector<float> actual_logits(SPEECH_VOCAB);
    ggml_backend_tensor_get(g.final_hidden, actual_final.data(), 0, actual_final.size() * sizeof(float));
    ggml_backend_tensor_get(g.logits, actual_logits.data(), 0, actual_logits.size() * sizeof(float));
    r.final_hidden = compare_arrays(actual_final, ref_final);
    r.logits = compare_arrays(actual_logits, ref_logits);
    r.actual_argmax = argmax(actual_logits);
    r.top10_overlap = overlap(topk(actual_logits, 10), topk(ref_logits, 10));
    r.ok = r.actual_argmax == r.expected_argmax && r.top10_overlap >= 8 && r.logits.allclose_1e_2;
    return r;
}

BackendResult run_backend(const std::string & label, ggml_backend_t backend) {
    BackendResult result;
    result.backend = label;
    result.device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    try {
        StackGraph g = build_graph();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate backend tensor buffer");
        }
        if (label == "vulkan") {
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        load_static_weights(g);
        result.steps.reserve(STEPS);
        double total = 0.0;
        for (int step = 0; step < STEPS; ++step) {
            StepResult r = run_step(backend, g, step);
            total += r.ms;
            result.steps.push_back(r);
        }
        result.mean_ms = total / static_cast<double>(STEPS);
        result.ok = std::all_of(result.steps.begin(), result.steps.end(), [](const StepResult & r) { return r.ok; });
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

void write_outputs(const std::vector<BackendResult> & results) {
    {
        std::ofstream f(output_json());
        f << "{\n";
        f << "  \"description\": \"Real-prompt multi-step Chatterbox T3 ggml/Vulkan validation with padded p935 cache and mask\",\n";
        f << "  \"fixture_manifest\": \"" << fixture_dir() << "/manifest.json\",\n";
        f << "  \"steps\": " << STEPS << ",\n";
        f << "  \"max_len\": " << PAST_LEN << ",\n";
        f << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            f << "    {\n";
            f << "      \"backend\": \"" << json_escape(r.backend) << "\",\n";
            f << "      \"device\": \"" << json_escape(r.device) << "\",\n";
            f << "      \"ok\": " << (r.ok ? "true" : "false") << ",\n";
            f << "      \"mean_ms\": " << r.mean_ms << ",\n";
            f << "      \"error\": \"" << json_escape(r.error) << "\",\n";
            f << "      \"steps\": [\n";
            for (size_t j = 0; j < r.steps.size(); ++j) {
                const auto & s = r.steps[j];
                f << "        {\"step\": " << s.step
                  << ", \"ms\": " << s.ms
                  << ", \"ok\": " << (s.ok ? "true" : "false")
                  << ", \"logits_max_abs_error\": " << s.logits.max_abs_error
                  << ", \"logits_allclose_1e_3\": " << (s.logits.allclose_1e_3 ? "true" : "false")
                  << ", \"expected_argmax\": " << s.expected_argmax
                  << ", \"actual_argmax\": " << s.actual_argmax
                  << ", \"top10_overlap\": " << s.top10_overlap
                  << "}" << (j + 1 == r.steps.size() ? "\n" : ",\n");
            }
            f << "      ]\n";
            f << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }
    {
        std::ofstream f(output_md());
        f << "# ggml Real-Prompt Multi-Step T3 Probe\n\n";
        f << "Fixture: `" << fixture_dir() << "/manifest.json`\n\n";
        f << "This validates real prompt caches and real speech-token inputs with a fixed p935 padded cache and attention mask. Cache tensors are loaded per step; this is not yet a device-resident cache-update loop.\n\n";
        f << "| Backend | Device | Mean ms/step | Steps ok | Status |\n";
        f << "|---|---|---:|---:|---|\n";
        for (const auto & r : results) {
            int ok_steps = 0;
            for (const auto & s : r.steps) {
                ok_steps += s.ok ? 1 : 0;
            }
            f << "| " << r.backend << " | " << r.device << " | " << r.mean_ms << " | " << ok_steps << "/" << r.steps.size() << " | " << (r.ok ? "ok" : r.error) << " |\n";
        }
        f << "\n| Backend | Step | ms | Logits max abs | Argmax | Top10 overlap | Status |\n";
        f << "|---|---:|---:|---:|---:|---:|---|\n";
        for (const auto & r : results) {
            for (const auto & s : r.steps) {
                f << "| " << r.backend << " | " << s.step << " | " << s.ms << " | " << s.logits.max_abs_error << " | " << s.actual_argmax << "/" << s.expected_argmax << " | " << s.top10_overlap << "/10 | " << (s.ok ? "ok" : "fail") << " |\n";
            }
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
    std::vector<BackendResult> results;
    results.push_back(run_backend("cpu", cpu));
    results.push_back(run_backend("vulkan", vk));
    write_outputs(results);
    std::cout << "Wrote " << output_json() << "\n";
    std::cout << "Wrote " << output_md() << "\n\n";
    for (const auto & r : results) {
        int ok_steps = 0;
        for (const auto & s : r.steps) {
            ok_steps += s.ok ? 1 : 0;
        }
        std::cout << r.backend << " device=\"" << r.device << "\" mean_ms=" << r.mean_ms
                  << " ok_steps=" << ok_steps << "/" << r.steps.size()
                  << " status=" << (r.ok ? "ok" : r.error) << "\n";
    }
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    return std::all_of(results.begin(), results.end(), [](const BackendResult & r) { return r.ok; }) ? 0 : 1;
}
