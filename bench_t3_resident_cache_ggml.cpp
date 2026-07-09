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
constexpr int MAX_LEN = 935;
constexpr int SEQ_LEN = MAX_LEN + 1;
constexpr int FF = 4096;
constexpr int SPEECH_VOCAB = 6563;
constexpr int STEPS = 4;
constexpr int INITIAL_CONTEXT_LEN = 423;
constexpr float LN_EPS = 1e-5f;
constexpr float ATTN_SCALE = 0.125f;
using clock_type = std::chrono::steady_clock;

std::string fixture_dir() {
    return "/root/chatterbox/exports/ggml_t3_real_prompt_multistep_chunk270_s4_p935";
}

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

void set_tensor(ggml_tensor * t, const std::vector<float> & data) {
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

void set_tensor_file(ggml_tensor * t, const std::string & name, size_t count) {
    set_tensor(t, read_f32(path_for(name), count));
}

std::vector<float> read_step_v_cache_update_layout(int step, int layer) {
    char name[96];
    std::snprintf(name, sizeof(name), "step_%02d_layer_%02d_past_v", step, layer);
    const auto src = read_f32(path_for(name), MAX_LEN * HEAD_DIM * N_HEAD); // [seq, dim, head]
    std::vector<float> dst(HEAD_DIM * MAX_LEN * N_HEAD);                   // [dim, seq, head]
    for (int h = 0; h < N_HEAD; ++h) {
        for (int t = 0; t < MAX_LEN; ++t) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                dst[d + HEAD_DIM * (t + MAX_LEN * h)] = src[t + MAX_LEN * (d + HEAD_DIM * h)];
            }
        }
    }
    return dst;
}

struct Comparison {
    double max_abs_error = 0.0;
    bool allclose_1e_3 = false;
};

Comparison compare_arrays(const std::vector<float> & actual, const std::vector<float> & expected) {
    Comparison c;
    bool all = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double ref = std::abs(static_cast<double>(expected[i]));
        c.max_abs_error = std::max(c.max_abs_error, diff);
        all = all && diff <= (1e-3 + 1e-3 * ref);
    }
    c.allclose_1e_3 = all;
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

struct Layer {
    ggml_tensor * cache_k = nullptr; // [64, max_len, 16]
    ggml_tensor * cache_v = nullptr; // [64, max_len, 16], permuted for attention
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

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * attn_mask = nullptr;
    ggml_tensor * slot_index = nullptr;
    std::vector<Layer> layers;
    std::vector<ggml_tensor *> updates;
    ggml_tensor * ln_f_w = nullptr;
    ggml_tensor * ln_f_b = nullptr;
    ggml_tensor * speech_head_w = nullptr;
    ggml_tensor * speech_head_b = nullptr;
    ggml_tensor * final_hidden = nullptr;
    ggml_tensor * logits = nullptr;
};

ggml_tensor * repeat_to(ggml_context * ctx, ggml_tensor * src, ggml_tensor * dst_like) {
    return ggml_repeat(ctx, src, dst_like);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    return ggml_add(ctx, y, repeat_to(ctx, bias, y));
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * norm = ggml_norm(ctx, x, LN_EPS);
    ggml_tensor * scaled = ggml_mul(ctx, norm, repeat_to(ctx, weight, norm));
    return ggml_add(ctx, scaled, repeat_to(ctx, bias, norm));
}

Layer new_layer(ggml_context * ctx) {
    Layer l;
    l.cache_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, MAX_LEN, N_HEAD);
    l.cache_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, MAX_LEN, N_HEAD);
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

ggml_tensor * build_layer(Graph & g, ggml_tensor * cur, Layer & l) {
    ggml_context * ctx = g.ctx;
    ggml_tensor * ln1 = layer_norm(ctx, cur, l.ln1_w, l.ln1_b);
    ggml_tensor * qkv = linear(ctx, l.c_attn_w, ln1, l.c_attn_b);
    ggml_tensor * q_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 0);
    ggml_tensor * k_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], HIDDEN * sizeof(float));
    ggml_tensor * v_flat = ggml_view_2d(ctx, qkv, HIDDEN, 1, qkv->nb[1], 2 * HIDDEN * sizeof(float));
    ggml_tensor * q_3d = ggml_reshape_3d(ctx, q_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * k_3d = ggml_reshape_3d(ctx, k_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * v_3d = ggml_reshape_3d(ctx, v_flat, HEAD_DIM, N_HEAD, 1);
    ggml_tensor * q = ggml_permute(ctx, q_3d, 0, 2, 1, 3);
    ggml_tensor * k_new = ggml_permute(ctx, k_3d, 0, 2, 1, 3);      // [64, 1, 16]
    ggml_tensor * v_new = ggml_permute(ctx, v_3d, 0, 2, 1, 3);      // [64, 1, 16], update layout
    ggml_tensor * v_new_attn = ggml_permute(ctx, v_3d, 1, 2, 0, 3); // [1, 64, 16]
    ggml_tensor * k_all = ggml_concat(ctx, l.cache_k, k_new, 1);
    ggml_tensor * v_cache_attn = ggml_permute(ctx, l.cache_v, 1, 0, 2, 3);
    ggml_tensor * v_all = ggml_concat(ctx, v_cache_attn, v_new_attn, 0);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k_all, q), g.attn_mask, ATTN_SCALE, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_all, probs);
    ggml_tensor * attn_out = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), HIDDEN, 1);
    ggml_tensor * resid1 = ggml_add(ctx, cur, linear(ctx, l.c_proj_w, attn_out, l.c_proj_b));
    ggml_tensor * ln2 = layer_norm(ctx, resid1, l.ln2_w, l.ln2_b);
    ggml_tensor * mlp = linear(ctx, l.mlp_proj_w, ggml_gelu(ctx, linear(ctx, l.c_fc_w, ln2, l.c_fc_b)), l.mlp_proj_b);
    g.updates.push_back(ggml_set_rows(ctx, l.cache_k, k_new, g.slot_index));
    g.updates.push_back(ggml_set_rows(ctx, l.cache_v, v_new, g.slot_index));
    return ggml_add(ctx, resid1, mlp);
}

Graph build_graph() {
    ggml_init_params params = { 256ull * 1024ull * 1024ull, nullptr, true };
    Graph g;
    g.ctx = ggml_init(params);
    if (g.ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    g.input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, HIDDEN, 1);
    g.attn_mask = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, SEQ_LEN, 1, 1);
    g.slot_index = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
    g.layers.reserve(N_LAYER);
    ggml_tensor * cur = g.input;
    for (int i = 0; i < N_LAYER; ++i) {
        g.layers.push_back(new_layer(g.ctx));
        cur = build_layer(g, cur, g.layers.back());
    }
    g.ln_f_w = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, HIDDEN);
    g.ln_f_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, HIDDEN);
    g.final_hidden = layer_norm(g.ctx, cur, g.ln_f_w, g.ln_f_b);
    g.speech_head_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, HIDDEN, SPEECH_VOCAB);
    g.speech_head_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, SPEECH_VOCAB);
    g.logits = linear(g.ctx, g.speech_head_w, g.final_hidden, g.speech_head_b);
    g.graph = ggml_new_graph_custom(g.ctx, 4096, false);
    ggml_build_forward_expand(g.graph, g.logits);
    for (ggml_tensor * update : g.updates) {
        ggml_build_forward_expand(g.graph, update);
    }
    return g;
}

void load_layer_weights(Layer & l, int i) {
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

void load_initial_state(Graph & g) {
    for (int i = 0; i < N_LAYER; ++i) {
        load_layer_weights(g.layers[i], i);
        char name[96];
        std::snprintf(name, sizeof(name), "step_00_layer_%02d_past_k", i);
        set_tensor_file(g.layers[i].cache_k, name, HEAD_DIM * MAX_LEN * N_HEAD);
        set_tensor(g.layers[i].cache_v, read_step_v_cache_update_layout(0, i));
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

struct StepResult {
    int step = 0;
    double ms = 0.0;
    double logits_max_abs_error = 0.0;
    int expected_argmax = -1;
    int actual_argmax = -1;
    int top10_overlap = 0;
    bool ok = false;
};

StepResult run_step(ggml_backend_t backend, Graph & g, int step) {
    StepResult r;
    r.step = step;
    set_tensor_file(g.input, step_name(step, "input_hidden"), HIDDEN);
    set_tensor_file(g.attn_mask, step_name(step, "attn_mask"), SEQ_LEN);
    int32_t slot = INITIAL_CONTEXT_LEN + step;
    ggml_backend_tensor_set(g.slot_index, &slot, 0, sizeof(slot));
    auto ref_logits = read_f32(path_for(step_name(step, "ref_logits")), SPEECH_VOCAB);
    r.expected_argmax = argmax(ref_logits);
    ggml_backend_synchronize(backend);
    const auto t0 = clock_type::now();
    ggml_status status = ggml_backend_graph_compute(backend, g.graph);
    ggml_backend_synchronize(backend);
    const auto t1 = clock_type::now();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
    }
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> logits(SPEECH_VOCAB);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, logits.size() * sizeof(float));
    auto cmp = compare_arrays(logits, ref_logits);
    r.logits_max_abs_error = cmp.max_abs_error;
    r.actual_argmax = argmax(logits);
    r.top10_overlap = overlap(topk(logits, 10), topk(ref_logits, 10));
    r.ok = cmp.allclose_1e_3 && r.actual_argmax == r.expected_argmax && r.top10_overlap == 10;
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

void write_outputs(const std::vector<StepResult> & steps, const std::string & device, double mean_ms, bool ok) {
    const std::string json_path = "/root/chatterbox/exports/benchmarks/ggml_t3_resident_cache_chunk270_s4_p935_2026-07-08.json";
    const std::string md_path = "/root/chatterbox/exports/benchmarks/ggml_t3_resident_cache_chunk270_s4_p935_2026-07-08.md";
    {
        std::ofstream f(json_path);
        f << "{\n";
        f << "  \"description\": \"ggml Vulkan resident-cache T3 chunk270 4-step validation\",\n";
        f << "  \"fixture_manifest\": \"" << fixture_dir() << "/manifest.json\",\n";
        f << "  \"device\": \"" << json_escape(device) << "\",\n";
        f << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        f << "  \"mean_ms\": " << mean_ms << ",\n";
        f << "  \"steps\": [\n";
        for (size_t i = 0; i < steps.size(); ++i) {
            const auto & s = steps[i];
            f << "    {\"step\": " << s.step << ", \"ms\": " << s.ms
              << ", \"ok\": " << (s.ok ? "true" : "false")
              << ", \"logits_max_abs_error\": " << s.logits_max_abs_error
              << ", \"expected_argmax\": " << s.expected_argmax
              << ", \"actual_argmax\": " << s.actual_argmax
              << ", \"top10_overlap\": " << s.top10_overlap << "}"
              << (i + 1 == steps.size() ? "\n" : ",\n");
        }
        f << "  ]\n";
        f << "}\n";
    }
    {
        std::ofstream f(md_path);
        f << "# ggml Vulkan Resident-Cache T3 Probe\n\n";
        f << "Fixture: `" << fixture_dir() << "/manifest.json`\n\n";
        f << "This loads chunk270 step-0 KV cache once, then updates K/V slots on device with `ggml_set_rows` for subsequent steps.\n\n";
        f << "| Device | Mean ms/step | Steps ok | Status |\n";
        f << "|---|---:|---:|---|\n";
        int ok_steps = 0;
        for (const auto & s : steps) ok_steps += s.ok ? 1 : 0;
        f << "| " << device << " | " << mean_ms << " | " << ok_steps << "/" << steps.size() << " | " << (ok ? "ok" : "fail") << " |\n\n";
        f << "| Step | ms | Logits max abs | Argmax | Top10 overlap | Status |\n";
        f << "|---:|---:|---:|---:|---:|---|\n";
        for (const auto & s : steps) {
            f << "| " << s.step << " | " << s.ms << " | " << s.logits_max_abs_error << " | "
              << s.actual_argmax << "/" << s.expected_argmax << " | " << s.top10_overlap << "/10 | " << (s.ok ? "ok" : "fail") << " |\n";
        }
    }
    std::cout << "Wrote " << json_path << "\n";
    std::cout << "Wrote " << md_path << "\n";
}

} // namespace

int main() {
    ggml_backend_load_all_from_path("/root/llama.cpp/build-vulkan/bin");
    ggml_backend_register(ggml_backend_vk_reg());
    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (vk == nullptr) {
        std::cerr << "failed to initialize Vulkan backend\n";
        return 2;
    }
    bool ok = false;
    std::vector<StepResult> steps;
    double mean_ms = 0.0;
    std::string device = ggml_backend_dev_description(ggml_backend_get_device(vk));
    try {
        Graph g = build_graph();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, vk);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Vulkan tensor buffer");
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        load_initial_state(g);
        double total = 0.0;
        for (int step = 0; step < STEPS; ++step) {
            StepResult r = run_step(vk, g, step);
            total += r.ms;
            steps.push_back(r);
        }
        mean_ms = total / static_cast<double>(steps.size());
        ok = std::all_of(steps.begin(), steps.end(), [](const StepResult & r) { return r.ok; });
        ggml_backend_buffer_free(buffer);
        ggml_free(g.ctx);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        ok = false;
    }
    write_outputs(steps, device, mean_ms, ok);
    for (const auto & s : steps) {
        std::cout << "step=" << s.step << " ms=" << s.ms << " argmax=" << s.actual_argmax << "/" << s.expected_argmax
                  << " top10=" << s.top10_overlap << "/10 status=" << (s.ok ? "ok" : "fail") << "\n";
    }
    std::cout << "mean_ms=" << mean_ms << " status=" << (ok ? "ok" : "fail") << "\n";
    ggml_backend_free(vk);
    return ok ? 0 : 1;
}
