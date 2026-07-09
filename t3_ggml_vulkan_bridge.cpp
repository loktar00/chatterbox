#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
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
constexpr float LN_EPS = 1e-5f;
constexpr float ATTN_SCALE = 0.125f;
#ifdef CB_T3_F16_WEIGHTS
constexpr ggml_type WEIGHT_TYPE = GGML_TYPE_F16;
#else
constexpr ggml_type WEIGHT_TYPE = GGML_TYPE_F32;
#endif
using clock_type = std::chrono::steady_clock;

void set_error(char * err, size_t err_len, const std::string & msg) {
    if (err == nullptr || err_len == 0) {
        return;
    }
    std::strncpy(err, msg.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
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

struct Layer {
    ggml_tensor * cache_k = nullptr;
    ggml_tensor * cache_v = nullptr;
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
    ggml_tensor * logits = nullptr;
};

struct Bridge {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    Graph g;
    std::string weights_dir;
    std::string device;
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
    l.c_attn_w = ggml_new_tensor_2d(ctx, WEIGHT_TYPE, HIDDEN, 3 * HIDDEN);
    l.c_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * HIDDEN);
    l.c_proj_w = ggml_new_tensor_2d(ctx, WEIGHT_TYPE, HIDDEN, HIDDEN);
    l.c_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    l.c_fc_w = ggml_new_tensor_2d(ctx, WEIGHT_TYPE, HIDDEN, FF);
    l.c_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FF);
    l.mlp_proj_w = ggml_new_tensor_2d(ctx, WEIGHT_TYPE, FF, HIDDEN);
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
    ggml_tensor * k_new = ggml_permute(ctx, k_3d, 0, 2, 1, 3);
    ggml_tensor * v_new = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
    ggml_tensor * v_new_attn = ggml_permute(ctx, v_3d, 1, 2, 0, 3);
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
    ggml_tensor * final_hidden = layer_norm(g.ctx, cur, g.ln_f_w, g.ln_f_b);
    g.speech_head_w = ggml_new_tensor_2d(g.ctx, WEIGHT_TYPE, HIDDEN, SPEECH_VOCAB);
    g.speech_head_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, SPEECH_VOCAB);
    g.logits = linear(g.ctx, g.speech_head_w, final_hidden, g.speech_head_b);
    g.graph = ggml_new_graph_custom(g.ctx, 4096, false);
    ggml_build_forward_expand(g.graph, g.logits);
    for (ggml_tensor * update : g.updates) {
        ggml_build_forward_expand(g.graph, update);
    }
    return g;
}

std::string file_path(const Bridge * b, const std::string & name) {
    return b->weights_dir + "/" + name + ".f32";
}

void set_tensor_file(const Bridge * b, ggml_tensor * t, const std::string & name, size_t count) {
    std::vector<float> data = read_f32(file_path(b, name), count);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data_f16(count);
        ggml_fp32_to_fp16_row(data.data(), data_f16.data(), static_cast<int64_t>(count));
        ggml_backend_tensor_set(t, data_f16.data(), 0, data_f16.size() * sizeof(ggml_fp16_t));
    } else {
        throw std::runtime_error("unsupported tensor type while loading " + name);
    }
}

void load_layer_weights(const Bridge * b, Layer & l, int i) {
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "layer_%02d", i);
    const std::string p(prefix);
    set_tensor_file(b, l.ln1_w, p + "_ln1_weight", HIDDEN);
    set_tensor_file(b, l.ln1_b, p + "_ln1_bias", HIDDEN);
    set_tensor_file(b, l.c_attn_w, p + "_attn_c_attn_weight", HIDDEN * 3 * HIDDEN);
    set_tensor_file(b, l.c_attn_b, p + "_attn_c_attn_bias", 3 * HIDDEN);
    set_tensor_file(b, l.c_proj_w, p + "_attn_c_proj_weight", HIDDEN * HIDDEN);
    set_tensor_file(b, l.c_proj_b, p + "_attn_c_proj_bias", HIDDEN);
    set_tensor_file(b, l.ln2_w, p + "_ln2_weight", HIDDEN);
    set_tensor_file(b, l.ln2_b, p + "_ln2_bias", HIDDEN);
    set_tensor_file(b, l.c_fc_w, p + "_mlp_c_fc_weight", HIDDEN * FF);
    set_tensor_file(b, l.c_fc_b, p + "_mlp_c_fc_bias", FF);
    set_tensor_file(b, l.mlp_proj_w, p + "_mlp_c_proj_weight", FF * HIDDEN);
    set_tensor_file(b, l.mlp_proj_b, p + "_mlp_c_proj_bias", HIDDEN);
}

void load_weights(Bridge * b) {
    for (int i = 0; i < N_LAYER; ++i) {
        load_layer_weights(b, b->g.layers[i], i);
    }
    set_tensor_file(b, b->g.ln_f_w, "ln_f_weight", HIDDEN);
    set_tensor_file(b, b->g.ln_f_b, "ln_f_bias", HIDDEN);
    set_tensor_file(b, b->g.speech_head_w, "speech_head_weight_t", HIDDEN * SPEECH_VOCAB);
    set_tensor_file(b, b->g.speech_head_b, "speech_head_bias", SPEECH_VOCAB);
}

void destroy_bridge(Bridge * b) {
    if (b == nullptr) {
        return;
    }
    if (b->buffer != nullptr) {
        ggml_backend_buffer_free(b->buffer);
        b->buffer = nullptr;
    }
    if (b->g.ctx != nullptr) {
        ggml_free(b->g.ctx);
        b->g.ctx = nullptr;
    }
    if (b->backend != nullptr) {
        ggml_backend_free(b->backend);
        b->backend = nullptr;
    }
    delete b;
}

void set_layer_cache_range_impl(
    Bridge * b,
    int layer,
    const float * key_segment,
    const float * value_segment,
    int start_pos,
    int segment_len
) {
    if (b == nullptr) {
        throw std::runtime_error("bridge handle is null");
    }
    if (layer < 0 || layer >= N_LAYER) {
        throw std::runtime_error("layer index out of range");
    }
    if (key_segment == nullptr || value_segment == nullptr) {
        throw std::runtime_error("cache segment pointers are required");
    }
    if (start_pos < 0 || segment_len < 0 || start_pos + segment_len > MAX_LEN) {
        throw std::runtime_error("cache segment range out of range");
    }
    const size_t segment_bytes = static_cast<size_t>(segment_len) * HEAD_DIM * sizeof(float);
    const size_t src_head_stride = static_cast<size_t>(segment_len) * HEAD_DIM;
    const size_t dst_head_stride_bytes = static_cast<size_t>(MAX_LEN) * HEAD_DIM * sizeof(float);
    const size_t dst_start_offset_bytes = static_cast<size_t>(start_pos) * HEAD_DIM * sizeof(float);
    for (int head = 0; head < N_HEAD; ++head) {
        const size_t src_offset = static_cast<size_t>(head) * src_head_stride;
        const size_t dst_offset = static_cast<size_t>(head) * dst_head_stride_bytes + dst_start_offset_bytes;
        ggml_backend_tensor_set(
            b->g.layers[layer].cache_k,
            key_segment + src_offset,
            dst_offset,
            segment_bytes
        );
        ggml_backend_tensor_set(
            b->g.layers[layer].cache_v,
            value_segment + src_offset,
            dst_offset,
            segment_bytes
        );
    }
}

} // namespace

extern "C" {

Bridge * cb_t3_ggml_create(
    const char * weights_dir,
    const char * ggml_lib_dir,
    int device_index,
    char * err,
    size_t err_len
) {
    try {
        if (weights_dir == nullptr || weights_dir[0] == '\0') {
            throw std::runtime_error("weights_dir is required");
        }
        if (ggml_lib_dir != nullptr && ggml_lib_dir[0] != '\0') {
            ggml_backend_load_all_from_path(ggml_lib_dir);
        }
        ggml_backend_register(ggml_backend_vk_reg());

        Bridge * b = new Bridge();
        b->weights_dir = weights_dir;
        b->backend = ggml_backend_vk_init(device_index);
        if (b->backend == nullptr) {
            delete b;
            throw std::runtime_error("failed to initialize Vulkan backend");
        }
        b->device = ggml_backend_dev_description(ggml_backend_get_device(b->backend));
        b->g = build_graph();
        b->buffer = ggml_backend_alloc_ctx_tensors(b->g.ctx, b->backend);
        if (b->buffer == nullptr) {
            destroy_bridge(b);
            throw std::runtime_error("failed to allocate Vulkan tensor buffer");
        }
        ggml_backend_buffer_set_usage(b->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        load_weights(b);
        set_error(err, err_len, "");
        return b;
    } catch (const std::exception & e) {
        set_error(err, err_len, e.what());
        return nullptr;
    }
}

void cb_t3_ggml_destroy(Bridge * b) {
    destroy_bridge(b);
}

const char * cb_t3_ggml_device(Bridge * b) {
    if (b == nullptr) {
        return "";
    }
    return b->device.c_str();
}

int cb_t3_ggml_hidden() {
    return HIDDEN;
}

int cb_t3_ggml_layers() {
    return N_LAYER;
}

int cb_t3_ggml_heads() {
    return N_HEAD;
}

int cb_t3_ggml_head_dim() {
    return HEAD_DIM;
}

int cb_t3_ggml_max_len() {
    return MAX_LEN;
}

int cb_t3_ggml_seq_len() {
    return SEQ_LEN;
}

int cb_t3_ggml_speech_vocab() {
    return SPEECH_VOCAB;
}

int cb_t3_ggml_set_layer_cache(
    Bridge * b,
    int layer,
    const float * key_cache,
    const float * value_cache,
    char * err,
    size_t err_len
) {
    try {
        if (b == nullptr) {
            throw std::runtime_error("bridge handle is null");
        }
        if (layer < 0 || layer >= N_LAYER) {
            throw std::runtime_error("layer index out of range");
        }
        if (key_cache == nullptr || value_cache == nullptr) {
            throw std::runtime_error("cache pointers are required");
        }
        const size_t bytes = static_cast<size_t>(HEAD_DIM) * MAX_LEN * N_HEAD * sizeof(float);
        ggml_backend_tensor_set(b->g.layers[layer].cache_k, key_cache, 0, bytes);
        ggml_backend_tensor_set(b->g.layers[layer].cache_v, value_cache, 0, bytes);
        set_error(err, err_len, "");
        return 0;
    } catch (const std::exception & e) {
        set_error(err, err_len, e.what());
        return -1;
    }
}

int cb_t3_ggml_set_layer_cache_prefix(
    Bridge * b,
    int layer,
    const float * key_prefix,
    const float * value_prefix,
    int valid_len,
    char * err,
    size_t err_len
) {
    try {
        set_layer_cache_range_impl(b, layer, key_prefix, value_prefix, 0, valid_len);
        set_error(err, err_len, "");
        return 0;
    } catch (const std::exception & e) {
        set_error(err, err_len, e.what());
        return -1;
    }
}

int cb_t3_ggml_set_layer_cache_range(
    Bridge * b,
    int layer,
    const float * key_segment,
    const float * value_segment,
    int start_pos,
    int segment_len,
    char * err,
    size_t err_len
) {
    try {
        set_layer_cache_range_impl(b, layer, key_segment, value_segment, start_pos, segment_len);
        set_error(err, err_len, "");
        return 0;
    } catch (const std::exception & e) {
        set_error(err, err_len, e.what());
        return -1;
    }
}

int cb_t3_ggml_run_step(
    Bridge * b,
    const float * input_hidden,
    const float * attn_mask,
    int slot_index,
    float * logits_out,
    double * elapsed_ms,
    char * err,
    size_t err_len
) {
    try {
        if (b == nullptr) {
            throw std::runtime_error("bridge handle is null");
        }
        if (input_hidden == nullptr || attn_mask == nullptr || logits_out == nullptr) {
            throw std::runtime_error("input_hidden, attn_mask, and logits_out are required");
        }
        if (slot_index < 0 || slot_index >= MAX_LEN) {
            throw std::runtime_error("slot_index out of range");
        }
        ggml_backend_tensor_set(b->g.input, input_hidden, 0, HIDDEN * sizeof(float));
        ggml_backend_tensor_set(b->g.attn_mask, attn_mask, 0, SEQ_LEN * sizeof(float));
        int32_t slot = static_cast<int32_t>(slot_index);
        ggml_backend_tensor_set(b->g.slot_index, &slot, 0, sizeof(slot));

#ifndef CB_T3_SKIP_PRE_COMPUTE_SYNC
        ggml_backend_synchronize(b->backend);
#endif
        const auto t0 = clock_type::now();
        ggml_status status = ggml_backend_graph_compute(b->backend, b->g.graph);
        ggml_backend_synchronize(b->backend);
        const auto t1 = clock_type::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
        }
        ggml_backend_tensor_get(b->g.logits, logits_out, 0, SPEECH_VOCAB * sizeof(float));
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        set_error(err, err_len, "");
        return 0;
    } catch (const std::exception & e) {
        set_error(err, err_len, e.what());
        return -1;
    }
}

} // extern "C"
