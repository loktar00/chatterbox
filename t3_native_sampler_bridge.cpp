#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

struct TokenScore {
    int token;
    float score;
};

int sample_impl(
    const float * logits,
    int vocab,
    const int32_t * input_ids,
    int input_len,
    uint64_t seed,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int32_t * out_token
) {
    if (logits == nullptr || out_token == nullptr || vocab <= 0 || top_k <= 0 || temperature <= 0.0f) {
        return -1;
    }
    const int k = std::min(top_k, vocab);
    std::vector<TokenScore> top;
    top.reserve(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        top.push_back({i, logits[i] / temperature});
    }
    std::nth_element(
        top.begin(),
        top.begin() + k,
        top.end(),
        [](const TokenScore & a, const TokenScore & b) {
            if (a.score == b.score) {
                return a.token < b.token;
            }
            return a.score > b.score;
        }
    );
    top.resize(static_cast<size_t>(k));

    std::vector<int> order(static_cast<size_t>(k));
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            if (top[static_cast<size_t>(a)].score == top[static_cast<size_t>(b)].score) {
                return top[static_cast<size_t>(a)].token < top[static_cast<size_t>(b)].token;
            }
            return top[static_cast<size_t>(a)].score < top[static_cast<size_t>(b)].score;
        }
    );

    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    std::vector<double> weights(static_cast<size_t>(k));
    double denom = 0.0;
    for (int i = 0; i < k; ++i) {
        weights[static_cast<size_t>(i)] = std::exp(static_cast<double>(top[static_cast<size_t>(i)].score - max_score));
        denom += weights[static_cast<size_t>(i)];
    }

    std::vector<uint8_t> remove(static_cast<size_t>(k), 0);
    double cumulative = 0.0;
    const double threshold = 1.0 - static_cast<double>(top_p);
    for (int pos = 0; pos < k; ++pos) {
        const int idx = order[static_cast<size_t>(pos)];
        cumulative += weights[static_cast<size_t>(idx)] / denom;
        if (cumulative <= threshold) {
            remove[static_cast<size_t>(idx)] = 1;
        }
    }
    remove[static_cast<size_t>(order[static_cast<size_t>(k - 1)])] = 0;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        if (remove[static_cast<size_t>(i)]) {
            top[static_cast<size_t>(i)].score = neg_inf;
        }
    }

    if (repetition_penalty != 1.0f && input_ids != nullptr && input_len > 0) {
        std::vector<uint8_t> seen(static_cast<size_t>(vocab), 0);
        for (int i = 0; i < input_len; ++i) {
            const int token = static_cast<int>(input_ids[i]);
            if (0 <= token && token < vocab) {
                seen[static_cast<size_t>(token)] = 1;
            }
        }
        for (auto & item : top) {
            if (item.score == neg_inf || !seen[static_cast<size_t>(item.token)]) {
                continue;
            }
            if (item.score < 0.0f) {
                item.score *= repetition_penalty;
            } else {
                item.score /= repetition_penalty;
            }
        }
    }

    max_score = neg_inf;
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    if (max_score == neg_inf) {
        return 1;
    }

    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        if (top[static_cast<size_t>(i)].score == neg_inf) {
            weights[static_cast<size_t>(i)] = 0.0;
        } else {
            weights[static_cast<size_t>(i)] = std::exp(static_cast<double>(top[static_cast<size_t>(i)].score - max_score));
            sum += weights[static_cast<size_t>(i)];
        }
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, sum);
    const double target = uniform(rng);
    cumulative = 0.0;
    int sampled = top[static_cast<size_t>(k - 1)].token;
    for (int i = 0; i < k; ++i) {
        cumulative += weights[static_cast<size_t>(i)];
        if (target <= cumulative) {
            sampled = top[static_cast<size_t>(i)].token;
            break;
        }
    }
    *out_token = static_cast<int32_t>(sampled);
    return 0;
}

int filter_probs_impl(
    const float * logits,
    int vocab,
    const int32_t * input_ids,
    int input_len,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int32_t * out_tokens,
    float * out_probs,
    int max_out,
    int32_t * out_count
) {
    if (
        logits == nullptr || out_tokens == nullptr || out_probs == nullptr || out_count == nullptr || vocab <= 0
        || top_k <= 0 || temperature <= 0.0f || max_out <= 0
    ) {
        return -1;
    }
    const int k = std::min(std::min(top_k, vocab), max_out);
    std::vector<TokenScore> top;
    top.reserve(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        top.push_back({i, logits[i] / temperature});
    }
    std::nth_element(
        top.begin(),
        top.begin() + k,
        top.end(),
        [](const TokenScore & a, const TokenScore & b) {
            if (a.score == b.score) {
                return a.token < b.token;
            }
            return a.score > b.score;
        }
    );
    top.resize(static_cast<size_t>(k));

    std::vector<int> order(static_cast<size_t>(k));
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            if (top[static_cast<size_t>(a)].score == top[static_cast<size_t>(b)].score) {
                return top[static_cast<size_t>(a)].token < top[static_cast<size_t>(b)].token;
            }
            return top[static_cast<size_t>(a)].score < top[static_cast<size_t>(b)].score;
        }
    );

    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    std::vector<double> weights(static_cast<size_t>(k));
    double denom = 0.0;
    for (int i = 0; i < k; ++i) {
        weights[static_cast<size_t>(i)] = std::exp(static_cast<double>(top[static_cast<size_t>(i)].score - max_score));
        denom += weights[static_cast<size_t>(i)];
    }

    std::vector<uint8_t> remove(static_cast<size_t>(k), 0);
    double cumulative = 0.0;
    const double threshold = 1.0 - static_cast<double>(top_p);
    for (int pos = 0; pos < k; ++pos) {
        const int idx = order[static_cast<size_t>(pos)];
        cumulative += weights[static_cast<size_t>(idx)] / denom;
        if (cumulative <= threshold) {
            remove[static_cast<size_t>(idx)] = 1;
        }
    }
    remove[static_cast<size_t>(order[static_cast<size_t>(k - 1)])] = 0;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        if (remove[static_cast<size_t>(i)]) {
            top[static_cast<size_t>(i)].score = neg_inf;
        }
    }

    if (repetition_penalty != 1.0f && input_ids != nullptr && input_len > 0) {
        std::vector<uint8_t> seen(static_cast<size_t>(vocab), 0);
        for (int i = 0; i < input_len; ++i) {
            const int token = static_cast<int>(input_ids[i]);
            if (0 <= token && token < vocab) {
                seen[static_cast<size_t>(token)] = 1;
            }
        }
        for (auto & item : top) {
            if (item.score == neg_inf || !seen[static_cast<size_t>(item.token)]) {
                continue;
            }
            if (item.score < 0.0f) {
                item.score *= repetition_penalty;
            } else {
                item.score /= repetition_penalty;
            }
        }
    }

    max_score = neg_inf;
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    if (max_score == neg_inf) {
        *out_count = 0;
        return 1;
    }

    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        if (top[static_cast<size_t>(i)].score == neg_inf) {
            weights[static_cast<size_t>(i)] = 0.0;
        } else {
            weights[static_cast<size_t>(i)] = std::exp(static_cast<double>(top[static_cast<size_t>(i)].score - max_score));
            sum += weights[static_cast<size_t>(i)];
        }
    }

    int count = 0;
    for (int i = 0; i < k; ++i) {
        if (weights[static_cast<size_t>(i)] <= 0.0) {
            continue;
        }
        out_tokens[count] = static_cast<int32_t>(top[static_cast<size_t>(i)].token);
        out_probs[count] = static_cast<float>(weights[static_cast<size_t>(i)] / sum);
        ++count;
    }
    *out_count = static_cast<int32_t>(count);
    return 0;
}

} // namespace

extern "C" int cb_t3_native_sample(
    const float * logits,
    int vocab,
    const int32_t * input_ids,
    int input_len,
    uint64_t seed,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int32_t * out_token
) {
    return sample_impl(
        logits,
        vocab,
        input_ids,
        input_len,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        out_token
    );
}

extern "C" int cb_t3_native_filter_probs(
    const float * logits,
    int vocab,
    const int32_t * input_ids,
    int input_len,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    int32_t * out_tokens,
    float * out_probs,
    int max_out,
    int32_t * out_count
) {
    return filter_probs_impl(
        logits,
        vocab,
        input_ids,
        input_len,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        out_tokens,
        out_probs,
        max_out,
        out_count
    );
}
