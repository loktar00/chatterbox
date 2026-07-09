#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct TokenScore {
    int token;
    float score;
};

struct Timings {
    double topk_ms = 0.0;
    double top_p_ms = 0.0;
    double repetition_ms = 0.0;
    double softmax_sample_ms = 0.0;
    double total_ms = 0.0;
};

struct CaseResult {
    int seen_len = 0;
    int unique_len = 0;
    int vocab = 0;
    int trials = 0;
    Timings timings;
    double checksum = 0.0;
};

double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

int sample_native(
    const std::vector<float> & logits,
    const std::vector<uint8_t> & seen_mask,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    std::mt19937 & rng,
    Timings & timings
) {
    const int vocab = static_cast<int>(logits.size());
    const int k = std::min(top_k, vocab);
    const auto total_start = clock_type::now();

    auto start = clock_type::now();
    std::vector<TokenScore> top;
    top.reserve(vocab);
    for (int i = 0; i < vocab; ++i) {
        top.push_back({i, logits[i] / temperature});
    }
    auto nth = top.begin() + k;
    std::nth_element(
        top.begin(),
        nth,
        top.end(),
        [](const TokenScore & a, const TokenScore & b) {
            if (a.score == b.score) {
                return a.token < b.token;
            }
            return a.score > b.score;
        }
    );
    top.resize(k);
    auto end = clock_type::now();
    timings.topk_ms += elapsed_ms(start, end);

    start = clock_type::now();
    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            if (top[a].score == top[b].score) {
                return top[a].token < top[b].token;
            }
            return top[a].score < top[b].score;
        }
    );
    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    double denom = 0.0;
    std::vector<double> probs(k);
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(static_cast<double>(top[i].score - max_score));
        denom += probs[i];
    }
    double cumulative = 0.0;
    std::vector<uint8_t> remove(k, 0);
    const double threshold = 1.0 - static_cast<double>(top_p);
    for (int pos = 0; pos < k; ++pos) {
        const int idx = order[pos];
        cumulative += probs[idx] / denom;
        if (cumulative <= threshold) {
            remove[idx] = 1;
        }
    }
    remove[order[k - 1]] = 0;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        if (remove[i]) {
            top[i].score = neg_inf;
        }
    }
    end = clock_type::now();
    timings.top_p_ms += elapsed_ms(start, end);

    start = clock_type::now();
    for (auto & item : top) {
        if (item.score == neg_inf || !seen_mask[static_cast<size_t>(item.token)]) {
            continue;
        }
        if (item.score < 0.0f) {
            item.score *= repetition_penalty;
        } else {
            item.score /= repetition_penalty;
        }
    }
    end = clock_type::now();
    timings.repetition_ms += elapsed_ms(start, end);

    start = clock_type::now();
    max_score = -std::numeric_limits<float>::infinity();
    for (const auto & item : top) {
        max_score = std::max(max_score, item.score);
    }
    if (max_score == neg_inf) {
        timings.softmax_sample_ms += elapsed_ms(start, clock_type::now());
        timings.total_ms += elapsed_ms(total_start, clock_type::now());
        return -1;
    }
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        if (top[i].score == neg_inf) {
            probs[i] = 0.0;
        } else {
            probs[i] = std::exp(static_cast<double>(top[i].score - max_score));
            sum += probs[i];
        }
    }
    std::uniform_real_distribution<double> uniform(0.0, sum);
    const double target = uniform(rng);
    cumulative = 0.0;
    int sampled = top[k - 1].token;
    for (int i = 0; i < k; ++i) {
        cumulative += probs[i];
        if (target <= cumulative) {
            sampled = top[i].token;
            break;
        }
    }
    end = clock_type::now();
    timings.softmax_sample_ms += elapsed_ms(start, end);
    timings.total_ms += elapsed_ms(total_start, end);
    return sampled;
}

CaseResult run_case(int seen_len, int vocab, int trials) {
    std::mt19937 rng(20260708u + static_cast<unsigned>(seen_len));
    std::uniform_int_distribution<int> token_dist(0, vocab - 1);
    std::normal_distribution<float> logit_dist(0.0f, 1.0f);

    std::vector<int> seen_tokens(static_cast<size_t>(seen_len));
    for (int i = 0; i < seen_len; ++i) {
        seen_tokens[static_cast<size_t>(i)] = token_dist(rng);
    }
    if (seen_len > 4) {
        for (int i = 0; i < seen_len; i += 7) {
            seen_tokens[static_cast<size_t>(i)] = seen_tokens[0];
        }
    }
    std::vector<uint8_t> seen_mask(static_cast<size_t>(vocab), 0);
    std::unordered_set<int> unique;
    for (int token : seen_tokens) {
        seen_mask[static_cast<size_t>(token)] = 1;
        unique.insert(token);
    }

    CaseResult result;
    result.seen_len = seen_len;
    result.unique_len = static_cast<int>(unique.size());
    result.vocab = vocab;
    result.trials = trials;

    std::vector<float> logits(static_cast<size_t>(vocab));
    for (int trial = 0; trial < trials; ++trial) {
        for (int i = 0; i < vocab; ++i) {
            logits[static_cast<size_t>(i)] = logit_dist(rng);
        }
        int sampled = sample_native(logits, seen_mask, 1000, 0.95f, 0.8f, 1.2f, rng, result.timings);
        result.checksum += static_cast<double>(sampled + 1) * static_cast<double>(trial + 1);
    }
    return result;
}

void print_case_json(const CaseResult & item, bool last) {
    std::cout << "    {\n";
    std::cout << "      \"seen_len\": " << item.seen_len << ",\n";
    std::cout << "      \"unique_len\": " << item.unique_len << ",\n";
    std::cout << "      \"vocab\": " << item.vocab << ",\n";
    std::cout << "      \"trials\": " << item.trials << ",\n";
    std::cout << "      \"checksum\": " << item.checksum << ",\n";
    std::cout << "      \"timings_ms\": {\n";
    std::cout << "        \"topk\": " << item.timings.topk_ms << ",\n";
    std::cout << "        \"top_p\": " << item.timings.top_p_ms << ",\n";
    std::cout << "        \"repetition\": " << item.timings.repetition_ms << ",\n";
    std::cout << "        \"softmax_sample\": " << item.timings.softmax_sample_ms << ",\n";
    std::cout << "        \"total\": " << item.timings.total_ms << "\n";
    std::cout << "      },\n";
    std::cout << "      \"per_trial_ms\": {\n";
    std::cout << "        \"topk\": " << item.timings.topk_ms / item.trials << ",\n";
    std::cout << "        \"top_p\": " << item.timings.top_p_ms / item.trials << ",\n";
    std::cout << "        \"repetition\": " << item.timings.repetition_ms / item.trials << ",\n";
    std::cout << "        \"softmax_sample\": " << item.timings.softmax_sample_ms / item.trials << ",\n";
    std::cout << "        \"total\": " << item.timings.total_ms / item.trials << "\n";
    std::cout << "      }\n";
    std::cout << "    }" << (last ? "\n" : ",\n");
}

} // namespace

int main(int argc, char ** argv) {
    int trials = 1000;
    int vocab = 8192;
    if (argc > 1) {
        trials = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        vocab = std::max(128, std::atoi(argv[2]));
    }

    std::vector<CaseResult> cases;
    for (int seen_len : {1, 64, 358}) {
        cases.push_back(run_case(seen_len, vocab, trials));
    }

    std::cout << "{\n";
    std::cout << "  \"description\": \"Standalone native C++ T3 sampler microbenchmark. No Chatterbox model load and no Vulkan execution.\",\n";
    std::cout << "  \"compiler_probe\": \"native_cpp_sampler\",\n";
    std::cout << "  \"trials\": " << trials << ",\n";
    std::cout << "  \"vocab\": " << vocab << ",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        print_case_json(cases[i], i + 1 == cases.size());
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    return 0;
}
