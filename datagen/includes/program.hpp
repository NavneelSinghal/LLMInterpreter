#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <random>
#include <stack>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

struct ProgramParams {
  // sizes
  size_t N_A = 20;
  size_t N_I = 20;
  size_t N_INPUT = 20;
  size_t N_OUTPUT = 20;
  size_t B = 4;

  // instructions allowed
  bool IO_ALLOWED = true;
  bool LOOPS_ALLOWED = true;
  bool STOPS_ALLOWED = true;
};

enum struct InstructionToken {
  NEXT,
  PREV,
  INC,
  DEC,
  PUT,
  GET,
  WHILE,
  ENDW,
  STOP
};
enum struct SpecialToken { EOI, EOO };
enum struct OneHotToken { ZERO, ONE };
enum struct Parens { LC, RC, LS, RS };

struct Number {
  size_t n{};
};

static std::string to_string(InstructionToken token) {
  switch (token) {
  case InstructionToken::NEXT:
    return "NEXT";
  case InstructionToken::PREV:
    return "PREV";
  case InstructionToken::INC:
    return "INC";
  case InstructionToken::DEC:
    return "DEC";
  case InstructionToken::PUT:
    return "PUT";
  case InstructionToken::GET:
    return "GET";
  case InstructionToken::WHILE:
    return "WHILE";
  case InstructionToken::ENDW:
    return "ENDW";
  case InstructionToken::STOP:
    return "STOP";
  default:
    __builtin_unreachable();
  }
}

static std::string to_string(SpecialToken token) {
  switch (token) {
  case SpecialToken::EOI:
    return "EOI";
  case SpecialToken::EOO:
    return "EOO";
  default:
    __builtin_unreachable();
  }
}

static std::string to_string(OneHotToken token) {
  switch (token) {
  case OneHotToken::ZERO:
    return "0";
  case OneHotToken::ONE:
    return "1";
  default:
    __builtin_unreachable();
  }
}

static std::string to_string(Parens token) {
  switch (token) {
  case Parens::LC:
    return "{";
  case Parens::RC:
    return "}";
  case Parens::LS:
    return "[";
  case Parens::RS:
    return "]";
  default:
    __builtin_unreachable();
  }
}

static std::string to_string(Number number) {
  return "N_" + std::to_string(number.n);
}

template <size_t N> struct OneHot {
  std::array<OneHotToken, N> index{};
  size_t index_int{};
  constexpr OneHot(size_t x = 0) {
    index.fill(OneHotToken::ZERO);
    index_int = x;
    if (x >= N)
      return;
    index[x] = OneHotToken::ONE;
  }
  constexpr auto operator<=>(const OneHot &) const = default;
  void set(size_t new_index) {
    if (index_int < N)
      index[index_int] = OneHotToken::ZERO;
    index_int = new_index;
    if (new_index >= N)
      return;
    index[new_index] = OneHotToken::ONE;
  }
  bool is_valid() const {
    if (index_int >= N)
      return false;
    for (size_t i = 0; i != N; ++i) {
      if (i == index_int && index[i] == OneHotToken::ZERO)
        return false;
      if (i != index_int && index[i] == OneHotToken::ONE)
        return false;
    }
    return true;
  }
};

template <ProgramParams PROGRAM_PARAMS> struct Program {

  static_assert(PROGRAM_PARAMS.B <= 16);
  static_assert(PROGRAM_PARAMS.N_I >= 1);

  static std::optional<std::array<size_t, PROGRAM_PARAMS.N_I>>
  get_matching_indices(
      const std::array<InstructionToken, PROGRAM_PARAMS.N_I> &instructions) {
    std::stack<size_t> indices{};
    std::array<size_t, PROGRAM_PARAMS.N_I> res{};
    for (size_t i = 0; i != instructions.size(); ++i) {
      if (instructions[i] == InstructionToken::WHILE) {
        indices.push(i);
      } else if (instructions[i] == InstructionToken::ENDW) {
        if (indices.empty())
          return {};
        auto left = indices.top();
        indices.pop();
        res[left] = i;
        res[i] = left;
      }
    }
    if (!indices.empty())
      return {};
    return res;
  }

  static constexpr size_t calculate_single_state_token_length() {
    constexpr auto params = PROGRAM_PARAMS;
    return 1 + 1 + params.N_I + 1 + 1 + params.N_I + 1 + 1 + params.N_A + 1 +
           1 + params.N_A + 1 + 1 + params.N_INPUT + 1 + 1 + params.N_OUTPUT +
           1 + 1;
  }

  struct Vocabulary {
    std::map<std::string, ptrdiff_t> token_to_id;
    std::vector<std::string> id_to_token;
    ptrdiff_t unk_id = -1;

    ptrdiff_t add(const std::string &token) {
      if (token_to_id.find(token) == token_to_id.end()) {
        ptrdiff_t id = id_to_token.size();
        token_to_id[token] = id;
        id_to_token.push_back(token);
        return id;
      }
      return token_to_id[token];
    }

    template <typename T> T get_id(const std::string &token) const {
      auto it = token_to_id.find(token);
      if (it == token_to_id.end()) {
        if (unk_id != -1)
          return unk_id;
        throw std::runtime_error("Unknown token in vocab: " + token);
      }
      auto id = it->second;
      if (id > std::numeric_limits<T>::max() ||
          id < std::numeric_limits<T>::min()) {
        throw std::runtime_error("Vocabulary size exceeds capacity."
                                 " Current ID: " +
                                 std::to_string(id) + " for token: " + token);
      }
      return id;
    }

    template <typename TokenType> void populate_from_params() {
      constexpr auto params = PROGRAM_PARAMS;
      token_to_id.clear();
      id_to_token.clear();
      // unk_id = add("<UNK>"); // Optional
      add("{");
      add("}");
      add("[");
      add("]");
      add(::to_string(OneHotToken::ZERO));
      add(::to_string(OneHotToken::ONE));
      add(::to_string(InstructionToken::NEXT));
      add(::to_string(InstructionToken::PREV));
      add(::to_string(InstructionToken::INC));
      add(::to_string(InstructionToken::DEC));
      if (params.IO_ALLOWED) {
        add(::to_string(InstructionToken::PUT));
        add(::to_string(InstructionToken::GET));
      }
      if (params.LOOPS_ALLOWED) {
        add(::to_string(InstructionToken::WHILE));
        add(::to_string(InstructionToken::ENDW));
      }
      add(::to_string(InstructionToken::STOP));
      for (size_t i = 0; i != (1ULL << params.B); ++i)
        add(::to_string(Number{i}));
      add(::to_string(SpecialToken::EOI));
      add(::to_string(SpecialToken::EOO));
      if (token_to_id.size() > std::numeric_limits<TokenType>::max()) {
        throw std::runtime_error("Vocabulary size exceeds capacity. Please "
                                 "check your TokenType again.");
      }
    }
  };

  struct ProgramState {
    OneHot<PROGRAM_PARAMS.N_I> instruction_ptr{};
    std::array<InstructionToken, PROGRAM_PARAMS.N_I> instructions{};
    OneHot<PROGRAM_PARAMS.N_A> array_ptr{};
    std::array<Number, PROGRAM_PARAMS.N_A> array_data{};
    std::array<std::variant<Number, SpecialToken>, PROGRAM_PARAMS.N_INPUT>
        input_buffer{};
    std::array<std::variant<Number, SpecialToken>, PROGRAM_PARAMS.N_OUTPUT>
        output_buffer{};

    std::array<size_t, PROGRAM_PARAMS.N_I> matching_indices{};
    size_t output_ptr{};

    constexpr ProgramState() {
      instruction_ptr = {};
      instructions.fill(InstructionToken::STOP);
      array_ptr = {};
      array_data.fill(Number{0});
      input_buffer.fill(SpecialToken::EOI);
      output_buffer.fill(SpecialToken::EOO);
      matching_indices.fill(0);
    }

    constexpr auto operator<=>(const ProgramState &) const = default;

    bool compute_helpers() {
      auto computed_matching_indices = get_matching_indices(instructions);
      if (computed_matching_indices == std::nullopt)
        return false;
      matching_indices = *computed_matching_indices;
      output_ptr =
          std::find_if(output_buffer.begin(), output_buffer.end(),
                       [](const auto &x) {
                         return std::holds_alternative<SpecialToken>(x) &&
                                std::get<SpecialToken>(x) == SpecialToken::EOO;
                       }) -
          output_buffer.begin();
      return true;
    }

    bool is_valid() const {

      // validate instruction ptr
      if (!instruction_ptr.is_valid())
        return false;

      // validate STOP instruction
      if (instructions[PROGRAM_PARAMS.N_I - 1] != InstructionToken::STOP)
        return false;

      // check for allowed instructions
      for (size_t i = 0; i != PROGRAM_PARAMS.N_I; ++i) {
        if (!PROGRAM_PARAMS.IO_ALLOWED &&
            (instructions[i] == InstructionToken::PUT ||
             instructions[i] == InstructionToken::GET))
          return false;
        if (!PROGRAM_PARAMS.LOOPS_ALLOWED &&
            (instructions[i] == InstructionToken::WHILE ||
             instructions[i] == InstructionToken::ENDW))
          return false;
      }

      // check stops
      if (!PROGRAM_PARAMS.STOPS_ALLOWED &&
          std::find(instructions.begin(), instructions.end(),
                    InstructionToken::STOP) != instructions.end() - 1) {
        return false;
      }

      // check for balanced parentheses
      ptrdiff_t balance = 0;
      for (size_t i = 0; i != PROGRAM_PARAMS.N_I; ++i) {
        if (instructions[i] == InstructionToken::WHILE) {
          balance += 1;
        } else if (instructions[i] == InstructionToken::ENDW) {
          balance -= 1;
        }
        if (balance < 0)
          return false;
      }
      if (balance != 0)
        return false;

      // validate matching indices
      auto computed_matching_indices = get_matching_indices(instructions);
      if (computed_matching_indices == std::nullopt ||
          *computed_matching_indices != matching_indices)
        return false;

      // validate array ptr
      if (!array_ptr.is_valid())
        return false;

      // validate array
      for (auto x : array_data) {
        if (x.n >= (1 << PROGRAM_PARAMS.B)) {
          return false;
        }
      }

      auto find_eoo = [](const auto &x) {
        return std::holds_alternative<SpecialToken>(x) &&
               std::get<SpecialToken>(x) == SpecialToken::EOO;
      };
      auto find_eoi = [](const auto &x) {
        return std::holds_alternative<SpecialToken>(x) &&
               std::get<SpecialToken>(x) == SpecialToken::EOI;
      };

      // validate input buffer
      if (std::find_if(input_buffer.begin(), input_buffer.end(), find_eoo) !=
          input_buffer.end())
        return false;
      auto it_input =
          std::find_if(input_buffer.begin(), input_buffer.end(), find_eoi);
      auto count_eoi = std::count_if(it_input, input_buffer.end(), find_eoi);
      if (count_eoi != input_buffer.end() - it_input)
        return false;

      for (auto x : input_buffer) {
        if (std::holds_alternative<Number>(x) &&
            std::get<Number>(x).n >= (1 << PROGRAM_PARAMS.B)) {
          return false;
        }
      }

      // validate output buffer
      if (std::find_if(output_buffer.begin(), output_buffer.end(), find_eoi) !=
          output_buffer.end())
        return false;
      auto it_output =
          std::find_if(output_buffer.begin(), output_buffer.end(), find_eoo);
      if (output_buffer.begin() + output_ptr != it_output)
        return false;
      auto count_eoo = std::count_if(it_output, output_buffer.end(), find_eoo);
      if (count_eoo != output_buffer.end() - it_output)
        return false;

      for (auto x : output_buffer) {
        if (std::holds_alternative<Number>(x) &&
            std::get<Number>(x).n >= (1 << PROGRAM_PARAMS.B)) {
          return false;
        }
      }

      return true;
    }

    std::string to_string() const {
      std::string repr = "{ [";
      for (auto x : instruction_ptr.index) {
        repr += ' ';
        repr += ::to_string(x);
      }
      repr += " ] [";
      for (auto x : instructions) {
        repr += ' ';
        repr += ::to_string(x);
      }
      repr += " ] [";
      for (auto x : array_ptr.index) {
        repr += ' ';
        repr += ::to_string(x);
      }
      repr += " ] [";
      for (auto x : array_data) {
        repr += ' ';
        repr += ::to_string(x);
      }
      repr += " ] [";
      for (auto x : input_buffer) {
        repr += ' ';
        repr += std::visit([](auto &&arg) { return ::to_string(arg); }, x);
      }
      repr += " ] [";
      for (auto x : output_buffer) {
        repr += ' ';
        repr += std::visit([](auto &&arg) { return ::to_string(arg); }, x);
      }
      repr += " ] }";
      return repr;
    }

    template <typename T>
    std::vector<T> to_token_indices(const Vocabulary &vocab) const {
      std::vector<T> indices_vec;
      auto add_token = [&](const std::string &token_str) {
        indices_vec.push_back(vocab.template get_id<T>(token_str));
      };
      auto add_token_variant = [&](const auto &variant_val) {
        indices_vec.push_back(vocab.template get_id<T>(std::visit(
            [](auto &&arg) { return ::to_string(arg); }, variant_val)));
      };
      add_token("{");
      add_token("[");
      for (auto x : instruction_ptr.index) {
        add_token(::to_string(x));
      }
      add_token("]");

      add_token("[");
      for (auto x : instructions) {
        add_token(::to_string(x));
      }
      add_token("]");

      add_token("[");
      for (auto x : array_ptr.index) {
        add_token(::to_string(x));
      }
      add_token("]");

      add_token("[");
      for (auto x : array_data) {
        add_token(::to_string(x));
      }
      add_token("]");

      add_token("[");
      for (const auto &x : input_buffer) {
        add_token_variant(x);
      }
      add_token("]");

      add_token("[");
      for (const auto &x : output_buffer) {
        add_token_variant(x);
      }
      add_token("]");
      add_token("}");

      return indices_vec;
    }

    auto execute_instruction() {
      if (!is_valid())
        throw std::string{"Program not valid, please check: "} + to_string();
      auto instruction = instructions[instruction_ptr.index_int];
      if (instruction == InstructionToken::STOP) {
        return;
      }
      // the instruction ptr now has at least something after it
      switch (instruction) {
      case InstructionToken::NEXT: {
        if (array_ptr.index_int == array_data.size() - 1)
          return;
        array_ptr.set(array_ptr.index_int + 1);
        break;
      }
      case InstructionToken::PREV: {
        if (array_ptr.index_int == 0)
          return;
        array_ptr.set(array_ptr.index_int - 1);
        break;
      }
      case InstructionToken::INC: {
        if (array_data[array_ptr.index_int].n == (1 << PROGRAM_PARAMS.B) - 1) {
          return;
        }
        array_data[array_ptr.index_int].n += 1;
        break;
      }
      case InstructionToken::DEC: {
        if (array_data[array_ptr.index_int].n == 0) {
          return;
        }
        array_data[array_ptr.index_int].n -= 1;
        break;
      }
      case InstructionToken::PUT: {
        if (output_ptr == output_buffer.size()) {
          return;
        }
        output_buffer[output_ptr++] = array_data[array_ptr.index_int];
        break;
      }
      case InstructionToken::GET: {
        if (!std::holds_alternative<Number>(input_buffer[0])) {
          return;
        }
        array_data[array_ptr.index_int] = std::get<Number>(input_buffer[0]);
        std::copy(input_buffer.begin() + 1, input_buffer.end(),
                  input_buffer.begin());
        input_buffer.back() = SpecialToken::EOI;
        break;
      }
      case InstructionToken::WHILE: {
        if (array_data[array_ptr.index_int].n == 0) {
          instruction_ptr.set(matching_indices[instruction_ptr.index_int]);
        }
        break;
      }
      case InstructionToken::ENDW: {
        if (array_data[array_ptr.index_int].n != 0) {
          instruction_ptr.set(matching_indices[instruction_ptr.index_int]);
        }
        break;
      }
      default: {
        // std::unreachable();
        __builtin_unreachable();
        break;
      }
      }
      instruction_ptr.set(instruction_ptr.index_int + 1);
    }

    template <ProgramParams NEW_PARAMS>
    std::optional<typename Program<NEW_PARAMS>::ProgramState> extend() const {
      if (NEW_PARAMS.N_I < PROGRAM_PARAMS.N_I)
        return {};
      if (NEW_PARAMS.N_A < PROGRAM_PARAMS.N_A)
        return {};
      if (NEW_PARAMS.N_INPUT < PROGRAM_PARAMS.N_INPUT)
        return {};
      if (NEW_PARAMS.N_OUTPUT < PROGRAM_PARAMS.N_OUTPUT)
        return {};
      if (NEW_PARAMS.B < PROGRAM_PARAMS.B)
        return {};
      if (!NEW_PARAMS.IO_ALLOWED && PROGRAM_PARAMS.IO_ALLOWED)
        return {};
      if (!NEW_PARAMS.LOOPS_ALLOWED && PROGRAM_PARAMS.LOOPS_ALLOWED)
        return {};
      if (!NEW_PARAMS.STOPS_ALLOWED && PROGRAM_PARAMS.STOPS_ALLOWED)
        return {};

      typename Program<NEW_PARAMS>::ProgramState res{};
      res.instruction_ptr.index_int = instruction_ptr.index_int;
      std::copy(instruction_ptr.index.begin(), instruction_ptr.index.end(),
                res.instruction_ptr.index.begin());
      std::copy(instructions.begin(), instructions.end(),
                res.instructions.begin());
      res.array_ptr.index_int = array_ptr.index_int;
      std::copy(array_ptr.index.begin(), array_ptr.index.end(),
                res.array_ptr.index.begin());
      std::copy(array_data.begin(), array_data.end(), res.array_data.begin());
      std::copy(input_buffer.begin(), input_buffer.end(),
                res.input_buffer.begin());
      std::copy(output_buffer.begin(), output_buffer.end(),
                res.output_buffer.begin());
      std::copy(matching_indices.begin(), matching_indices.end(),
                res.matching_indices.begin());
      res.output_ptr = output_ptr;
      return res;
    }
  };

  struct Samplers {

    // helper functions

    static long double log_combinations(int n, int k) {
      if (k < 0 || k > n)
        return -std::numeric_limits<long double>::infinity();
      if (k == 0 || k == n)
        return 0.0L;
      if (k > n / 2)
        k = n - k;
      return std::lgamma(n + 1.0L) - std::lgamma(k + 1.0L) -
             std::lgamma(n - k + 1.0L);
    }

    static long double log_catalan_number(int n) {
      if (n < 0)
        return -std::numeric_limits<long double>::infinity();
      if (n == 0)
        return 0.0L;
      return log_combinations(2 * n, n) - std::log((long double)n + 1.0L);
    }

    static std::string balanced_bracket_sequence(size_t n, std::mt19937 &rng) {
      auto res = std::string(n, '(') + std::string(n + 1, ')');
      std::shuffle(res.begin(), res.end(), rng);
      std::vector<int> pref(2 * n + 2, 0);
      for (size_t x = 0; x <= 2 * n; x++)
        pref[x + 1] = pref[x] + (res[x] == '(' ? 1 : -1);
      size_t idx = static_cast<size_t>(
          std::min_element(pref.begin(), pref.end()) - pref.begin());
      std::rotate(res.begin(), res.begin() + idx, res.end());
      return res.substr(0, 2 * n);
    }

    struct InstructionSampler {

      size_t N_PROG;
      std::vector<InstructionToken> filler_ops_list;
      size_t num_filler_types;
      int max_k_val;

      std::vector<long double> probabilities_Pk;
      std::vector<long double> cdf_Pk;
      bool successfully_initialized = false;

      InstructionSampler() {
        constexpr auto params = PROGRAM_PARAMS;
        N_PROG = params.N_I - 1;

        filler_ops_list.push_back(InstructionToken::NEXT);
        filler_ops_list.push_back(InstructionToken::PREV);
        filler_ops_list.push_back(InstructionToken::INC);
        filler_ops_list.push_back(InstructionToken::DEC);
        if (params.IO_ALLOWED) {
          filler_ops_list.push_back(InstructionToken::PUT);
          filler_ops_list.push_back(InstructionToken::GET);
        }
        if (params.STOPS_ALLOWED) {
          filler_ops_list.push_back(InstructionToken::STOP);
        }
        num_filler_types = filler_ops_list.size();

        size_t max_k_possible =
            params.LOOPS_ALLOWED ? static_cast<size_t>(N_PROG / 2) : 0;
        max_k_val = max_k_possible;

        std::vector<long double> log_Wk_values;
        long double max_log_Wk = -std::numeric_limits<long double>::infinity();

        for (size_t k = 0; k <= max_k_possible; ++k) {
          long double log_Ck = Samplers::log_catalan_number(k);
          long double log_Npk = Samplers::log_combinations(N_PROG, 2 * k);
          long double log_Nfk = 0.0L;
          if (N_PROG - 2 * k > 0)
            log_Nfk = (long double)(N_PROG - 2 * k) *
                      std::log((long double)num_filler_types);
          long double current_log_Wk = log_Ck + log_Npk + log_Nfk;
          log_Wk_values.push_back(current_log_Wk);
          if (current_log_Wk > max_log_Wk)
            max_log_Wk = current_log_Wk;
        }

        probabilities_Pk.resize(max_k_possible + 1);
        long double W_total_scaled = 0.0L;
        for (size_t k = 0; k <= max_k_possible; ++k) {
          probabilities_Pk[k] = std::exp(log_Wk_values[k] - max_log_Wk);
          W_total_scaled += probabilities_Pk[k];
        }

        if (W_total_scaled <= 0.0L) {
          successfully_initialized = false;
          return;
        }

        cdf_Pk.resize(max_k_possible + 1);
        long double cumulative_p = 0.0L;
        for (size_t k = 0; k <= max_k_possible; ++k) {
          probabilities_Pk[k] /= W_total_scaled;
          cumulative_p += probabilities_Pk[k];
          cdf_Pk[k] = cumulative_p;
        }
        for (auto &x : cdf_Pk)
          x /= cumulative_p;
        successfully_initialized = true;
      }

      std::optional<int> sample_k(std::mt19937 &rng) {
        if (!successfully_initialized || cdf_Pk.empty())
          return std::nullopt;
        std::uniform_real_distribution<long double> dist(0.0L, 1.0L);
        long double u = dist(rng);
        return std::lower_bound(cdf_Pk.begin(), cdf_Pk.begin() + max_k_val + 1,
                                u) -
               cdf_Pk.begin();
      }

      template <bool use_uniform_k = false>
      std::optional<
          std::pair<size_t, std::array<InstructionToken, PROGRAM_PARAMS.N_I>>>
      sample_instructions(std::mt19937 &rng) {
        size_t k{};
        if constexpr (!use_uniform_k) {
          auto k_ = sample_k(rng);
          if (k_ == std::nullopt)
            return {};
          k = *k_;
        } else {
          std::uniform_int_distribution<size_t> dist(
              0, (PROGRAM_PARAMS.N_I - 1) / 2);
          k = dist(rng);
        }
        static constexpr auto n = PROGRAM_PARAMS.N_I;
        std::vector<size_t> positions(2 * k);
        std::uniform_int_distribution<size_t> dist(0, n - 1 - 2 * k);
        std::generate(positions.begin(), positions.end(),
                      [&]() { return dist(rng); });
        std::sort(positions.begin(), positions.end());
        auto bracket_sequence = balanced_bracket_sequence(k, rng);
        std::array<InstructionToken, n> res{};
        res.fill(InstructionToken::STOP);
        for (size_t i = 0; i != 2 * k; ++i) {
          res[positions[i] + i] = bracket_sequence[i] == '('
                                      ? InstructionToken::WHILE
                                      : InstructionToken::ENDW;
        }
        std::uniform_int_distribution<int> filler_dist(
            0, filler_ops_list.size() - 1);
        for (size_t i = 0; i + 1 < n; ++i)
          if (res[i] == InstructionToken::STOP)
            res[i] = filler_ops_list[filler_dist(rng)];
        return std::pair{std::uniform_int_distribution<size_t>(0, n - 1)(rng),
                         res};
      }
    };

    // array sampling
    std::optional<std::vector<Number>> sample_array(size_t N, double p_uniform,
                                                    double p_geo, double p_rgeo,
                                                    double p_gauss,
                                                    std::mt19937 &rng) {
      static constexpr std::array<double, (1 << PROGRAM_PARAMS.B)> geo = [] {
        std::array<double, (1 << PROGRAM_PARAMS.B)> res{}; // unnormalized
        res[0] = 1;
        double factor =
            1.0 - 1.0 / (1 << PROGRAM_PARAMS.B); // for 1/e at the end
        for (size_t i = 1; i != res.size(); ++i)
          res[i] = res[i - 1] * factor;
        // use fenwick for error reduction if B is large
        for (size_t i = 1; i != res.size(); ++i)
          res[i] += res[i - 1];
        return res;
      }();

      static constexpr std::array<double, (1 << PROGRAM_PARAMS.B)> gauss = [] {
        std::array<double, (1 << PROGRAM_PARAMS.B)> res{}; // unnormalized
        double sigma = std::sqrt((1 << PROGRAM_PARAMS.B) * 0.25);
        double mu = (1 << PROGRAM_PARAMS.B) * 0.5;
        for (size_t i = 0; i != res.size(); ++i)
          res[i] = std::exp(-(i - mu) * (i - mu) / (2 * sigma * sigma));
        // use fenwick for error reduction if B is large
        for (size_t i = 1; i != res.size(); ++i)
          res[i] += res[i - 1];
        return res;
      }();

      std::vector<Number> res(N);
      std::array<double, 4> p{};
      p[0] = p_uniform;
      p[1] = p[0] + p_geo;
      p[2] = p[1] + p_rgeo;
      p[3] = p[2] + p_gauss;
      std::uniform_real_distribution<double> dist(0, 1);
      for (auto &x : res) {
        double p_dist = dist(rng) * p[3];
        auto choice = std::lower_bound(p.begin(), p.end(), p_dist) - p.begin();
        double p_res = dist(rng);
        switch (choice) {
        case 0: {
          x.n = p_res * (1 << PROGRAM_PARAMS.B);
          break;
        }
        case 1: {
          x.n =
              std::lower_bound(geo.begin(), geo.end(), p_res * *geo.rbegin()) -
              geo.begin();
          break;
        }
        case 2: {
          x.n = geo.end() - 1 -
                std::lower_bound(geo.begin(), geo.end(), p_res * *geo.rbegin());
          break;
        }
        case 3: {
          x.n = std::lower_bound(gauss.begin(), gauss.end(),
                                 p_res * *gauss.rbegin()) -
                gauss.begin();
          break;
        }
        default: {
          return {};
        }
        }
      }
      return res;
    }

    enum struct LengthSamplingDistributions {
      Uniform,
      Exponential,
      Empty,
      Full,
    };

    template <SpecialToken sentinel>
    std::optional<std::vector<std::variant<Number, SpecialToken>>>
    sample_with_sentinel(size_t N, double pl_uniform, double pl_geo,
                         double pl_empty, double pl_full, double param_geo,
                         double p_uniform, double p_geo, double p_rgeo,
                         double p_gauss, std::mt19937 &rng) {
      std::array<double, 4> pl{};
      pl[0] = pl_uniform;
      pl[1] = pl[0] + pl_geo;
      pl[2] = pl[1] + pl_empty;
      pl[3] = pl[2] + pl_full;
      std::uniform_real_distribution<double> dist(0, 1);
      double p_len = pl[3] * dist(rng);
      auto choice = std::lower_bound(pl.begin(), pl.end(), p_len) - pl.begin();
      size_t len = 0;
      switch (choice) {
      case 0: {
        len = N * dist(rng);
        break;
      }
      case 1: {
        int64_t l = -1;
        int64_t r = N;
        auto binexp = [](auto x, size_t N) {
          std::decay_t<decltype(x)> ans = 1;
          while (N) {
            if (N & 1)
              ans *= x;
            x *= x;
            N >>= 1;
          }
          return ans;
        };
        double roll = dist(rng) * (1 - binexp(param_geo, N));
        while (r - l > 1) {
          int64_t m = (l + r) / 2;
          if (1 - binexp(param_geo, m + 1) >= roll) {
            r = m;
          } else {
            l = m;
          }
        }
        len = r;
        break;
      }
      case 2: {
        len = 0;
        break;
      }
      case 3: {
        len = N;
        break;
      }
      default: {
        return {};
      }
      }
      std::vector<std::variant<Number, SpecialToken>> ans(N, {sentinel});
      auto numbers = sample_array(len, p_uniform, p_geo, p_rgeo, p_gauss, rng);
      std::copy(numbers->begin(), numbers->end(), ans.begin());
      return ans;
    }
  };
};
