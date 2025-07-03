#include "program.hpp"       // Adjusted path

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

template <ProgramParams PROGRAM_PARAMS>
std::optional<std::pair<typename Program<PROGRAM_PARAMS>::ProgramState,
                        typename Program<PROGRAM_PARAMS>::ProgramState>>
get_states(auto &rng, typename Program<PROGRAM_PARAMS>::Samplers &samplers,
           typename Program<PROGRAM_PARAMS>::Samplers::InstructionSampler
               &instruction_sampler) {

  typename Program<PROGRAM_PARAMS>::ProgramState s_t;

  auto sampled_instructions_pair =
      instruction_sampler.template sample_instructions<false>(
          rng); // can be true - for uniformly distributed n_loops

  if (!sampled_instructions_pair) {
    std::cerr
        << "Failed to sample instructions. This might be due to "
        << "InstructionSampler initialization issues or if N_I is too small."
        << '\n';
    return {};
  }
  s_t.instruction_ptr.set(sampled_instructions_pair->first);
  s_t.instructions = sampled_instructions_pair->second;
  assert(s_t.instructions[PROGRAM_PARAMS.N_I - 1] == InstructionToken::STOP);

  double p_uniform = 0.25, p_geo = 0.25, p_rgeo = 0.25, p_gauss = 0.25;
  auto sampled_array_data_opt = samplers.sample_array(
      PROGRAM_PARAMS.N_A, p_uniform, p_geo, p_rgeo, p_gauss, rng);
  if (!sampled_array_data_opt) {
    std::cerr << "Failed to sample array data." << '\n';
    return {};
  }
  std::copy(sampled_array_data_opt->begin(), sampled_array_data_opt->end(),
            s_t.array_data.begin());

  std::uniform_int_distribution<size_t> array_ptr_dist(0,
                                                       PROGRAM_PARAMS.N_A - 1);
  s_t.array_ptr.set(array_ptr_dist(rng));

  double pl_uniform = 0.25, pl_geo = 0.25, pl_empty = 0.25, pl_full = 0.25;
  double param_geo_len = 0.5;
  auto sampled_input_buffer_opt =
      samplers.template sample_with_sentinel<SpecialToken::EOI>(
          PROGRAM_PARAMS.N_INPUT, pl_uniform, pl_geo, pl_empty, pl_full,
          param_geo_len, p_uniform, p_geo, p_rgeo, p_gauss, rng);
  if (!sampled_input_buffer_opt) {
    std::cerr << "Failed to sample input buffer." << '\n';
    return {};
  }
  std::copy(sampled_input_buffer_opt->begin(), sampled_input_buffer_opt->end(),
            s_t.input_buffer.begin());

  auto sampled_output_buffer_opt =
      samplers.template sample_with_sentinel<SpecialToken::EOO>(
          PROGRAM_PARAMS.N_OUTPUT, pl_uniform, pl_geo, pl_empty, pl_full,
          param_geo_len, p_uniform, p_geo, p_rgeo, p_gauss, rng);
  if (!sampled_output_buffer_opt) {
    std::cerr << "Failed to sample output buffer." << '\n';
    return {};
  }
  std::copy(sampled_output_buffer_opt->begin(),
            sampled_output_buffer_opt->end(), s_t.output_buffer.begin());

  if (!s_t.compute_helpers()) {
    std::cerr << "Failed to compute helpers (e.g., unbalanced WHILE/ENDW). "
                 "Resampling might be needed."
              << '\n';
    return {};
  }

  if (!s_t.is_valid()) {
    std::cerr << "Generated ProgramState S_t is invalid!" << '\n';
    std::cerr << "S_t: " << s_t.to_string() << '\n';
    return {};
  }

  auto s_t_plus_1 = s_t;
  s_t_plus_1.execute_instruction();
  return std::pair{s_t, s_t_plus_1};
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <output_dir> <output_file_prefix> <num_records> <seed>" << '\n';
    std::cerr << "Example: " << argv[0] << " ./my_data run01 1000000 0" << '\n';
    return 1;
  }

  std::string output_dir = argv[1];
  std::string output_file_prefix = argv[2];
  size_t num_records_to_generate = std::stoul(argv[3]);
  uint64_t seed = std::stoull(argv[4]);

  try {
    std::filesystem::create_directories(output_dir);
  } catch (const std::exception &e) {
    std::cerr << "Error creating directory " << output_dir << ": " << e.what()
              << '\n';
    return 1;
  }

  std::string base_filepath = output_dir + "/" + output_file_prefix;
  std::string output_npy_filename = base_filepath + ".npy";

  constexpr ProgramParams PROGRAM_PARAMS = {.N_A = 10,
                                            .N_I = 15,
                                            .N_INPUT = 5,
                                            .N_OUTPUT = 5,
                                            .B = 4,
                                            .IO_ALLOWED = true,
                                            .LOOPS_ALLOWED = true,
                                            .STOPS_ALLOWED = true};

  using ProgramType = Program<PROGRAM_PARAMS>;

  typename ProgramType::Vocabulary vocab;

  using TokenType = int32_t;
  vocab.template populate_from_params<TokenType>();

  size_t single_state_len = ProgramType::calculate_single_state_token_length();

  typename ProgramType::Samplers samplers;
  typename ProgramType::Samplers::InstructionSampler instruction_sampler;
  if (!instruction_sampler.successfully_initialized &&
      PROGRAM_PARAMS.LOOPS_ALLOWED) {
    std::cerr << "InstructionSampler failed to initialize. Exiting." << '\n';
    if (instruction_sampler.cdf_Pk.empty() && PROGRAM_PARAMS.LOOPS_ALLOWED) {
      std::cerr << "CDF_Pk is empty, cannot sample k." << '\n';
    }
    return 1;
  }

  std::mt19937 rng(seed);

  size_t record_len = 2 * single_state_len;

  std::cout << "Starting generation of " << num_records_to_generate
            << " records..." << '\n';
  std::cout << "Parameters: N_I=" << PROGRAM_PARAMS.N_I
            << ", N_A=" << PROGRAM_PARAMS.N_A
            << ", N_INPUT=" << PROGRAM_PARAMS.N_INPUT
            << ", N_OUTPUT=" << PROGRAM_PARAMS.N_OUTPUT
            << ", B=" << PROGRAM_PARAMS.B
            << ", IO=" << PROGRAM_PARAMS.IO_ALLOWED
            << ", LOOPS=" << PROGRAM_PARAMS.LOOPS_ALLOWED
            << ", STOPS=" << PROGRAM_PARAMS.STOPS_ALLOWED << '\n';
  std::cout << "Single state token length: " << single_state_len << '\n';
  std::cout << "Record token length (S_t + S_t+1): " << record_len << '\n';

  for (size_t i = 0; i < num_records_to_generate; ++i) {
    std::cout << "============" << " Record number " << i + 1 << " ============\n";
    auto states_pair_opt =
        get_states<PROGRAM_PARAMS>(rng, samplers, instruction_sampler);
    if (!states_pair_opt) {
      std::cerr
          << "Warning: Failed to generate (S_t, S_t+1) pair for record index "
          << i << ". Skipping." << '\n';
      continue;
    }
    auto &[s_t, s_t_plus_1] = *states_pair_opt;
    std::cout << "Initial state = " << s_t.to_string() << '\n';
    std::cout << "Next state = " << s_t_plus_1.to_string() << '\n';
    std::vector<TokenType> tokens_t =
        s_t.template to_token_indices<TokenType>(vocab);
    if (tokens_t.size() != single_state_len) {
      std::cerr << "Error: S_t serialization length mismatch for record " << i
                << ". Expected " << single_state_len << ", got "
                << tokens_t.size() << ". Skipping." << '\n';
      continue;
    }

    std::vector<TokenType> tokens_t_plus_1 =
        s_t_plus_1.template to_token_indices<TokenType>(vocab);
    if (tokens_t_plus_1.size() != single_state_len) {
      std::cerr << "Error: S_t+1 serialization length mismatch for record " << i
                << ". Expected " << single_state_len << ", got "
                << tokens_t_plus_1.size() << ". Skipping." << '\n';
      continue;
    }
  }

  return 0;
}
