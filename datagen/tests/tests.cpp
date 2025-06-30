#include "program.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <type_traits>

int total_tests = 0;
int passed_tests = 0;
std::string current_test_suite_name;
std::string current_test_name;

#define TEST_ASSERT_TRUE(condition) \
    do { \
        total_tests++; \
        if (condition) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAIL: " << current_test_suite_name << "::" << current_test_name \
                      << " - Condition failed: " #condition \
                      << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        } \
    } while (false)

#define TEST_ASSERT_FALSE(condition) TEST_ASSERT_TRUE(!(condition))

#define TEST_ASSERT_EQ(val1, val2) \
    do { \
        total_tests++; \
        auto v1 = (val1); \
        auto v2 = (val2); \
        if (v1 == v2) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAIL: " << current_test_suite_name << "::" << current_test_name \
                      << " - Values not equal: " #val1 " (" << v1 << ") != " #val2 " (" << v2 << ")" \
                      << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        } \
    } while (false)

#define TEST_ASSERT_NE(val1, val2) \
    do { \
        total_tests++; \
        auto v1 = (val1); \
        auto v2 = (val2); \
        if (v1 != v2) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAIL: " << current_test_suite_name << "::" << current_test_name \
                      << " - Values equal: " #val1 " (" << v1 << ") == " #val2 " (" << v2 << ")" \
                      << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        } \
    } while (false)


void RUN_TEST(const std::string& suite_name, const std::string& test_name, auto&& test_func) {
    current_test_suite_name = suite_name;
    current_test_name = test_name;
    std::cout << "[ RUN      ] " << suite_name << "." << test_name << std::endl;
    try {
        test_func();
        std::cout << "[       OK ] " << suite_name << "." << test_name << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << suite_name << "::" << test_name
                  << " - Unhandled exception: " << e.what() << std::endl;
         std::cout << "[  FAILED  ] " << suite_name << "." << test_name << std::endl;
    } catch (...) {
        std::cerr << "FAIL: " << suite_name << "::" << test_name
                  << " - Unknown unhandled exception" << std::endl;
        std::cout << "[  FAILED  ] " << suite_name << "." << test_name << std::endl;
    }
}


static constexpr ProgramParams defaultTestParams{};

using TestProgram = Program<defaultTestParams>;
using TestProgramState = typename TestProgram::ProgramState;
using TestInstructionToken = InstructionToken;
using TestSpecialToken = SpecialToken;
using TestNumber = Number;
using TestOneHotToken = OneHotToken;

bool operator==(const TestNumber& lhs, const TestNumber& rhs) {
    return lhs.n == rhs.n;
}
std::ostream& operator<<(std::ostream& os, const TestNumber& num) {
  return os << to_string(num);
}
std::ostream& operator<<(std::ostream& os, TestOneHotToken token) {
  return os << to_string(token);
}
std::ostream& operator<<(std::ostream& os, TestSpecialToken token) {
  return os << to_string(token);
}
std::ostream& operator<<(std::ostream& os, const std::variant<TestNumber, TestSpecialToken>& var) {
    std::visit([&os](auto&& arg) {
        os << to_string(arg);
    }, var);
    return os;
}


// --- Tests for OneHot ---
void OneHotTest_Construction() {
    OneHot<5> oh_default;
    TEST_ASSERT_EQ(oh_default.index_int, 0);
    TEST_ASSERT_EQ(oh_default.index[0], TestOneHotToken::ONE);
    TEST_ASSERT_EQ(oh_default.index[1], TestOneHotToken::ZERO);

    OneHot<5> oh_at_2(2);
    TEST_ASSERT_EQ(oh_at_2.index_int, 2);
    TEST_ASSERT_EQ(oh_at_2.index[2], TestOneHotToken::ONE);
    TEST_ASSERT_EQ(oh_at_2.index[0], TestOneHotToken::ZERO);
}

void OneHotTest_SetValue() {
    OneHot<5> oh(1);
    oh.set(3);
    TEST_ASSERT_EQ(oh.index_int, 3);
    TEST_ASSERT_EQ(oh.index[3], TestOneHotToken::ONE);
    TEST_ASSERT_EQ(oh.index[1], TestOneHotToken::ZERO); // Old position should be zero
}

void OneHotTest_IsValid() {
    OneHot<5> oh_valid(2);
    TEST_ASSERT_TRUE(oh_valid.is_valid());

    OneHot<5> oh_invalid_int(5); // index_int out of bounds
    TEST_ASSERT_FALSE(oh_invalid_int.is_valid());
    
    oh_invalid_int.index_int = 6; // Further out of bounds
    TEST_ASSERT_FALSE(oh_invalid_int.is_valid());

    OneHot<5> oh_corrupt_vector(2);
    oh_corrupt_vector.index[2] = TestOneHotToken::ZERO; // Correct int, but vector[int] is ZERO
    TEST_ASSERT_FALSE(oh_corrupt_vector.is_valid());

    oh_corrupt_vector.index[2] = TestOneHotToken::ONE; // Fix it
    oh_corrupt_vector.index[3] = TestOneHotToken::ONE; // Another ONE appears
    TEST_ASSERT_FALSE(oh_corrupt_vector.is_valid());
}

// --- Tests for get_matching_indices ---
void GetMatchingIndicesTest_NoLoops() {
    std::array<TestInstructionToken, defaultTestParams.N_I> instructions{};
    instructions.fill(TestInstructionToken::STOP);
    instructions[0] = TestInstructionToken::NEXT;
    instructions[1] = TestInstructionToken::INC;

    auto result = TestProgram::get_matching_indices(instructions);
    TEST_ASSERT_TRUE(result.has_value()); // Should succeed even with no loops
}

void GetMatchingIndicesTest_SimpleLoop() {
    std::array<TestInstructionToken, defaultTestParams.N_I> instructions{};
    instructions.fill(TestInstructionToken::STOP);
    instructions[0] = TestInstructionToken::WHILE; // Index 0
    instructions[1] = TestInstructionToken::INC;
    instructions[2] = TestInstructionToken::ENDW;   // Index 2

    auto result = TestProgram::get_matching_indices(instructions);
    TEST_ASSERT_TRUE(result.has_value());
    if (result) {
        TEST_ASSERT_EQ((*result)[0], 2);
        TEST_ASSERT_EQ((*result)[2], 0);
    }
}

void GetMatchingIndicesTest_NestedLoops() {
    std::array<TestInstructionToken, defaultTestParams.N_I> instructions{};
    instructions.fill(TestInstructionToken::STOP);
    instructions[0] = TestInstructionToken::WHILE; // Outer WHILE: 0
    instructions[1] = TestInstructionToken::WHILE; // Inner WHILE: 1
    instructions[2] = TestInstructionToken::INC;
    instructions[3] = TestInstructionToken::ENDW;   // Inner ENDW: 3 (matches 1)
    instructions[4] = TestInstructionToken::ENDW;   // Outer ENDW: 4 (matches 0)

    auto result = TestProgram::get_matching_indices(instructions);
    TEST_ASSERT_TRUE(result.has_value());
    if (result) {
        TEST_ASSERT_EQ((*result)[0], 4);
        TEST_ASSERT_EQ((*result)[4], 0);
        TEST_ASSERT_EQ((*result)[1], 3);
        TEST_ASSERT_EQ((*result)[3], 1);
    }
}

void GetMatchingIndicesTest_UnbalancedEndw() {
    std::array<TestInstructionToken, defaultTestParams.N_I> instructions{};
    instructions.fill(TestInstructionToken::STOP);
    instructions[0] = TestInstructionToken::ENDW; // Unmatched ENDW
    
    auto result = TestProgram::get_matching_indices(instructions);
    TEST_ASSERT_FALSE(result.has_value());
}

void GetMatchingIndicesTest_UnbalancedWhile() {
    std::array<TestInstructionToken, defaultTestParams.N_I> instructions{};
    instructions.fill(TestInstructionToken::STOP);
    instructions[0] = TestInstructionToken::WHILE; // Unmatched WHILE
    
    auto result = TestProgram::get_matching_indices(instructions);
    TEST_ASSERT_FALSE(result.has_value());
}


struct ProgramStateTester {
    TestProgramState state;
    ProgramStateTester() : state() {}

    void DefaultStateIsValid() {
        bool helpers_computed = state.compute_helpers();
        TEST_ASSERT_TRUE(helpers_computed);
        TEST_ASSERT_TRUE(state.is_valid());
    }

    void InvalidInstructionPointer() {
        state.instruction_ptr.index_int = defaultTestParams.N_I; // Out of bounds
        state.instruction_ptr.index[0] = TestOneHotToken::ZERO; // ensure vector reflects some change if needed by is_valid() directly
        // For OneHot<N>::is_valid(), just changing index_int is enough for it to fail
        TEST_ASSERT_FALSE(state.is_valid());
    }

    void LastInstructionNotStop() {
        state.instructions[defaultTestParams.N_I - 1] = TestInstructionToken::NEXT;
        bool helpers_computed = state.compute_helpers(); 
        TEST_ASSERT_TRUE(helpers_computed);
        TEST_ASSERT_FALSE(state.is_valid());
    }
    
    void UnbalancedLoopsInInstructions() {
        state.instructions[0] = TestInstructionToken::WHILE;
        bool helpers_computed = state.compute_helpers();
        TEST_ASSERT_FALSE(helpers_computed); 
        TEST_ASSERT_FALSE(state.is_valid()); 
    }

    void InvalidArrayDataValue() {
        state.array_data[0].n = (1ULL << defaultTestParams.B) + 10; 
        bool helpers_computed = state.compute_helpers();
        TEST_ASSERT_TRUE(helpers_computed);
        TEST_ASSERT_FALSE(state.is_valid());
    }

    void InvalidOutputPtrCache() {
        TestProgramState fresh_state; 
        fresh_state.compute_helpers(); 
        TEST_ASSERT_EQ(fresh_state.output_ptr, 0); 
        TEST_ASSERT_TRUE(fresh_state.is_valid());

        fresh_state.output_ptr = 1; 
        TEST_ASSERT_FALSE(fresh_state.is_valid());
    }

    void ExecuteNEXTNormal() {
        state.instructions[0] = TestInstructionToken::NEXT;
        state.instructions[1] = TestInstructionToken::STOP; 
        state.compute_helpers();
        TEST_ASSERT_TRUE(state.is_valid());

        size_t initial_ap_int = state.array_ptr.index_int;
        state.execute_instruction(); 

        TEST_ASSERT_EQ(state.array_ptr.index_int, initial_ap_int + 1);
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, 1); 
    }
    
    void ExecuteNEXTFreezeAtEnd() {
        state.instructions[0] = TestInstructionToken::NEXT;
        state.instructions[1] = TestInstructionToken::STOP;
        state.array_ptr.set(defaultTestParams.N_A - 1); 
        state.compute_helpers();
        TEST_ASSERT_TRUE(state.is_valid());
        
        TestProgramState original_state = state; // Copy for comparison
        state.execute_instruction(); 

        TEST_ASSERT_EQ(state.array_ptr.index_int, original_state.array_ptr.index_int);
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, original_state.instruction_ptr.index_int);
	    bool states_are_equal = state == original_state;
        TEST_ASSERT_TRUE(states_are_equal);
    }

    void ExecuteINCNormal() {
        state.instructions[0] = TestInstructionToken::INC;
        state.instructions[1] = TestInstructionToken::STOP;
        state.array_data[state.array_ptr.index_int].n = 5;
        state.compute_helpers();
        TEST_ASSERT_TRUE(state.is_valid());

        state.execute_instruction();

        TEST_ASSERT_EQ(state.array_data[state.array_ptr.index_int].n, 6);
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, 1);
    }

    void ExecuteINCFreezeAtMax() {
        state.instructions[0] = TestInstructionToken::INC;
        state.instructions[1] = TestInstructionToken::STOP;
        size_t max_val = (1ULL << defaultTestParams.B) - 1;
        state.array_data[state.array_ptr.index_int].n = max_val;
        state.compute_helpers();
        TEST_ASSERT_TRUE(state.is_valid());

        TestProgramState original_state = state; 
        state.execute_instruction(); 

        TEST_ASSERT_EQ(state.array_data[original_state.array_ptr.index_int].n, max_val); 
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, original_state.instruction_ptr.index_int);
    }
    
    void ExecutePUTNormal() {
        state.instructions[0] = TestInstructionToken::PUT;
        state.instructions[1] = TestInstructionToken::STOP;
        state.array_data[state.array_ptr.index_int].n = 7;
        state.compute_helpers();
        TEST_ASSERT_TRUE(state.is_valid());
        TEST_ASSERT_EQ(state.output_ptr, 0); 

        state.execute_instruction(); 

        TEST_ASSERT_TRUE(std::holds_alternative<TestNumber>(state.output_buffer[0]));
        if(std::holds_alternative<TestNumber>(state.output_buffer[0])) {
            TEST_ASSERT_EQ(std::get<TestNumber>(state.output_buffer[0]).n, 7);
        }
        TEST_ASSERT_EQ(state.output_ptr, 1); 
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, 1);
    }

    void ExecuteWHILEJump() {
        state.instructions[0] = TestInstructionToken::WHILE;
        state.instructions[1] = TestInstructionToken::INC; 
        state.instructions[2] = TestInstructionToken::ENDW;
        state.instructions[3] = TestInstructionToken::STOP;
        state.array_ptr.set(0);
        state.array_data[0].n = 0; 

        bool helpers_ok = state.compute_helpers();
        TEST_ASSERT_TRUE(helpers_ok);
        TEST_ASSERT_TRUE(state.is_valid());
        if(helpers_ok) TEST_ASSERT_EQ(state.matching_indices[0], 2);

        state.execute_instruction(); 
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, 3); 
    }

    void ExecuteWHILENoJump() {
        state.instructions[0] = TestInstructionToken::WHILE;
        state.instructions[1] = TestInstructionToken::INC; 
        state.instructions[2] = TestInstructionToken::ENDW;
        state.instructions[3] = TestInstructionToken::STOP;
        state.array_ptr.set(0);
        state.array_data[0].n = 1; 

        bool helpers_ok = state.compute_helpers();
        TEST_ASSERT_TRUE(helpers_ok);
        TEST_ASSERT_TRUE(state.is_valid());

        state.execute_instruction(); 
        TEST_ASSERT_EQ(state.instruction_ptr.index_int, 1); 
    }
};


int main() {
    std::cout << "Running tests..." << std::endl;

    RUN_TEST("OneHotTest", "Construction", OneHotTest_Construction);
    RUN_TEST("OneHotTest", "SetValue", OneHotTest_SetValue);
    RUN_TEST("OneHotTest", "IsValid", OneHotTest_IsValid);

    RUN_TEST("GetMatchingIndicesTest", "NoLoops", GetMatchingIndicesTest_NoLoops);
    RUN_TEST("GetMatchingIndicesTest", "SimpleLoop", GetMatchingIndicesTest_SimpleLoop);
    RUN_TEST("GetMatchingIndicesTest", "NestedLoops", GetMatchingIndicesTest_NestedLoops);
    RUN_TEST("GetMatchingIndicesTest", "UnbalancedEndw", GetMatchingIndicesTest_UnbalancedEndw);
    RUN_TEST("GetMatchingIndicesTest", "UnbalancedWhile", GetMatchingIndicesTest_UnbalancedWhile);
    
    ProgramStateTester ps_tester_is_valid;
    RUN_TEST("ProgramStateTest", "DefaultStateIsValid", [&](){ ps_tester_is_valid.DefaultStateIsValid(); });
    RUN_TEST("ProgramStateTest", "InvalidInstructionPointer", [&](){ ProgramStateTester t; t.InvalidInstructionPointer(); });
    RUN_TEST("ProgramStateTest", "LastInstructionNotStop", [&](){ ProgramStateTester t; t.LastInstructionNotStop(); });
    RUN_TEST("ProgramStateTest", "UnbalancedLoopsInInstructions", [&](){ ProgramStateTester t; t.UnbalancedLoopsInInstructions(); });
    RUN_TEST("ProgramStateTest", "InvalidArrayDataValue", [&](){ ProgramStateTester t; t.InvalidArrayDataValue(); });
    RUN_TEST("ProgramStateTest", "InvalidOutputPtrCache", [&](){ ProgramStateTester t; t.InvalidOutputPtrCache(); });

    ProgramStateTester ps_tester_exec;
    RUN_TEST("ProgramStateTest", "ExecuteNEXTNormal", [&](){ ProgramStateTester t; t.ExecuteNEXTNormal(); });
    RUN_TEST("ProgramStateTest", "ExecuteNEXTFreezeAtEnd", [&](){ ProgramStateTester t; t.ExecuteNEXTFreezeAtEnd(); });
    RUN_TEST("ProgramStateTest", "ExecuteINCNormal", [&](){ ProgramStateTester t; t.ExecuteINCNormal(); });
    RUN_TEST("ProgramStateTest", "ExecuteINCFreezeAtMax", [&](){ ProgramStateTester t; t.ExecuteINCFreezeAtMax(); });
    RUN_TEST("ProgramStateTest", "ExecutePUTNormal", [&](){ ProgramStateTester t; t.ExecutePUTNormal(); });
    RUN_TEST("ProgramStateTest", "ExecuteWHILEJump", [&](){ ProgramStateTester t; t.ExecuteWHILEJump(); });
    RUN_TEST("ProgramStateTest", "ExecuteWHILENoJump", [&](){ ProgramStateTester t; t.ExecuteWHILENoJump(); });

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Total tests run: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;

    return (total_tests - passed_tests == 0) ? 0 : 1;
}
