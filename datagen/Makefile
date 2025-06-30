CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
LDFLAGS = -lz

INC_DIR = includes
SRC_DATA_DIR = datagen
SRC_TEST_DIR = tests
BIN_DIR = bin
LIB_DIR = $(CURDIR)/libs

CPPFLAGS = -I$(INC_DIR) -I$(LIB_DIR)/json/include -I$(LIB_DIR)/cnpy

SRC_CNPY = $(SRC_DATA_DIR)/cnpy.cpp
GEN_SRC = $(SRC_DATA_DIR)/gen.cpp
GEN2_SRC = $(SRC_DATA_DIR)/gen2.cpp
TEST_SRC = $(SRC_TEST_DIR)/tests.cpp

OBJ_CNPY = $(BIN_DIR)/cnpy.o
GEN_OBJ = $(BIN_DIR)/gen.o
GEN2_OBJ = $(BIN_DIR)/gen2.o
TEST_OBJ = $(BIN_DIR)/tests.o

GEN2_EXE = $(BIN_DIR)/generate_data2
TEST_EXE = $(BIN_DIR)/run_tests

all: $(GEN_EXE) $(TEST_EXE) $(GEN2_EXE)

$(GEN_EXE): $(GEN_OBJ) $(OBJ_CNPY)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built data generator: $@"

$(GEN2_EXE): $(GEN2_OBJ) $(OBJ_CNPY)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built data generator: $@"

$(TEST_EXE): $(TEST_OBJ) $(OBJ_CNPY)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built tests: $@"

$(GEN_OBJ): $(GEN_SRC) $(INC_DIR)/program.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(GEN2_OBJ): $(GEN2_SRC) $(INC_DIR)/program.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_SRC) $(INC_DIR)/program.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(OBJ_CNPY): $(SRC_CNPY) $(LIB_DIR)/cnpy/cnpy.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

.PHONY: all clean run_generator run_generator2 run_tests

clean:
	@echo "Cleaning up..."
	rm -f $(BIN_DIR)/*
	@echo "Done."

run_generator: $(GEN_EXE)
	@echo "Running data generator..."
	./$(GEN_EXE) ./data_output gen_run1 100 42

run_generator2: $(GEN2_EXE)
	@echo "Running data generator..."
	./$(GEN2_EXE) ./data_output gen2_run1 100 42

run_tests: $(TEST_EXE)
	@echo "Running tests..."
	./$(TEST_EXE)
