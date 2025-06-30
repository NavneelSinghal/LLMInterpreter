# LLM Interpreter

This project explores the capability of Large Language Models (LLMs) to interpret and simulate a simple, custom-designed esoteric programming language. The core idea is to investigate state-tracking and execution simulation within different LLM architectures.

## About The Project

The primary objective is to determine whether current LLM architectures, such as Transformers and RNNs, starting with a transformer baseline, can effectively simulate a custom programming language (one that is Turing-complete with unbounded buffers) by tracking its state transitions.

The methodology involves iteratively pretraining an LLM on state transitions, gradually increasing the language's complexity. By observing the model's behavior at each stage, we aim to understand the limits and capabilities of different architectures in a complexity-theoretic sense.

### Key Goals

* **Architectural Comparison:** Compare the performance of modern LLM architectures, with the potential to build more expressive architectures if needed.
* **Iterative Language Development:** Systematically build a programming language, from simple operations to complex control flow, that an LLM can interpret with high accuracy.
* **Mechanism Analysis:** Investigate the internal mechanisms the LLM learns to handle different programming constructs and state manipulations.

## The Language

A custom esoteric programming language, similar in spirit to Brainfck, is used. A key feature is the tight coupling of the program's instructions with its execution state.

### State Representation

The state $S_t$ of a program at a given time $t$ is represented as a single, contiguous sequence of tokens:

$$S_t = [ \mathrm{instruction\_ptr}_t ] [ \mathrm{instructions} ] [ \mathrm{array\_ptr}_t ] [ \mathrm{array}_t ] [ \mathrm{input}_t ] [ \mathrm{output}_t ]$$

Where:
* **`instructions`**: A buffer of size $N_I$ containing the program's code, padded with at least one `STOP` token.
* **`instruction_ptr_t`**: A one-hot encoded vector of size $N_I$ pointing to the current instruction in `instructions`.
* **`array_t`**: A data buffer of size $N_A$. Each element is a number less than $2^B$ (where $B=4$).
* **`array_ptr_t`**: A one-hot encoded vector of size $N_A$ pointing to the current cell in `array_t`.
* **`input_t`**: An input buffer of size $N_{\text{input}}$. It contains numbers or `EOI` (End of Input) tokens.
* **`output_t`**: An output buffer of size $N_{\text{output}}$. It starts filled with `EOO` (End of Output) tokens, which are replaced by numbers as the program executes.

If any operation is invalid (e.g., overflow, out-of-bounds access, reading from empty input), the program **freezes**, meaning its state remains unchanged in the next step.

### Instruction Set

| Instruction | Description |
| :--- | :--- |
| `NEXT` | Moves the `array_ptr` to the right. Freezes if at the end of the array. |
| `PREV` | Moves the `array_ptr` to the left. Freezes if at the beginning of the array. |
| `INC` | Increments the value in `array_t` at the `array_ptr`. Freezes on overflow. |
| `DEC` | Decrements the value in `array_t` at the `array_ptr`. Freezes on underflow (below zero). |
| `PUT` | Appends the value from `array_t` at the `array_ptr` to the `output_t` buffer. Freezes if the output buffer is full. |
| `GET` | Reads a token from the `input_t` buffer and writes it to `array_t` at the `array_ptr`. Freezes if the input buffer is empty (`EOI`). |
| `WHILE` | If the value at `array_ptr` is zero, jumps the `instruction_ptr` to after the matching `ENDW`. Otherwise, advances to the next instruction. |
| `ENDW` | If the value at `array_ptr` is non-zero, jumps the `instruction_ptr` to after the matching `WHILE`. Otherwise, advances to the next instruction. |
| `STOP` | Halts program execution. |

### Vocabulary Tokens

| Token | Type | Description |
| :--- | :--- | :--- |
| `NUM_i` | Data | Represents a number $i$, where $0 \le i < 2^B$. |
| `0`, `1` | State | Used for one-hot encoding of pointers. |
| `EOI` | I/O | Marks the end of valid data in the input buffer. |
| `EOO` | I/O | Marks the end of written data in the output buffer. |
| `{`, `}` | Structure | Signals the beginning and end of a program state. |
| `[`, `]` | Structure | Signals the beginning and end of a block within the state. |

## Models

The project currently focuses on experimenting with a transformer baseline. Optimizers like AdamW and Muon have been experimented with, with Muon performing better.

## Dataset

### Data Generation

The training data consists of synthetic single-step state transitions ($S_t \rightarrow S_{t+1}$). The data is generated using a C++ program located in `datagen/`.

The generation process involves:
1.  **Random Program Sampling:** Programs with valid syntax (e.g., balanced `WHILE`/`ENDW` loops) are randomly generated uniformly from the space of all valid programs.
2.  **Random State Sampling:** For each program, a valid initial state ($S_t$) is sampled with randomized buffer contents and pointer locations.
3.  **Execution:** The single-step transition is executed by the C++ interpreter to produce the next state ($S_{t+1}$).
4.  **Serialization:** The pair of states $(S_t, S_{t+1})$ is tokenized and saved in `.npy` format for efficient loading in PyTorch.

### Training Data Strategies

* **Random Generation:** Creating a diverse set of random, valid programs.
* **Genetic Algorithms:** (to be tried) An approach inspired by [generating Brainfck code with genetic algorithms](https://blog.lapinozz.com/project/2016/03/27/generating-brainfuck-code-using-genetic-algorithm.html) to evolve more "meaningful" or complex programs.

## Project Structure

```
.
├── datagen
│   ├── datagen
│   │   ├── cnpy.cpp
│   │   ├── dataloader.py
│   │   ├── gen2.cpp
│   │   └── gen.cpp         # Main C++ executable for generating the dataset
│   ├── includes
│   │   └── program.hpp     # Core C++ logic for the language, state, and sampling
│   ├── libs
│   │   ├── cnpy
│   │   └── json
│   ├── Makefile
│   └── tests
│       └── tests.cpp       # Unit tests for the C++ data generator
├── README.md
├── runs
│   └── train.out
└── src
    ├── config_muon.py      # Dataclasses for model, training, and data configuration
    ├── dataset.py          # PyTorch Dataset class for loading .npy data
    ├── model.py            # GPT model implementation with RoPE, KV Caching
    ├── muon.py             # Muon optimizer implementation
    └── train.py            # Main training script

```

## Getting Started

### Prerequisites

* A C++ compiler with C++20 support (e.g., GCC, Clang)
* Python 3.12
* PyTorch

### 1. Data Generation

First, compile the data generator (using the provided Makefile) and run it to create the training and validation datasets.

### 2\. Training

Once the data is generated, you can start training the model using the `train.py` script.

```bash
# Navigate to the source directory
cd src/

# Run training (single-node, multi-GPU example with torchrun)
# Adjust --nproc_per_node to the number of available GPUs
torchrun --standalone --nproc_per_node=2 train.py
```

### Configuration

All training parameters are managed via the `TrainConfig` class in `src/config_muon.py`. You can modify this file to change:

  * Model architecture (`d_model`, `n_layers`, etc.)
  * Training hyperparameters (`learning_rate`, `batch_size`, etc.)
  * Data and logging paths.

## Current Status & Observations

### Initial Runs

Early experiments with the Transformer architecture show that the model learns the single-step state transition task relatively quickly, indicating the feasibility of this approach.

## Future Work and Open Questions

This section outlines planned improvements and areas for further investigation.

### Architecture & Model Exploration

  - **RNNs:** Complete the training and evaluation for modern RNN-style architectures (e.g. linear attention variants).
  - **Other models:** Experiment with Universal Transformer-style and encoder-decoder setups since input and output sequences have a fixed size.
  - **Diffusion Models:** Investigate the potential of diffusion models for this task.
  - **Mechanistic Interpretability:** Use tools like `TransformerLens` to analyze the circuits the model learns for different instructions.
  - **Positional Encodings:** Test PaTH attention position encodings, which may be more suitable for multi-state transition prediction.

### Data Enhancements

  - **Better Multi-Step Data:** For multi-step prediction, filter out samples where the program state gets stuck in a loop to create more informative training data.
  - **Harder Data:** Introduce variable-length buffers instead of fixed sizes.
  - **Genetic/Evolved Data:** Implement the genetic algorithm to generate more complex and interesting programs for the training set.
  - **Length Generalization:** Create specific benchmarks to test the model's ability to generalize to longer programs, arrays, and I/O buffers.

### Training & Evaluation

  - **Multi-Step Transitions:** Pretrain the model on chains of state transitions ($S_t \rightarrow S_{t+1} \rightarrow ... \rightarrow S_{t+n}$) instead of just single steps.
  - **CoT & RL:** After pretraining, use RL to improve the model's performance.
  - **Error Tracking:** Augment the model to predict and output error states (e.g., `OVERFLOW_ERROR`, `INPUT_EOF_ERROR`).
  - **Evaluation Metrics:** Develop more comprehensive evaluation metrics beyond simple accuracy, possibly including partial credit for correct state components.

### Language & State Representation

  - **Optimal Layout:** Investigate whether token spacing or different serialization formats for the program state can improve model performance by giving it "room to think."
  - **Making the layout harder:** Investigate what layouts can an LLM learn - currently the states are in what one could call a CoT format, the other end of this spectrum would be predicting just the output buffer after a few steps, which is hard for the transformer baseline to learn. Can we figure out the limits on the kinds of representations each architecture/optimizer is able to learn to grok?
