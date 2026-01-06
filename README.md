# EC2ER: Enhancing Emotional Intelligence in Large Language Models

[![EmoBench](https://img.shields.io/badge/Benchmark-EmoBench-blue)](https://aclanthology.org/2024.acl-long.326/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Improving LLM Performance on Emotional Understanding and Application Tasks**

This repository contains our work on enhancing the emotional intelligence capabilities of Large Language Models (LLMs) through synthetic data generation and fine-tuning, evaluated on the [EmoBench benchmark](https://aclanthology.org/2024.acl-long.326/). Paper is available [here](https://arxiv.org/abs/2601.01407).

## 📋 Table of Contents

- [Overview](#overview)
- [EmoBench Benchmark](#emobench-benchmark)
- [Our Approach](#our-approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Pipelines](#pipelines)
- [Citation](#citation)

## 🎯 Overview

Large Language Models have shown impressive capabilities across many tasks, but their emotional intelligence—particularly in understanding complex emotions and applying appropriate emotional responses—remains a challenge. This project tackles this limitation through:

1. **Synthetic Data Generation**: Creating high-quality EmoBench-style training examples using multi-agent dialogue systems
2. **Fine-tuning**: Training open-source LLMs on generated data to improve emotional understanding
3. **Comprehensive Evaluation**: Testing on EmoBench's Emotional Understanding (EU) and Emotional Application (EA) tasks

## 📊 EmoBench Benchmark

EmoBench is a comprehensive benchmark for evaluating the Emotional Intelligence of LLMs, consisting of **400 hand-crafted scenarios** across two key tasks:

### Emotional Understanding (EU)
Tests the model's ability to recognize emotions and their causes in complex scenarios. Contains 4 categories:
- **Complex Emotions**: Understanding mixed or nuanced emotional states
- **Emotional Cues**: Interpreting subtle emotional signals
- **Personal Beliefs and Experiences**: Recognizing how background influences emotions
- **Perspective-Taking**: Understanding emotions from others' viewpoints

**Example:**
```
Scenario: "Dorea was trying to cook a Baklava. When she took it out of the oven, 
the Baklava was ruined. Her daughter came home, tasted it and gave a thumbs-up."

Question: What emotion is Dorea feeling and why?
Answer: Delight - Her daughter enjoyed the Baklava despite it being ruined
```

### Emotional Application (EA)
Evaluates the model's ability to recommend effective emotional responses or actions in emotionally charged dilemmas. Organized by:
- **Relationship Type**: Personal vs. Social
- **Problem Type**: Self-focused vs. Others-focused

**Example:**
```
Scenario: "Sarah found out that her younger brother is being bullied at school 
but he begged her not to tell their parents."

Question: What should Sarah do?
Best Action: Suggest her brother talk to a teacher or school counselor
```

## 🚀 Our Approach

We implemented a two-pronged strategy to improve LLM performance on EmoBench:

### 1. Synthetic Data Generation

We developed two innovative pipelines for generating EmoBench-style training data:

#### Multi-Agent Dialogue System (MADS)
- **Architecture**: Therapist-Client dialogue simulation with supervisor oversight
- **Agents**:
  - Background Generator: Creates detailed patient personas
  - Client Agent: Embodies persona with emotional depth
  - Therapist Agent: Conducts structured therapeutic conversations
  - Supervisor: Ensures dialogue quality and completion
- **Output**: Rich emotional dialogues that are distilled into EU/EA examples
- **Advantages**: Captures complex emotional dynamics and natural conversation flow

#### Single Conversation Generator
- **Approach**: Direct generation of EmoBench-style scenarios
- **Focus**: Category-aware upsampling to balance dataset
- **Output**: Targeted EU/EA examples with specific emotional categories
- **Advantages**: Faster generation, precise category control

### 2. Fine-tuning Strategy

We fine-tuned multiple open-source LLMs using:
- **Base Models**: Llama-3.1-8B, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Gemma-7B-IT
- **Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Training Data**: Synthetically generated EmoBench-style examples
- **Framework**: HuggingFace Transformers with custom training loops

## 📈 Results

Our approach demonstrated significant improvements across both tasks and multiple models:

### Emotional Application (EA) Results

| Model | Personal-Others | Personal-Self | Social-Others | Social-Self | **Overall** |
|-------|----------------|---------------|---------------|-------------|-------------|
| **Qwen2.5-7B-Instruct** | 0.62 | 0.78 | 0.60 | 0.72 | **0.680** |
| Finetuned-Qwen2.5-7B | 0.60 | 0.80 | 0.60 | 0.74 | **0.685** ✅ |
| **Mistral-7B-Instruct-v0.3** | 0.38 | 0.46 | 0.30 | 0.48 | **0.405** |
| Finetuned-Mistral-7B | 0.66 | 0.74 | 0.50 | 0.60 | **0.625** ⬆️ +54% |
| **Llama-3.1-8B** | 0.12 | 0.12 | 0.12 | 0.16 | **0.130** |
| Finetuned-Llama-3.1-8B | 0.54 | 0.72 | 0.54 | 0.62 | **0.605** ⬆️ +365% |
| **Gemma-7B-IT** | 0.54 | 0.60 | 0.50 | 0.62 | **0.565** |
| Finetuned-Gemma-7B | 0.54 | 0.60 | 0.48 | 0.64 | **0.565** |

### Emotional Understanding (EU) Results

| Model | Complex Emotions | Emotional Cues | Personal Beliefs | Perspective-Taking | **Overall** |
|-------|-----------------|----------------|-----------------|-------------------|-------------|
| **Qwen2.5-7B-Instruct** | 0.429 | 0.393 | 0.286 | 0.209 | **0.310** |
| Finetuned-Qwen2.5-7B | 0.388 | 0.357 | 0.286 | 0.194 | **0.290** |
| **Mistral-7B-Instruct-v0.3** | 0.143 | 0.179 | 0.054 | 0.090 | **0.105** |
| Finetuned-Mistral-7B | 0.224 | 0.179 | 0.107 | 0.134 | **0.155** ⬆️ +48% |
| **Llama-3.1-8B** | 0.082 | 0.000 | 0.036 | 0.030 | **0.040** |
| Finetuned-Llama-3.1-8B | 0.265 | 0.179 | 0.161 | 0.239 | **0.215** ⬆️ +438% |
| **Gemma-7B-IT** | 0.245 | 0.250 | 0.161 | 0.149 | **0.190** |
| Finetuned-Gemma-7B | 0.204 | 0.214 | 0.179 | 0.149 | **0.180** |

### Key Findings

✨ **Dramatic improvements** for models with initially poor performance (Llama-3.1-8B, Mistral-7B)
- Llama-3.1-8B: +365% on EA, +438% on EU
- Mistral-7B: +54% on EA, +48% on EU

⚠️ **Ceiling effects** observed for already high-performing models
- Qwen2.5 and Gemma showed marginal or slight decreases
- Suggests these models may already be near optimal for EmoBench tasks

🎯 **Task-specific patterns**
- EA tasks showed more consistent improvements across models
- EU tasks demonstrated higher variance in model performance

## 📁 Project Structure

```
EC2ER/
├── README.md                          # This file
├── Report final.pdf                   # Detailed project report
│
├── benchmarking/                      # EmoBench evaluation framework
│   ├── data/
│   │   ├── EA.jsonl                  # Emotional Application test set (400 examples)
│   │   └── EU.jsonl                  # Emotional Understanding test set (400 examples)
│   ├── results/
│   │   ├── EA/                       # EA evaluation results
│   │   │   ├── leaderboard.json     # EA performance summary
│   │   │   └── *.jsonl              # Model predictions
│   │   └── EU/                       # EU evaluation results
│   │       ├── leaderboard.json     # EU performance summary
│   │       └── *.jsonl              # Model predictions
│   ├── src/
│   │   ├── main.py                   # Main evaluation script
│   │   ├── model.py                  # LLM wrapper classes
│   │   ├── data.py                   # Data loading and evaluation
│   │   └── utils.py                  # Utility functions
│   ├── requirements.txt              # Benchmarking dependencies
│   └── README.md                     # Benchmarking documentation
│
├── generation/                        # Synthetic data generation pipelines
│   ├── Mult-Agent_Dialogue_System/
│   │   └── emobench_mads_pipeline.ipynb    # Multi-agent dialogue generation
│   └── Single_Conversation_Pipeline/
│       └── Single_Conversation_Generator.ipynb  # Direct scenario generation
│
└── finetuning/                        # Model fine-tuning
    └── finetune_llm_emobench.ipynb   # Fine-tuning notebook (LoRA)
```

## 🚦 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU (recommended for fine-tuning)
nvidia-smi
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd EC2ER
```

2. **Set up benchmarking environment**
```bash
cd benchmarking
pip install -r requirements.txt
```

3. **Configure API keys** (if using API models)
```bash
# Create .env file in benchmarking directory
echo "API_KEY=your_api_key_here" > .env
echo "API_URL=your_api_endpoint" >> .env  # Optional for OpenAI-compatible APIs
```

### Quick Start: Running Benchmarks

#### Evaluate a Hugging Face Model

```bash
cd benchmarking

# Evaluate on all tasks and languages
python src/main.py \
    --model_type HF \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lang all \
    --task all \
    --device 0

# Evaluate on specific task (EA only)
python src/main.py \
    --model_type HF \
    --model_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --task EA \
    --lang en \
    --device 0
```

#### Evaluate an OpenAI Model

```bash
python src/main.py \
    --model_type openai \
    --model_path "gpt-4" \
    --lang all \
    --task all
```

#### Evaluate with Chain-of-Thought

```bash
python src/main.py \
    --model_type HF \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --use_cot \
    --device 0
```

#### Evaluate Existing Predictions

```bash
# If you already have model predictions and just want to compute metrics
python src/main.py \
    --model_type HF \
    --model_path "your-model-name" \
    --eval_only
```

### Command-Line Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--model_type` | str | "openai" | openai, openai-compatible, HF | Type of model API to use |
| `--model_path` | str | "gpt-4o" | - | Model name or path |
| `--lang` | str | "all" | en, zh, all | Language for evaluation |
| `--task` | str | "all" | EU, EA, all | Task to evaluate |
| `--device` | int | -1 | - | GPU device ID (-1 for CPU) |
| `--iter_num` | int | 3 | - | Number of generation iterations |
| `--num_retries` | int | 5 | - | Number of retries for failed requests |
| `--use_cot` | flag | False | - | Enable chain-of-thought reasoning |
| `--eval_only` | flag | False | - | Only evaluate existing predictions |
| `--is_llama` | flag | False | - | Use Llama-specific prompt format |

## 🔬 Pipelines

### 1. Synthetic Data Generation

#### Multi-Agent Dialogue System (MADS)

**Location**: `generation/Mult-Agent_Dialogue_System/emobench_mads_pipeline.ipynb`

**Purpose**: Generate rich, contextual emotional scenarios through simulated therapy sessions

**How it works**:
1. Background Generator creates detailed patient personas
2. Client Agent embodies the persona in dialogue
3. Therapist Agent conducts structured conversation
4. Supervisor monitors and signals completion
5. Single-agent extractor distills dialogues into EU/EA examples

**To run**:
```bash
# Open the notebook in Jupyter or Google Colab
jupyter notebook generation/Mult-Agent_Dialogue_System/emobench_mads_pipeline.ipynb

# Or use Google Colab
# Upload the notebook and follow the instructions
```

**Key features**:
- 🎭 Realistic multi-turn emotional dialogues
- 🎯 Category-aware generation for balanced datasets
- 🔄 Quality control through supervisor agent
- 📊 Automatic extraction to EmoBench format

#### Single Conversation Generator

**Location**: `generation/Single_Conversation_Pipeline/Single_Conversation_Generator.ipynb`

**Purpose**: Direct generation of EmoBench-style scenarios with category control

**How it works**:
1. Category-aware upsampling identifies underrepresented categories
2. LLM generates scenarios in EmoBench format
3. JSON validation ensures format correctness
4. Output saved in JSONL format

**To run**:
```bash
jupyter notebook generation/Single_Conversation_Pipeline/Single_Conversation_Generator.ipynb
```

**Key features**:
- ⚡ Fast generation pipeline
- 🎯 Precise category targeting
- ✅ Built-in format validation
- 📈 Automatic balancing of dataset categories

### 2. Fine-tuning Pipeline

**Location**: `finetuning/finetune_llm_emobench.ipynb`

**Purpose**: Fine-tune open-source LLMs on EmoBench-style data using LoRA

**Requirements**:
- GPU with at least 16GB VRAM (for 7B models)
- CUDA-compatible PyTorch
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)

**How it works**:
1. Load pre-trained model and tokenizer
2. Configure LoRA adapters for efficient training
3. Prepare training data in instruction format
4. Fine-tune with custom loss functions
5. Save and evaluate fine-tuned model

**To run**:
```bash
# Open notebook
jupyter notebook finetuning/finetune_llm_emobench.ipynb

# Follow the cells to:
# 1. Install dependencies
# 2. Load your model
# 3. Configure LoRA parameters
# 4. Train on synthetic data
# 5. Save fine-tuned weights
```

**Supported models**:
- Llama 3.1 (8B)
- Mistral (7B Instruct v0.3)
- Qwen 2.5 (7B Instruct)
- Gemma (7B IT)
- Any HuggingFace-compatible causal LM

**Key features**:
- 💾 Parameter-efficient training with LoRA
- 📊 Training metrics and visualization
- 🔄 Checkpoint saving and resumption
- ⚡ Mixed-precision training support
- 🎯 Task-specific loss functions

## 📊 Understanding the Results

### Reading the Leaderboard

The `results/*/leaderboard.json` files contain performance metrics for each model:

```json
{
  "Model-Name": {
    "en": {
      "Category-1": 0.65,
      "Category-2": 0.72,
      "Overall": 0.685
    }
  }
}
```

### Result Files

Individual model predictions are saved as JSONL files in `results/EA/` and `results/EU/`:
- Each line contains the model's prediction for one example
- Includes the generated response and extracted answer
- Can be re-evaluated using `--eval_only` flag

## 🎓 Key Takeaways

1. **Synthetic data generation is effective** for improving emotional intelligence in LLMs
2. **Fine-tuning works best** for models with initially poor performance
3. **Multi-agent systems** can generate high-quality emotional dialogue data
4. **EmoBench** provides a comprehensive benchmark for emotional intelligence
5. **Task-specific patterns** suggest different strategies may be needed for EU vs EA

## Contributors

- [Arjhun Sreedar](https://github.com/kernelism)
- [Laukik Patade](https://github.com/LaukikPatade)
- [Rohan Pillay](https://github.com/pillayroh)

## 📧 Cite

If you build upon our work, do cite our work:
```
@misc{sreedar2026emotionclassificationemotionalreasoning,
      title={From Emotion Classification to Emotional Reasoning: Enhancing Emotional Intelligence in Large Language Models}, 
      author={Arjhun Sreedar and Rohan Pillay and Laukik Patade},
      year={2026},
      eprint={2601.01407},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.01407}, 
}
```

## 📄 License

This project builds upon EmoBench, which is licensed under its original license. See the [benchmarking/LICENSE](benchmarking/LICENSE) file for details.
