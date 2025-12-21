# Fine-Tuning LLaMA 3.2-3B with LoRA for Dialogue Summarization

This project demonstrates how to fine-tune a large language model (LLaMA 3.2-3B) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters for dialogue summarization tasks.

## 📋 Overview

The pipeline fine-tunes a 4-bit quantized LLaMA 3.2-3B model on the DialogSum dataset to generate concise summaries of conversations. By using LoRA (Low-Rank Adaptation), we train only ~0.5% of the model's parameters while achieving meaningful improvements.

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────────┤
│  DialogSum Dataset → Filter (1/100) → Tokenize → Label Masking  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL SETUP                              │
├─────────────────────────────────────────────────────────────────┤
│  LLaMA 3.2-3B (4-bit quantized) → Apply LoRA Adapters           │
│  Target modules: q_proj, k_proj, v_proj, o_proj                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                 │
├─────────────────────────────────────────────────────────────────┤
│  Gradient Checkpointing + FP16 + Cosine LR Schedule             │
│  Learning Rate: 2e-4 | Epochs: 5 | Effective Batch Size: 4      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        EVALUATION                               │
├─────────────────────────────────────────────────────────────────┤
│  Compare Original vs Fine-tuned using BERTScore (P, R, F1)      │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Key Components

### 1. Model & Quantization

-   **Base Model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
-   **Quantization**: 4-bit using `bitsandbytes` library
-   **Architecture**: Decoder-only (Causal LM)

### 2. LoRA Configuration

| Parameter        | Value                  | Description                   |
| ---------------- | ---------------------- | ----------------------------- |
| `r`              | 16                     | Rank of the low-rank matrices |
| `lora_alpha`     | 32                     | Scaling factor (2× rank)      |
| `target_modules` | q, k, v, o projections | Attention layers to adapt     |
| `lora_dropout`   | 0.1                    | Dropout for regularization    |

### 3. Training Configuration

| Parameter              | Value   |
| ---------------------- | ------- |
| Batch size             | 1       |
| Gradient accumulation  | 4 steps |
| Learning rate          | 2e-4    |
| Epochs                 | 5       |
| Precision              | FP16    |
| Gradient checkpointing | Enabled |

### 4. Tokenization Strategy

-   **Training**: Right padding (standard for causal LM)
-   **Generation**: Left padding (better for batch inference)
-   **Label Masking**: Prompt tokens masked with `-100` (not included in loss)

## 📊 Dataset

**DialogSum** - A dialogue summarization dataset containing:

-   Multi-turn conversations
-   Human-written summaries
-   Filtered to 1/100 samples for faster training

## 🚀 Usage

### Prerequisites

```bash
pip install transformers peft bitsandbytes datasets evaluate bert_score
```

### Run the Pipeline

1. Open `final_question.ipynb`
2. **Restart kernel** to clear GPU memory
3. Run all cells sequentially

### Prompt Format

```
Summarize the following conversation.

[DIALOGUE]

Summary: [GENERATED SUMMARY]
```

## 📈 Evaluation

The pipeline compares the original (non-fine-tuned) model against the PEFT model using **BERTScore**:

| Metric    | Description                                   |
| --------- | --------------------------------------------- |
| Precision | How much of the generated summary is relevant |
| Recall    | How much of the reference is captured         |
| F1        | Harmonic mean of precision and recall         |

## 💡 Memory Optimization Techniques

1. **4-bit Quantization**: Reduces model size by ~4×
2. **Gradient Checkpointing**: Trades compute for memory
3. **LoRA**: Only trains ~0.5% of parameters
4. **FP16 Training**: Halves memory for activations

## 📁 Project Structure

```
tp-fine-tuning/
├── final_question.ipynb    # Main fine-tuning notebook
├── tp-fine-tuning.ipynb    # Additional experiments
├── README.md               # This file
├── resources/              # Additional resources
└── training-output/        # Saved LoRA adapters
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

## 🔗 References

-   [LoRA Paper](https://arxiv.org/abs/2106.09685)
-   [PEFT Library](https://github.com/huggingface/peft)
-   [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)
-   [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
