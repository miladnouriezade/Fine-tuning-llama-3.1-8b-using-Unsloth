# 🌌 Llama 3.1 8b for Astronomy: Fine-Tuning Project 🚀

Welcome to the **Llama 3.1 8b for Astronomy** repository! This project fine-tunes the state-of-the-art Llama 3.1 8b model to explore the cosmos and unlock insights from astronomical data. Whether you’re an AI enthusiast, a space lover, or both, you’re in the right place.

For an in-depth look at our methodology and results, check out our [presentation](docs/presentation.pdf).


## 🚀 Project Overview

This project demonstrates the complete pipeline for adapting a pre-trained Large Language Model (LLM) to a specific domain through continued pre-training, followed by instruction fine-tuning for question-answering capabilities.

**Developer**: Milad Nourizade  
**Email**: milad.nouriezade@gmail.com  
**Models**: Available at [🤗 huggingface.co/Dragonfluy](https://huggingface.co/Dragonfluy)

## 🎯 Objectives

1. **Domain Adaptation**: Perform continued pre-training on astronomical abstracts to adapt the model to the astronomy domain
2. **Instruction Fine-tuning**: Enable the model to follow instructions and answer questions effectively
3. **Efficient Training**: Utilize Unsloth framework for memory-efficient training on limited computational resources

## 🛠️ Tool Selection: Why Unsloth?

We chose [Unsloth](https://github.com/unslothai/unsloth) as our fine-tuning framework due to its exceptional efficiency:

- **2.2x faster** training compared to alternatives
- **70% less VRAM** usage
- **0% degradation** in accuracy for QLoRA (4-bit) and LoRA (16-bit)
- Optimized for single-GPU fine-tuning
- Free and open-source

### Comparison with Other Tools
For detailed comparisons with torchtune, axolotl, and other frameworks, see the [official comparison](https://github.com/unslothai/unsloth#-performance-benchmarking).

## 📋 Requirements

### Hardware
- GPU: Tesla T4 or better (minimum 16GB VRAM recommended)
- RAM: 16GB+ system memory

### Software
```bash
pip install unsloth==2024.11.7
pip install torch transformers datasets trl
```

### API Keys
- Hugging Face API token for model access and uploads

## 🏗️ Architecture & Approach

### Two-Phase Training Pipeline

```
Phase 1: Continued Pre-training
├── Base Model: LLaMA 3.1 8B
├── Dataset: Astronomical abstracts
├── Technique: LoRA with lm_head integration
└── Output: Domain-adapted model

Phase 2: Instruction Fine-tuning  
├── Input: Domain-adapted model
├── Dataset: Question-answering pairs
├── Technique: Standard LoRA on attention layers
└── Output: Instruction-following model
```

### Key Technical Decisions

**Model Selection**: LLaMA 3.1 8B
- High performance compared to proprietary models
- Long context window (128k tokens)
- Appropriate size for GPU memory constraints
- Strong performance on benchmarks ([LMArena](https://lmarena.ai/), [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard))

**Quantization Strategy**: 4-bit quantization
- Reduces memory usage for limited VRAM
- Faster model loading and inference
- Minimal performance degradation

## 📊 Datasets

### Phase 1: Continued Pre-training
- **Dataset**: [UniverseTBD/arxiv-astro-abstracts-all](https://huggingface.co/datasets/UniverseTBD/arxiv-astro-abstracts-all)
- **Size**: 5,000 samples (first 5k rows)
- **Split**: 95% train, 5% validation
- **Purpose**: Domain adaptation to astronomical literature

### Phase 2: Instruction Fine-tuning
- **Dataset**: [daven3/geosignal](https://huggingface.co/datasets/daven3/geosignal)
- **Size**: 5,000 samples
- **Split**: 99.5% train, 0.5% validation
- **Format**: Instruction-Input-Response triplets

## 🔧 Training Configuration

### Phase 1: Continued Pre-training
```python
# LoRA Configuration
r = 32                    # Higher rank for complex learning
target_modules = [        # Include lm_head for domain adaptation
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head"
]
lora_alpha = 16

# Training Parameters
learning_rate = 2e-5
embedding_learning_rate = 1e-5  # 10x smaller than main LR
batch_size = 2
gradient_accumulation_steps = 8
max_steps = 20
```

### Phase 2: Instruction Fine-tuning
```python
# LoRA Configuration  
r = 16                    # Standard rank for instruction following
target_modules = [        # Exclude lm_head and embed_tokens
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Parameters
learning_rate = 2e-4      # Higher LR for instruction tuning
batch_size = 2
gradient_accumulation_steps = 4
max_steps = 30
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/miladnouriezade/Fine-tuning-llama-3.1-8b-using-Unsloth.git
cd Fine-tuning-llama-3.1-8b-using-Unsloth
pip install -r requirements.txt
```

### 2. Configure Hugging Face Access
```python
from huggingface_hub import login
login("your_huggingface_token_here")
```

### 3. Run Training
```bash
# Phase 1: Domain adaptation
python continued_pretraining.py

# Phase 2: Instruction fine-tuning  
python instruction_finetuning.py
```

### 4. Inference
```python
from unsloth import FastLanguageModel

# Load the final model
model, tokenizer = FastLanguageModel.from_pretrained(
    "Dragonfluy/astro_instruct_llama_3.1_8b"
)

# Generate responses
FastLanguageModel.for_inference(model)
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
```

## 📈 Results

### Training Performance
- **Training Time**: ~20-30 steps per phase
- **Memory Usage**: Optimized for Tesla T4 (16GB VRAM)
- **Loss Trends**: Smooth convergence for both phases

### Model Capabilities
1. **Domain Knowledge**: Successfully generates astronomy-related abstracts
2. **Instruction Following**: Responds appropriately to structured prompts
3. **Context Understanding**: Maintains coherence in long-form responses

### Example Outputs

**Domain Adaptation**:
```
Input: "A detailed analysis of Reuven Ramaty High Energy Solar Spectroscopic Imager (RHESSI),"
Output: [Astronomy-focused continuation with technical terminology]
```

**Instruction Following**:
```
Instruction: "Give me a bulleted list of the past 10 Masters Tournament Champions."
Output: [Structured list format with proper formatting]
```

## 🔍 Key Insights

### LoRA Configuration Insights
- **Continued Pre-training**: Include `lm_head` and use higher rank (r=32) for complex domain learning
- **Instruction Fine-tuning**: Standard attention layers with moderate rank (r=16)
- **Finding**: LoRA rank impact is minimal when applied to all relevant layers

### Training Strategies
- **Embedding Learning Rate**: Use 2-10x smaller rate than main learning rate for stable continued pre-training
- **Gradient Checkpointing**: "unsloth" mode provides optimal memory-performance trade-off
- **Data Preparation**: Always include EOS tokens to prevent infinite generation

## 🚧 Limitations & Future Work

### Current Limitations
- Limited training steps (20-30) due to computational constraints
- Excluded `embed_tokens` from continued pre-training due to memory limitations
- Small dataset sizes (5k samples each phase)

### Evaluation Gaps
- Missing comprehensive benchmarking
- No perplexity measurements
- Limited qualitative assessment

### Future Improvements

#### Technical Enhancements
- [ ] Include `embed_tokens` in LoRA adapters with more compute
- [ ] Experiment with higher ranks and larger datasets
- [ ] Implement comprehensive evaluation framework
- [ ] Add perplexity measurements for domain adaptation assessment

#### Research Directions
- [ ] Compare with specialized astronomy models (AstroLLaMA)
- [ ] Explore QDyLoRA for dynamic rank adaptation
- [ ] Implement automated evaluation metrics
- [ ] Study scaling effects with larger datasets

#### Evaluation Framework
```python
# Planned evaluation metrics
- Domain-specific perplexity
- Instruction-following accuracy  
- Benchmark performance (MMLU, etc.)
- Human evaluation for astronomy tasks
```

## 📚 References & Related Work

### Key Papers
- **AstroLLaMA**: Towards Specialized Foundation Models in Astronomy
- **AstroMLab 3**: Achieving GPT-4o Level Performance in Astronomy with a 8B-Parameter LLM
- **QLoRA**: Efficient Finetuning of Quantized LLMs
- **QDyLoRA**: Quantized Dynamic Low-Rank Adaptation

### Evaluation Tools
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): Framework for LLM evaluation
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): Model performance rankings

## 📄 License

This project is open source. Please check individual model licenses on Hugging Face.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Contact

**Milad Nourizade**  
📧 milad.nouriezade@gmail.com  
🤗 [Hugging Face Profile](https://huggingface.co/Dragonfluy)

---

*This project demonstrates efficient LLM fine-tuning techniques using modern tools like Unsloth, making advanced NLP accessible with limited computational resources.*
