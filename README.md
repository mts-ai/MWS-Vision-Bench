# 🇷🇺 MWSVisionBench

<div align="center">

**A Russian-language document benchmark for multimodal large language models**

*Make informed decisions when choosing multimodal models for production: evaluate on real-world business documents*

[🤗 OCR Dataset (Validation)](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench) • [🧪 Anti-fraud Dataset (Validation)](https://huggingface.co/datasets/MTSAIR/MWS-Antifraud-Bench) • [🏆 Leaderboard](#-leaderboard) • [📰 Habr Article (RU)](https://habr.com/ru/companies/mts_ai/articles/953292/)

</div>

---

## Update — July 23, 2026

### New: experimental anti-fraud category

We added a separate anti-fraud score for detecting original, manually edited,
and AI-generated document images. It rewards both correct classification and,
for manually edited documents, a correct explanation of what was changed.

Anti-fraud is reported as a standalone research metric and is **not** included
in `Overall`, which remains the mean of the five original benchmark categories.

[Read more about the anti-fraud experiment in our Habr article (Russian).](https://habr.com/ru/articles/1062176/)

---

## Update — February 16, 2026

### VQA category update

We clarified the wording and accepted-answer formats for part of the
**Reasoning VQA (ru)** category; the images remained unchanged. VQA and
`Overall` results are not directly comparable with versions from before
February 16, 2026.

This update improves reliability of reasoning-based evaluation while keeping the benchmark structure intact.

---

## 🎯 Overview

**MWSVisionBench** evaluates multimodal large language models on OCR and
document-understanding tasks in Russian. Its core dataset covers business,
technical, and handwritten document types. The experimental anti-fraud
dataset separately tests classification and explanation of original,
manually edited, and AI-generated document images.

### 🔥 Why MWSVisionBench?

Document-processing systems need to handle contracts, tables, diagrams, and
handwritten notes. The core benchmark covers these capabilities using:

- 📄 **Business-style documents** - contracts, reports, invoices, diagrams
- 📊 **Complex layouts** - tables, charts, mixed text-graphics content
- ✍️ **Handwritten content** - including musical notation and forms
- 🏗️ **Structured extraction** - JSON, Markdown, coordinate-based tasks
- 🎯 **Document-oriented scenarios** - OCR, extraction, localization, and VQA

### 📸 Example Documents from the Benchmark

![Representative samples from MWSVisionBench](assets/preview.jpg)

*Representative samples from MWSVisionBench: contracts, reports, technical
diagrams, charts, floor plans, and handwritten notes including even musical
notation.*

---

## 🚀 Key Features

### 📚 **Russian Document Datasets**
- **Core OCR benchmark:** 2,580 question-answer pairs across 800 images
- **Experimental anti-fraud benchmark:** 430 document images across public
  validation and private test sets
- **Manually assembled and annotated content**
- **Business and technical documents**, including handwritten notes
- **Professional annotation** with human experts

### 🎨 **Core Task Types — Included in `Overall`**

These five category scores are averaged with equal weight to produce
`Overall`:

1. **📝 Text OCR** - Basic image-to-text conversion
2. **🏗️ Structured OCR** - Image-to-Markdown conversion (requiring layout understanding)
3. **📍 Text Localization** - Find and return bounding boxes for specific text
4. **🗂️ Key Information Extraction** - Extract structured data (JSON format)
5. **❓ Visual Question Answering** - Answer questions about document content

### 🧪 **Experimental Task — Reported Separately**

6. **🛡️ Document Anti-fraud** - Classify a document image as `original`,
   `edited`, or `ai_gen` and explain the decision. The AF v0.1 score is shown
   separately and is not included in `Overall`.

### 🔧 **Benchmark Runner**
- **Unified API support** - OpenAI, GigaChat, vLLM (OpenAI-compatible)
- **Single CLI** - model-specific adapters behind one benchmark command
- **Parallel evaluation** - fast processing with multiprocessing
- **Comprehensive metrics** - adapted from [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2) with Russian optimizations
- **API-first approach** - designed for reproducible evaluation through endpoints

## 📊 Leaderboard

> **Background and original benchmark analysis**: [📰 Habr Article (Russian)](https://habr.com/ru/companies/mts_ai/articles/953292/)

### 🔓 Validation Set (Public)

Top models evaluated on the publicly available [validation dataset](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench):

`Overall` is the mean of the five original categories. `Anti-fraud` is reported separately and is not included in `Overall`.

| Model | Overall | img→text | img→markdown | Grounding | KIE (JSON) | VQA | Anti-fraud |
|-------|---------|----------|--------------|-----------|------------|-----|------------|
| Claude Fable 5 | 0.799 | 0.828 | 0.737 | 0.635 | 0.843 | 0.950 | 0.521 |
| GPT-5.6 Sol | 0.790 | 0.818 | 0.716 | 0.621 | 0.867 | 0.926 | 0.467 |
| GPT-5.5 | 0.778 | 0.855 | 0.686 | 0.563 | 0.873 | 0.913 | 0.477 |
| Kimi K3 | 0.777 | 0.783 | 0.716 | 0.588 | 0.856 | 0.941 | 0.400 |
| GPT-5.6 Terra | 0.749 | 0.808 | 0.693 | 0.571 | 0.799 | 0.874 | 0.404 |
| Kimi K2.6 | 0.743 | 0.868 | 0.761 | 0.301 | 0.858 | 0.929 | 0.279 |
| Cotype Pro 3 | 0.736 | 0.861 | 0.747 | 0.384 | 0.832 | 0.858 | 0.233 |
| Claude Sonnet 5 | 0.731 | 0.842 | 0.729 | 0.370 | 0.788 | 0.924 | 0.391 |
| GPT-5.6 Luna | 0.727 | 0.776 | 0.673 | 0.547 | 0.772 | 0.864 | 0.302 |
| Gemini 3.6 Flash | 0.723 | 0.833 | 0.736 | 0.178 | 0.903 | 0.967 | 0.115 |
| Qwen 3.7 Plus | 0.708 | 0.843 | 0.766 | 0.154 | 0.844 | 0.935 | 0.250 |
| Claude-4.6-Opus | 0.704 | 0.841 | 0.748 | 0.168 | 0.852 | 0.908 | 0.385 |
| Qwen3.6-35B-A3B | 0.694 | 0.859 | 0.784 | 0.099 | 0.822 | 0.904 | 0.232 |
| Qwen3.6-27B | 0.691 | 0.863 | 0.768 | 0.071 | 0.833 | 0.919 | 0.195 |
| Gemini-2.5-pro | 0.690 | 0.840 | 0.717 | 0.070 | 0.888 | 0.935 | 0.088 |
| Kimi K2 Instruct | 0.686 | 0.874 | 0.726 | 0.148 | 0.805 | 0.876 | 0.331 |
| Cotype Light 3 | 0.685 | 0.855 | 0.770 | 0.125 | 0.873 | 0.799 | 0.318 |
| GPT-5.4 | 0.683 | 0.764 | 0.627 | 0.351 | 0.826 | 0.846 | 0.432 |
| Claude Sonnet 4.6 | 0.682 | 0.827 | 0.725 | 0.186 | 0.755 | 0.915 | 0.235 |
| Gemini-3-flash-preview | 0.681 | 0.836 | 0.724 | 0.051 | 0.845 | 0.950 | 0.153 |
| Gemini-3.1-flash-lite-preview | 0.674 | 0.846 | 0.741 | 0.047 | 0.813 | 0.923 | 0.217 |
| Gemini 3.5 Flash | 0.674 | 0.773 | 0.727 | 0.052 | 0.869 | 0.948 | 0.216 |
| Gemini-2.5-flash | 0.672 | 0.886 | 0.729 | 0.042 | 0.825 | 0.879 | 0.137 |
| Claude-4.5-Opus | 0.670 | 0.809 | 0.720 | 0.131 | 0.799 | 0.889 | 0.308 |
| Qwen3.5-35B-A3B | 0.670 | 0.806 | 0.711 | 0.072 | 0.853 | 0.910 | 0.142 |
| Claude-4.5-Sonnet | 0.669 | 0.741 | 0.660 | 0.459 | 0.727 | 0.759 | 0.280 |
| GPT-5.2 | 0.663 | 0.799 | 0.656 | 0.173 | 0.855 | 0.835 | 0.413 |
| Alice AI VLM dev | 0.662 | 0.881 | 0.777 | 0.063 | 0.747 | 0.841 | 0.046 |
| GPT-4.1-mini | 0.659 | 0.863 | 0.735 | 0.093 | 0.750 | 0.853 | 0.173 |
| Qwen3.5-27B | 0.658 | 0.773 | 0.699 | 0.078 | 0.856 | 0.882 | 0.190 |
| Cotype VL (32B 8 bit) | 0.649 | 0.802 | 0.754 | 0.267 | 0.683 | 0.737 | 0.078 |
| GPT-5-mini | 0.639 | 0.782 | 0.678 | 0.117 | 0.774 | 0.843 | 0.233 |
| Qwen3.5-9B | 0.625 | 0.766 | 0.649 | 0.075 | 0.782 | 0.852 | 0.178 |
| Gemma 4 31B IT | 0.624 | 0.734 | 0.732 | 0.054 | 0.769 | 0.829 | 0.119 |
| Qwen3-VL-235B-A22B-Instruct | 0.623 | 0.812 | 0.668 | 0.050 | 0.755 | 0.830 | 0.171 |
| Qwen2.5-VL-72B-Instruct | 0.621 | 0.847 | 0.706 | 0.173 | 0.615 | 0.765 | 0.091 |
| GLM-5V Turbo | 0.601 | 0.682 | 0.759 | 0.044 | 0.726 | 0.793 | 0.183 |
| Qwen3.5-4B | 0.599 | 0.733 | 0.616 | 0.061 | 0.776 | 0.809 | 0.100 |
| GPT-5.1 | 0.588 | 0.716 | 0.680 | 0.092 | 0.670 | 0.783 | 0.096 |
| Qwen3-VL-8B-Instruct | 0.584 | 0.780 | 0.700 | 0.084 | 0.592 | 0.766 | 0.196 |
| Qwen3-VL-32B-Instruct | 0.582 | 0.730 | 0.631 | 0.056 | 0.708 | 0.784 | 0.298 |
| GPT-4.1 | 0.574 | 0.692 | 0.681 | 0.093 | 0.624 | 0.779 | 0.131 |
| Mistral Large 3 2512 | 0.565 | 0.777 | 0.713 | 0.065 | 0.560 | 0.709 | 0.215 |
| Mistral Small 3.2 24B Instruct | 0.561 | 0.734 | 0.695 | 0.060 | 0.599 | 0.715 | 0.122 |
| Qwen3.5-2B | 0.489 | 0.743 | 0.621 | 0.041 | 0.466 | 0.574 | 0.277 |
| Qwen3.5-0.8B | 0.319 | 0.549 | 0.499 | 0.027 | 0.157 | 0.362 | 0.185 |

### 🔒 Test Set (Private)

Results on our held-out private test dataset:

`Overall` is the mean of the five original categories. `Anti-fraud` is reported separately and is not included in `Overall`.

| Model | Overall | img→text | img→markdown | Grounding | KIE (JSON) | VQA | Anti-fraud |
|-------|---------|----------|--------------|-----------|------------|-----|------------|
| GPT-5.6 Sol | 0.789 | 0.836 | 0.692 | 0.623 | 0.862 | 0.930 | 0.418 |
| GPT-5.6 Terra | 0.763 | 0.817 | 0.680 | 0.579 | 0.838 | 0.901 | 0.299 |
| GPT-5.6 Luna | 0.740 | 0.794 | 0.672 | 0.549 | 0.807 | 0.879 | 0.243 |
| Claude Sonnet 5 | 0.738 | 0.842 | 0.708 | 0.374 | 0.830 | 0.932 | 0.335 |
| Cotype Pro 3 | 0.726 | 0.837 | 0.729 | 0.393 | 0.845 | 0.828 | 0.157 |
| Qwen 3.7 Plus | 0.706 | 0.828 | 0.753 | 0.143 | 0.867 | 0.938 | 0.254 |
| Claude-4.6-Opus | 0.699 | 0.833 | 0.715 | 0.175 | 0.832 | 0.940 | 0.308 |
| Qwen3.6-27B | 0.692 | 0.861 | 0.753 | 0.062 | 0.893 | 0.889 | 0.118 |
| GPT-5.4 | 0.691 | 0.780 | 0.632 | 0.341 | 0.851 | 0.853 | 0.409 |
| Qwen3.6-35B-A3B | 0.691 | 0.847 | 0.755 | 0.094 | 0.840 | 0.920 | 0.181 |
| Gemini-3-flash-preview | 0.678 | 0.816 | 0.712 | 0.054 | 0.875 | 0.931 | 0.125 |
| Gemini-3.1-flash-lite-preview | 0.678 | 0.859 | 0.724 | 0.050 | 0.864 | 0.891 | 0.221 |
| Cotype Light 3 | 0.678 | 0.854 | 0.736 | 0.132 | 0.861 | 0.808 | 0.258 |
| Claude-4.5-Opus | 0.676 | 0.812 | 0.698 | 0.145 | 0.812 | 0.915 | 0.292 |
| Claude-4.5-Sonnet | 0.674 | 0.754 | 0.660 | 0.440 | 0.750 | 0.766 | 0.300 |
| Gemini-2.5-pro | 0.674 | 0.818 | 0.719 | 0.068 | 0.836 | 0.929 | 0.067 |
| Gemini 3.5 Flash | 0.672 | 0.789 | 0.705 | 0.050 | 0.882 | 0.934 | 0.180 |
| Claude Sonnet 4.6 | 0.671 | 0.829 | 0.691 | 0.186 | 0.756 | 0.896 | 0.350 |
| Qwen3.5-35B-A3B | 0.664 | 0.808 | 0.682 | 0.073 | 0.847 | 0.908 | 0.129 |
| Kimi K2 Instruct | 0.664 | 0.858 | 0.706 | 0.131 | 0.779 | 0.846 | 0.256 |
| Alice AI VLM dev | 0.654 | 0.891 | 0.751 | 0.066 | 0.751 | 0.809 | 0.000 |
| Gemini-2.5-flash | 0.654 | 0.869 | 0.675 | 0.047 | 0.814 | 0.866 | 0.143 |
| GPT-4.1-mini | 0.653 | 0.869 | 0.713 | 0.095 | 0.735 | 0.851 | 0.125 |
| Qwen3.5-27B | 0.648 | 0.737 | 0.685 | 0.073 | 0.833 | 0.913 | 0.069 |
| GPT-5.2 | 0.647 | 0.806 | 0.643 | 0.156 | 0.794 | 0.835 | 0.328 |
| Cotype VL (32B 8 bit) | 0.637 | 0.803 | 0.746 | 0.251 | 0.687 | 0.701 | 0.067 |
| Qwen2.5-VL-72B-Instruct | 0.630 | 0.844 | 0.701 | 0.193 | 0.645 | 0.770 | 0.024 |
| GPT-5-mini | 0.625 | 0.772 | 0.654 | 0.105 | 0.717 | 0.875 | 0.254 |
| Qwen3.5-9B | 0.624 | 0.772 | 0.638 | 0.072 | 0.789 | 0.850 | 0.159 |
| Gemma 4 31B IT | 0.613 | 0.734 | 0.700 | 0.043 | 0.793 | 0.792 | 0.162 |
| Qwen3-VL-235B-A22B-Instruct | 0.612 | 0.816 | 0.648 | 0.053 | 0.739 | 0.802 | 0.083 |
| GLM-5V Turbo | 0.609 | 0.657 | 0.753 | 0.051 | 0.770 | 0.813 | 0.199 |
| Qwen3.5-4B | 0.602 | 0.705 | 0.639 | 0.058 | 0.785 | 0.824 | 0.062 |
| GPT-5.1 | 0.582 | 0.713 | 0.688 | 0.087 | 0.650 | 0.770 | 0.062 |
| Qwen3-VL-8B-Instruct | 0.578 | 0.779 | 0.692 | 0.073 | 0.592 | 0.754 | 0.164 |
| Mistral Small 3.2 24B Instruct | 0.577 | 0.748 | 0.701 | 0.067 | 0.664 | 0.703 | 0.133 |
| Qwen3-VL-32B-Instruct | 0.576 | 0.740 | 0.630 | 0.050 | 0.671 | 0.786 | 0.254 |
| GPT-4.1 | 0.574 | 0.698 | 0.676 | 0.081 | 0.664 | 0.753 | 0.123 |
| Mistral Large 3 2512 | 0.551 | 0.753 | 0.691 | 0.063 | 0.542 | 0.704 | 0.185 |
| Qwen3.5-2B | 0.473 | 0.708 | 0.622 | 0.038 | 0.461 | 0.538 | 0.200 |
| Qwen3.5-0.8B | 0.317 | 0.538 | 0.493 | 0.023 | 0.215 | 0.318 | 0.177 |

*Scale: 0.0 - 1.0 (higher is better)*

**📝 Submit your model**: To evaluate on the private test set, contact [cotype@mts.ai](mailto:cotype@mts.ai)

> 🔧 **Note**: Results may vary slightly (±0.001-0.002) due to API sampling. We recommend running benchmarks through API endpoints for consistency.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mts-ai/MWS-Vision-Bench.git
cd MWS-Vision-Bench

# Create conda environment (recommended)
conda create -n mwsvision python=3.10
conda activate mwsvision

# Install dependencies
pip install -r requirements.txt
```

### Running the Benchmark

**The dataset automatically downloads from Hugging Face!** No manual download needed.

```bash
# Simplest way - dataset downloads automatically
python run_benchmark.py \
    --model_name "gpt-4o-mini" \
    --api_key "your-openai-key"

# Quick test with 100 samples
python run_benchmark.py \
    --model_name "gpt-4o-mini" \
    --api_key "your-openai-key" \
    --sample 100

# GigaChat Max
python run_benchmark.py \
    --model_name "gigachat-max" \
    --api_key "your-gigachat-key"

# Qwen2.5-VL (via vLLM endpoint)
python run_benchmark.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --api_url "http://your-vllm-server/v1/chat/completions"

# Adjust parallelism with --max_workers (default: 5 for OpenAI-compatible APIs, 1 for GigaChat)
python run_benchmark.py \
    --model_name "gpt-4o-mini" \
    --api_key "your-openai-key" \
    --max_workers 20

# Experimental anti-fraud validation set
python run_benchmark.py \
    --model_name "gpt-4o-mini" \
    --api_key "your-openai-key" \
    --dataset_family antifraud

# Reproducible validation + private test run with independent Hub revisions
python run_benchmark.py \
    --model_name "gpt-4o-mini" \
    --api_key "your-openai-key" \
    --dataset_family antifraud \
    --hf_token "$HF_TOKEN" \
    --hf_revision "<validation-commit-sha>" \
    --hf_test_revision "<test-commit-sha>"
```

**⚡ Performance Tips:**
- **OpenAI-compatible default**: 5 parallel workers
- **Higher-throughput endpoints**: Increase `--max_workers` cautiously and
  stay within the provider's concurrency and rate limits
- **GigaChat**: Use `--max_workers 1` (API has strict rate limits)
- **Local models**: Adjust based on GPU memory (typically 1-5)

### Evaluation Results

Results are automatically saved to:
- `results/{test_name}/{model_name}_validation_eval.json` - validation metrics
- `results/{test_name}/{model_name}_test_eval.json` - test metrics (if accessible)
- `logs/{test_name}.log` - execution logs

### Experimental anti-fraud category

The anti-fraud dataset contains three labels:

- `original` — an image treated as an unmodified source in this experiment;
- `edited` — a document manually modified in a graphics editor;
- `ai_gen` — a document image created by a generative model.

The model returns a JSON object:

```json
{"label": "original|edited|ai_gen", "arguments": "short explanation"}
```

The metric combines balanced three-class accuracy with the quality of the
explanation for manually edited documents:

```text
AF = 0.75 × max(0, balanced_accuracy − 1/3)
     + 0.5 × edited_reason_score
```

Unlike the five original metrics, AF is aggregated over a complete split:
balanced accuracy requires the full three-class confusion matrix. It is not a
plain mean of independent per-item scores.

Anti-fraud is an experimental `AF v0.1` category. It is displayed separately
and is excluded from the main Overall score, preserving comparability with
earlier benchmark results. It is released for research and model comparison,
not as a universal or production-grade anti-fraud score. The experimental
dataset can be optimized against, which is another reason to keep AF separate.

This category measures general-purpose multimodal language models and is not a
replacement for a dedicated production fraud-detection system.


---

## 📄 Article

- 📰 **[Habr Article (Russian)](https://habr.com/ru/companies/mts_ai/articles/953292/)** - original benchmark analysis and results

---

## 🙏 Acknowledgements

This benchmark was inspired by and adapted from [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2) by Yuliang Liu et al. We thank the authors for their pioneering work in multimodal OCR evaluation and for making their codebase publicly available.

---

## 🤝 Contributing & Feedback

We welcome contributions from the community! Whether you've found a bug, have a feature request, or want to improve documentation, your input is valuable.

### How to Contribute
- **Report Issues**: Found a bug or inconsistency? [Open an issue](https://github.com/mts-ai/MWS-Vision-Bench/issues)
- **Feature Requests**: Have ideas for improvement? We'd love to hear them
- **Pull Requests**: Code improvements and documentation updates are welcome
- **Discussions**: Questions or want to share your results? Start a [discussion](https://github.com/mts-ai/MWS-Vision-Bench/discussions)

Your input helps make MWSVisionBench better for the entire community!

---

## 📁 Dataset

### 🤗 Hugging Face Datasets (Recommended)

The datasets are hosted on the Hugging Face Hub and **automatically
downloaded** when you run the benchmark:

- **🔓 Validation Set (Public)**: [`MTSAIR/MWS-Vision-Bench`](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench)
  - 1,302 questions on 400 unique images
  - Publicly available for model development and evaluation
  
- **🔒 Test Set (Private)**: 
  - 1,278 questions on 400 different images
  - No public access

- **🧪 Anti-fraud Validation**:
  [`MTSAIR/MWS-Antifraud-Bench`](https://huggingface.co/datasets/MTSAIR/MWS-Antifraud-Bench)
  - 209 examples: `original`, `edited`, and `ai_gen`
  - Experimental AF v0.1 research dataset
  - Distributed under its own
    [dataset license](https://huggingface.co/datasets/MTSAIR/MWS-Antifraud-Bench/blob/main/LICENSE.md),
    separately from this repository's MIT-licensed code

- **🔒 Anti-fraud Test (Private)**:
  [`MTSAIR/MWS-Antifraud-Bench-Test`](https://huggingface.co/datasets/MTSAIR/MWS-Antifraud-Bench-Test)
  - 221 held-out examples
  - Requires an authorized `HF_TOKEN`

### Quick Access

```python
from datasets import load_dataset

# Load validation dataset (public)
dataset = load_dataset("MTSAIR/MWS-Vision-Bench")

# Load experimental anti-fraud validation (public; no token required)
antifraud = load_dataset("MTSAIR/MWS-Antifraud-Bench", split="train")
```

### Running the regression tests

```bash
python -m unittest discover -s tests -v
```

### Data Format

Each entry contains:
```json
{
  "id": "1",
  "type": "text grounding ru",
  "dataset_name": "business",
  "image_path": "/business/scans/b2b_scans_1.jpg",
  "question": "Где находится герб на документе? Выведи абсолютные координаты...",
  "answers": [398, 65, 467, 140]
}
```
