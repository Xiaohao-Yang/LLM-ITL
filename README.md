# 🧠 LLM-ITL: Neural Topic Modeling with Large Language Models in the Loop

![ACL 2025](https://img.shields.io/badge/ACL%202025-Main%20Conference-blueviolet)

This repository contains the official implementation of our paper:

> 📄 **Neural Topic Modeling with Large Language Models in the Loop**  
> 🏆 **Accepted to the Main Conference of [ACL 2025](https://2025.aclweb.org/)**  
> 🔗 [Read the paper](https://arxiv.org/abs/2411.08534) 

---

## 📂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Citation](#-citation)
- [External Files](#-external-files)
- [License](#-license)

---

## 📖 Overview
<p align="center">
  <img src="overview.png" alt="LLM-ITL Framework Diagram" width="700"/>
</p>

<p align="center">
  <em>Figure: Overview of the LLM-ITL framework.</em>
</p>

**LLM-ITL** is a novel **LLM-in-the-loop framework** that creates a synergistic integration between Large Language Models (LLMs) and Neural Topic Models (NTMs), enhancing topic interpretability while preserving efficient topic and document representation learning.

While LLMs have shown potential in topic discovery, directly applying them to topic modeling introduces challenges such as incomplete topic coverage, misalignment, and inefficiency. **LLM-ITL** addresses these limitations by combining the efficiency of NTMs with the interpretive strength of LLMs.

In this framework:
- NTMs are used to learn global topics and document representations.
- An LLM refines the learned topics via an **Optimal Transport (OT)-based alignment objective**, dynamically adjusting based on the LLM's confidence in suggesting topical words.

LLM-ITL is modular and compatible with many existing NTMs. It significantly improves topic interpretability without compromising the quality of learned document representations — as demonstrated through extensive experiments.

<p align="center">
  <img src="assets/llm_itl_framework.png" alt="LLM-ITL Framework Overview" width="600"/>
</p>

<p align="center">
  <em>Figure: Overview of the LLM-ITL framework. The NTM learns initial topics and document representations, while the LLM refines topics via OT-based alignment.</em>
</p>

---

## ✨ Features

- 🔄 **Flexible integration**: Seamlessly combine a wide range of Large Language Models (LLMs) with various Neural Topic Models (NTMs) through a unified interface.
- 🧠 **Supports multiple LLMs**: Compatible with:
  - [GPT](https://platform.openai.com/docs/models)
  - [DeepSeek](https://huggingface.co/deepseek-ai)
  - [Mistral](https://huggingface.co/mistralai)
- 📊 **Supports many NTMs**: Easily integrates with models like **ETM**, **NMF**, **ProdLDA**, and others through a flexible, extensible design.
- ⚙️ **Modular design**: Swap LLMs, NTMs, or evaluation modules independently — ideal for research experiments and benchmarking.
- 📈 **Built-in evaluation metrics**: Includes support for traditional metrics such as **topic coherence**, **topic alignment**, and **topic diversity**, as well as newly proposed metrics like [**WALM**](https://your-link-here.com), for comprehensive evaluation of topic models.
- 🗂️ **Dataset flexibility**: Compatible with widely used datasets such as **20Newsgroups**, **AG News**, and **Amazon Reviews** — and easy to extend to custom datasets.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/Xiaohao-Yang/LLM-ITL.git
cd LLM-ITL

# Create a Python 3.9 virtual environment named 'LLM-ITL'
python3.9 -m venv LLM-ITL
source LLM-ITL/bin/activate       # On Windows: LLM-ITL\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt


# Run a Topic Model (Base Model Only)
python main.py --dataset 20News --model nvdm --n_topic 50 --random_seed 1

# Run a Topic Model with LLM-ITL
python main.py --dataset 20News --model nvdm --n_topic 50 --random_seed 1 --llm_itl

# Run Evaluation
python eval.py --dataset 20News --model nvdm --n_topic 50 --eval_topics

# Output files:
# - Model checkpoints: LLM-ITL/save_models/
# - Learned topics:    LLM-ITL/save_topics/
# - Evaluation output: LLM-ITL/evaluation_output/

# Example evaluation output:
# Topic Coherence (NPMI):       0.264
# Topic Diversity:              0.872
# Topic Alignment (OT Score):   0.301
# WALM Score:                   0.447

