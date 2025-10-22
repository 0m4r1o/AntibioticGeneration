Can you change the instructions that are below to be better readable
<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="left">

<img src="antibiotics.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# ANTIBIOTICGENERATION

<em>Revolutionizing Antibiotics Through Intelligent Molecular Innovation</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/0m4r1o/AntibioticGeneration?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/0m4r1o/AntibioticGeneration?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/0m4r1o/AntibioticGeneration?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat&logo=Keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/Rich-FAE742.svg?style=flat&logo=Rich&logoColor=black" alt="Rich">
<br>
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">

</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)

---

## Overview

AntibioticGeneration is a comprehensive developer toolset aimed at advancing antibiotic discovery through computational methods. It integrates molecular filtering, similarity assessment, and deep learning-based molecule generation to streamline the identification of promising drug candidates.

**Why AntibioticGeneration?**

This project leverages cheminformatics and machine learning to facilitate the design, analysis, and optimization of antibiotics. The core features include:

- 🧬 **🔍 Molecular Filtering:** Filters molecules based on Lipinski's rules, validity, and uniqueness, ensuring high-quality candidates.
- 🧪 **⚙️ Similarity Assessment:** Provides scaffold-aware and fingerprint-based similarity metrics for nuanced compound comparison.
- 🧠 **🧬 Deep Learning Models:** Uses LSTM architectures to generate and optimize novel molecular structures.
- 💻 **🚀 GPU Verification:** Checks hardware readiness to maximize performance during model training and inference.
- 🔄 **🔧 Iterative Retraining:** Automates model refinement by selecting top molecules and retraining for continuous improvement.

---

## 🚀 Getting Started

### 🧩 Prerequisites

Ensure the following are installed:

- **Python 3.9+**
- **pip** or **conda**

---

### ⚙️ Installation

Clone and set up the project:

```bash
git clone https://github.com/0m4r1o/AntibioticGeneration
cd AntibioticGeneration
pip install -r requirements.txt
```

---

### ▶️ Usage

#### Step 1 — Train the Base Model
Ensure your dataset `all_smiles_clean.smi` is available.  
Open **LSTMSMILES.py** and set:

```python
Train = True
```

Then run:
```bash
python LSTMSMILES.py
```

This will:
- Test multiple LSTM architectures for **20 epochs** each.
- Train the best-performing model for **100 epochs**.
- Save the trained model in  
  `Results/Models/smiles_lstm_bn_ckpt_{GENERATION}/`

---

#### Step 2 — Generate Molecules
After training completes, switch:

```python
Train = False
```

Run:
```bash
python LSTMSMILES.py
```

This generates ~1200 molecules and saves them in:
```
Results/Generation/SMILES_gen_0.smi
```

---

#### Step 3 — Filter Molecules
Run:
```bash
python FilterSmiles.py
```

This filters invalid or duplicate SMILES, applies Lipinski and SAS checks,  
and saves the curated results to:
```
Results/Curated/scores_output_gen_{generation}.xlsx
```

---

#### Step 4 — Retrain with Top Molecules
Finally, retrain the model on the best 100 molecules:

```bash
python RetrainModel.py
```

The new fine-tuned model will be saved in:
```
Results/Models/smiles_lstm_bn_ckpt_{NEXTGENERATION}/
```

---

### 🧪 Testing

To verify GPU setup:

```bash
python testgpu.py
```
If successful, your CUDA-compatible GPU will be recognized by TensorFlow.


### Results

below are the results for training the Base Model : 

<img src="Results\figures\loss.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Loss over epochs"/>
<img src="Results\figures\accuracy.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Accuracy over epochs"/>

And these are the generation metrics : 
```json
{
  "Validity":97.40,
  "Uniqueness":99.17
}

```

---

<div align="left">
  <a href="#top">⬆ Back to Top</a>
</div>

---
