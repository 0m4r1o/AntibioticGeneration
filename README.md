<div id="top" align="left">

<img src="AntibioticGeneration.png" width="30%" alt="Project Logo"/>

# 🧫 ANTIBIOTICGENERATION  
*Revolutionizing Antibiotics Through Intelligent Molecular Innovation*

---

### 🧩 Built With

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat&logo=Keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/Rich-FAE742.svg?style=flat&logo=Rich&logoColor=black" alt="Rich">
<br>
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">

<br>

---

## 📚 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Return to Top](#top)

---

## 🧠 Overview

**AntibioticGeneration** is a comprehensive developer framework designed to accelerate antibiotic discovery through computational chemistry and deep learning.  
It combines molecular filtering, similarity scoring, and neural generation models to efficiently identify and refine promising antibiotic candidates.

### 💡 Why AntibioticGeneration?

This project unites **cheminformatics** and **machine learning** to optimize antibiotic design and evaluation.  
Its key features include:

- 🧬 **Molecular Filtering:** Enforces Lipinski’s rules, removes invalid or duplicate SMILES, and selects viable drug-like compounds.  
- 🧪 **Similarity Assessment:** Incorporates scaffold-based and fingerprint similarity metrics for better chemical comparison.  
- 🤖 **Deep Learning Models:** Trains LSTM architectures to generate and evolve novel molecular structures.  
- ⚙️ **GPU Verification:** Validates CUDA setup and GPU readiness for model training and inference.  
- 🔁 **Iterative Retraining:** Automatically refines the model using top-performing molecules for continuous improvement.

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

---

<div align="left">
  <a href="#top">⬆ Back to Top</a>
</div>

---
