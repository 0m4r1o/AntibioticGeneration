<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="left">

<img src="AntibioticGeneration.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

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

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build AntibioticGeneration from the source and install dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/0m4r1o/AntibioticGeneration
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd AntibioticGeneration
    ```

3. **Install the dependencies:**

**Using [pip](https://pypi.org/project/pip/):**

```sh
❯ pip install -r requirements.txt
```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**

In order to Start the pipeline make sure to have the data **all_smiles_clean.smi** and begin with :

```sh
python LSTMSMILES.py
```
with 
```python
Train = True
```
This will choose the best LSTM arch model after training each one of them for **20** Epochs. After choosing the best arch the model will train for **100** epochs.
After finishing the script will generate a folder **\Results\Models\smiles_lstm_bn_ckpt_{GENERATION}\**. After that you must run the same program but with 
```python
Train = False
```
In order to generate a sample of molecules (set to 1200 currently). The script will save the results here **\Results\Generation\SMILES_gen_0.smi** (for generation 0 for example). After that you can run the script FilterSmiles.py to filter the 1200 SMILES (remove invalids, duplicates and Lipinski & SAS violations and calculates the Simulation Score for each SMILES.
```sh
python FilterSmiles.py
```
The script will save an excel file in **Results/Curated/scores_output_gen_{generation}.xlsx**. Then run the Script **RetrainModel.py** which will take the best 100 and retrain the model through transfer learning and will output a new model **\Results\Models\smiles_lstm_bn_ckpt_{NEXTGENERATION}\**.
```sh
python RetrainModel.py
```


### Testing
To test the GPU make sure to have the requirements installed, and run the following : 
```sh
python testgpu.py
```
---

<div align="left"><a href="#top">⬆ Return</a></div>

---
