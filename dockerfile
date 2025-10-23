# -----------------------------------------------------------------------------
# GPTMOL / AntibioticGeneration — Docker image
# - Python 3.9 + RDKit (conda-forge) + TensorFlow 2.10.x (CPU)
# - Installs the rest of requirements via pip (with RDKit/TF filtered out)
# -----------------------------------------------------------------------------
FROM mambaorg/micromamba:1.4.2

# Avoid interactive tz prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Create working dir
WORKDIR /app

# Copy project metadata early (for cache)
COPY requirements.txt /app/requirements.txt

# Create a conda env and install RDKit + core deps
# Use bash -lc so micromamba activation works during build steps
SHELL ["/bin/bash", "-lc"]

# Create env 'gptmol' with Python + RDKit and common libs from conda-forge
RUN micromamba create -y -n gptmol -c conda-forge \
    python=3.9 \
    rdkit=2022.09.5 \
    numpy pandas scikit-learn matplotlib tqdm openpyxl \
 && micromamba clean -a -y

# Make the env default
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=gptmol
ENV PATH=/opt/conda/envs/gptmol/bin:$PATH

# Install TensorFlow CPU explicitly from pip (2.10.x is broadly compatible)
# Then install the rest of your requirements (filtering out rdkit/tensorflow entries)
RUN python -m pip install --upgrade pip \
 && python - <<'PY'\nimport re, sys\ninfile='requirements.txt'\noutfile='requirements.nordkit.notf.txt'\nblock = re.compile(r'^(rdkit|rdkit-pypi|tensorflow(-intel)?|tensorflow-estimator)\\b', re.I)\nwith open(infile, 'r', encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as g:\n    for line in f:\n        if not block.search(line.strip()):\n            g.write(line)\nprint('Wrote', outfile)\nPY\n && python -m pip install --no-cache-dir 'tensorflow==2.10.1' \
 && python -m pip install --no-cache-dir -r requirements.nordkit.notf.txt

# Copy the rest of the repo
# (You can narrow this to only the needed files if you prefer)
COPY . /app

# Default env toggles like in LSTMSMILES.py (can be overridden in compose)
ENV TF_CPP_MIN_LOG_LEVEL=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Create output folders so they exist inside the image
RUN mkdir -p /app/data /app/figures /app/Results

# Default command: open a shell; docker-compose will override with specific runs
CMD ["/bin/bash"]
