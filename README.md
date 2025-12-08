# TriH-ANNS: High-Dimensional Similarity Search with PCA and Heterogeneous (CPU+GPU) Acceleration

TriH-ANNS is an efficient high-dimensional nearest-neighbor search implementation that combines PCA dimensionality reduction with heterogeneous hardware acceleration (CPU + GPU). This repository provides a Linux + CUDA implementation, evaluation scripts, and theoretical validation materials.

---

## 1. Installation & Environment Setup

### 1.1 Dependencies & Versions
Please ensure the following tools and compilers are installed:

- `cmake`, version ≥ 3.29  
- `ninja`  
- `g++`, version ≥ 11.4  
- `nvcc` (CUDA toolkit)

Note: This implementation targets Linux systems with CUDA-capable GPUs. Building and running on a Linux machine with CUDA support is recommended.

### 1.2 Clone the Repository
```bash
git clone https://github.com/PneNPproof/TriH-ANNS.git
cd trih-anns
```

---

## 2. Datasets (HDF5 Format)

### 2.1 Required Data Files
Place your datasets under the `dist/` directory. Example filename for a GIST dataset:
- `dist/gist-960-euclidean-shuffle.hdf5`

Each HDF5 file must include the following datasets:
- `train`: training vectors  
- `test`: test vectors (query vectors)  
- `neighbors`: ground-truth neighbors

### 2.2 Data Sources & Where to Get Them
Common datasets can be obtained from these sources:
- GIST, SIFT: https://github.com/erikbern/ann-benchmarks  
- Cohere: https://vdbpublic.oss-cn-hangzhou.aliyuncs.com/cohere-768-euclidean.hdf5  
- OpenAI embeddings: assemble your own HDF5 (see VectorDBBench Issue #411 for guidance). Typically, `train` is shuffled data; you can reference msong, imagenet sources as needed.

### 2.3 Data Path Configuration
Edit `~/trih/src/main.cu` and update the following variables using absolute paths:
- `data_dir`: directory containing HDF5 data files (recommended to be shuffled)  
- `index_dir`: directory to store constructed index files (indexes can be large—use a dedicated location)

If your data are not shuffled, the repo includes a shuffle utility (`myshuffle2`) referenced later.

### 2.4 Optional Data Shuffling
`trih/src/main.cu` contains a commented code block that demonstrates using `myshuffle2` to shuffle HDF5 inputs. If your dataset is approximately random, you may skip this; if there is distribution bias, enabling shuffle often improves performance and robustness.

---

## 3. Build

From the project root run:

```bash
./cmake_configure_build.sh
```

This script configures the project, generates the build system, and compiles the code.

For custom builds, set `CMAKE_BUILD_TYPE` to `Release` for best performance.

---

## 4. Run & Test Examples

### 4.1 Default Settings
- The project by default returns Top-100 neighbors (controlled by `PHASE2_TOPK`).

### 4.2 GIST-Based Testing
The repository includes test scripts to evaluate the GIST dataset under different parameter settings:
- Query batch sizes (`QUERY_BATCH_SIZES`)  
- PCA output dimension (`DATA_TO_COLUMNS`)  
- Group size for reductions (`REDUCE_GROUP_SIZES`)  
- Phase-1 top-k (`PHASE1_TOPK`) (wk settings)  
- Whether to use exact re-ranking (`RERANK_METHOD`)

Test outputs include:
- QPS (queries per second)  
- Recall

Results from the test scripts are saved under the repository root in `anns_results/`.

### 4.3 Batch Test Execution
From the project root execute:

```bash
./test/batch_process.sh
```

Note: Modify script filenames and parameters as needed to match your dataset.

### 4.4 Manual Tuning
You can directly edit `trih/src/main.cu` to change runtime parameters, then recompile and run. The commands supplied in test scripts can serve as examples.

---

## 5. Repository Structure

- `trih/`: algorithm core and related source code (contains Eigen, GPU implementations, and test files)  
  - `trih/src/`  
    - `main.cu`: program entry and runtime parameters  
    - `gpu_pca_anns.cu`: core TriH algorithm implementation on GPU  
    - `dataset.h`: HDF5 reading and parsing utilities  
    - `rerank.cpp`, `sq.cpp`: CPU-side reranking and quantization implementations  
- `test/`: test scripts and documentation  
- `complete_proof/`, `empirical_validation_of_theoretical_results/`: theoretical proofs and experimental materials from the paper  
- `anns_results/`: stores output from test scripts

exactly description see `ONBOARDING.md`
