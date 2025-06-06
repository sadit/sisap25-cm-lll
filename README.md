# SISAP 2025 Index Challenge - DiskANN-based Solution

## Project Introduction

This project is a customized solution for the [SISAP 2025 Index Challenge](https://sisap-challenges.github.io/2025/) based on [DiskANN](https://github.com/microsoft/DiskANN) and related Python tools. It supports large-scale vector data indexing and evaluation, and provides both **manual source deployment** and **Docker containerization** modes.

---

## Method 1: Manual Source Deployment

**1. Install C++ build dependencies**

Install the following packages through apt-get

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

Install Intel MKL (Ubuntu 20.04 or newer)

```bash
sudo apt install libmkl-full-dev
```

**2. Clone the project and install Python dependencies**

```bash
git clone https://github.com/cm-lll/SISAP2025-Solution.git
cd SISAP2025-Solution
pip install -r requirements.txt
```

**3. Build the C++ code**

```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && cd ..
```

**4. Prepare the data**

Place your data file (e.g., `benchmark-dev-{dataname}.h5`, the file name must be in the form `benchmark-dev-{dataname}.h5`) into the `data/{dataname}/task1/` folder under the project root:

```
./data/pubmed23/task1/benchmark-dev-pubmed23.h5
```
If the data does not exist, the project can download the dataset via the network. You can configure this in `RunAndEval.py`.

**5. Run the script**

```bash
python3 RunAndEval.py --dataname pubmed23 --results ./results.csv
```
Other input parameters can be found in the **Parameter Description** section.

## Method 2: Docker Containerized Operation

**1. Build the image**

In the project root directory, execute:

```bash
docker build -t diskann-image .
```

**2. Prepare and mount the data**

Place your data file (e.g., `benchmark-dev-pubmed23.h5`, the file name must be in the form `benchmark-dev-{dataname}.h5`) in a directory on the host (e.g., `/home/your_user/diskann_host_data`).

**3. Run the container and mount the data directory**

```bash
docker run -it --rm \
    -v /home/your_user/diskann_host_data:/app/DiskANN/data{dataname}/task1 \
    diskann-image \
    python3 RunAndEval.py --dataname pubmed23 --results ./results.csv
```

---

## Parameter Description

| Parameter    | Description                                         | Default Value                   |
|--------------|-----------------------------------------------------|---------------------------------|
| --results    | Path to save the result CSV file                    | ./results.csv                   |
| --dataname   | Dataset name (e.g., pubmed23, ccnews, etc.)         | pubmed23                        |
| --R          | DiskANN build parameter max_degree                  | 64                              |
| --LB         | Build search list size L_build                      | 100                             |
| --B          | search_DRAM_budget                                  | 3                               |
| --M          | build_DRAM_budget                                   | 4                               |
| --T          | Number of threads                                   | 8                               |
| -LS          | List of L parameters for search (multiple allowed)  | 30 40 50 100 150 200            |
| --K          | Number of nearest neighbors to search               | 30                              |
| --RD         | The target dimension of dimensionality reduction    | 192                             |
---

After verification, the default values we set can make the results of the PubMed23 dataset meet the requirements of the Sisap2025 Challenge Task 1.

## Output

- Main results are saved to the path you specify with `--results` (e.g., `./results.csv`).
- Resource monitoring logs: `sys_usage.log`, `file_changes.log`. You can obtain the peak memory usage by running `temp2.py`.
- Generated indexes, bin files, etc., will be output to the `build` and `search_results` subdirectories under the `data` directory.
- `OnlyEvalRecall.py` can be used to evaluate recall on existing search results.

---

If you have any questions, feel free to open an issue or start a discussion!

