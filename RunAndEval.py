import os
import argparse
import csv
import struct
import h5py
import numpy as np
from subprocess import run, PIPE
from sklearn.decomposition import PCA
from joblib import dump
from joblib import load
import time
import psutil
from multiprocessing import Process, Event
import urllib.request

def download_h5_if_needed(h5_file, dataset_name):
    """
    download dataset if h5_file does not exist.
    """
    url = f"https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-{dataset_name}.h5?download=true"
    print(f"[INFO] {h5_file} does not exist, trying to download from {url} ...")
    os.makedirs(os.path.dirname(h5_file), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, h5_file)
        print(f"[INFO] Download finished: {h5_file}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        exit(1)

def get_recall(I, gt, k):
    """
    Calculate recall@k.
    """
    assert k <= I.shape[1]
    assert len(I) == len(gt)
    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)

def load_uint32_bin(bin_path):
    """
    Read DiskANN .bin file (uint32 type), return numpy array of shape (n, k)
    File format: int32 n, int32 k, n*k uint32
    """
    with open(bin_path, 'rb') as f:
        n, k = struct.unpack('<ii', f.read(8))
        arr = np.frombuffer(f.read(n * k * 4), dtype=np.uint32).reshape(n, k)
    return arr

# Resource monitoring
def get_dir_size_gb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 ** 3), 4)  # Return GB, keep 4 decimals

def monitor_resources(log_file, interval, stop_event, watch_dir=None):
    with open(log_file, "w") as f:
        f.write("time,mem_total_GB,mem_used_GB,mem_percent,disk_total_GB,disk_used_GB,disk_percent")
        if watch_dir:
            f.write(",dir_used_GB")
        f.write("\n")
        while not stop_event.is_set():
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"{now},{mem.total/2**30:.2f},{mem.used/2**30:.2f},{mem.percent},{disk.total/2**30:.2f},{disk.used/2**30:.2f},{disk.percent}"
            if watch_dir:
                dir_size = get_dir_size_gb(watch_dir)
                line += f",{dir_size}"
            f.write(line + "\n")
            f.flush()
            time.sleep(interval)

def monitor_file_changes(watch_dir, log_file, interval=2, stop_event=None):
    last_files = set(os.listdir(watch_dir))
    with open(log_file, "w") as f:
        f.write("time,added_files,removed_files\n")
        while stop_event is None or not stop_event.is_set():
            time.sleep(interval)
            current_files = set(os.listdir(watch_dir))
            added = current_files - last_files
            removed = last_files - current_files
            if added or removed:
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{now},{'|'.join(added)},{'|'.join(removed)}\n")
                f.flush()
            last_files = current_files
#--------------------------------------------------------------------------------

# Data reading and preprocessing
def export_full_data(h5_path, data_key, out_bin, align_bytes=64):
    """
    Export data to .bin format, no PCA, only alignment.
    """
    import struct
    import h5py
    with h5py.File(h5_path, 'r') as f:
        dataset = f[data_key]
        n, d = dataset.shape
        print(f"[INFO] Dataset shape: {n} x {d}")
        with open(out_bin, 'wb') as fout:
            fout.write(struct.pack('<I', n))
            fout.write(struct.pack('<I', d))
            row_bytes = d * 4
            pad = (align_bytes - (row_bytes % align_bytes)) % align_bytes
            for vec in dataset:
                fout.write(vec.astype(np.float32).tobytes())
                if pad:
                    fout.write(b'\x00' * pad)

def export_queries(h5_path, data_key, out_bin, align_bytes=64):
    """
    Export query data to .bin format, no PCA, only alignment.
    """
    export_full_data(h5_path, data_key, out_bin, align_bytes)

from sklearn.preprocessing import normalize

def export_full_data_with_pca(h5_path, data_key, out_bin, pca_model_path, align_bytes=64, max_sample_size=100000, pca_dim=192):
    """
    Export data to .bin format, with PCA and normalization, batch processing.
    """
    with h5py.File(h5_path, 'r') as f:
        dataset = f[data_key]
        n, d = dataset.shape
        print(f"[INFO] Dataset shape: {n} x {d}")

        # Determine sample size
        sample_size = min(n, max_sample_size)
        print(f"[INFO] Sampling {sample_size} points from dataset")

        # Randomly select start index and read sample_size points
        if sample_size < n:
            start_idx = np.random.randint(0, n - sample_size + 1)
            sampled_data = dataset[start_idx:start_idx + sample_size].astype(np.float32)
        else:
            sampled_data = dataset[:].astype(np.float32)

        # PCA
        print(f"[INFO] Performing PCA on sampled data with target dimension {pca_dim}")
        pca = PCA(n_components=pca_dim)
        pca.fit(sampled_data)

        # Save PCA model
        print(f"[INFO] Saving PCA model to {pca_model_path}")
        dump(pca, pca_model_path)

        # Batch PCA + normalization and write
        print(f"[INFO] Reducing all data to {pca_dim} dimensions, normalizing, and writing to {out_bin}")
        with open(out_bin, 'wb') as fout:
            fout.write(struct.pack('<I', n))
            fout.write(struct.pack('<I', pca_dim))
            row_bytes = pca_dim * 4
            pad = (align_bytes - (row_bytes % align_bytes)) % align_bytes

            bytes_per_point = d * 4
            max_points_per_batch = int((16 * 1024**3) / bytes_per_point)
            batch_size = min(max_points_per_batch, 100000)

            print(f"[INFO] Using batch size: {batch_size} points per batch")

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_data = dataset[start:end].astype(np.float32)
                reduced_batch = pca.transform(batch_data)
                # Normalize
                reduced_batch = normalize(reduced_batch, axis=1)
                for vec in reduced_batch:
                    fout.write(vec.astype(np.float32).tobytes())
                    if pad:
                        fout.write(b'\x00' * pad)

def export_queries_with_pca(h5_path, data_key, out_bin, pca_model_path, align_bytes=64):
    """
    Export query data with PCA and normalization.
    """
    from sklearn.preprocessing import normalize
    pca = load(pca_model_path)
    with h5py.File(h5_path, 'r') as f:
        dataset = f[data_key]
        n, d = dataset.shape
        pca_dim = pca.n_components_
        print(f"[INFO] Dataset shape: {n} x {d}")
        print(f"[INFO] Reducing queries to {pca_dim} dimensions using PCA")

        data = dataset[:].astype(np.float32)
        reduced_data = pca.transform(data) 

        reduced_data = normalize(reduced_data, axis=1)

        assert reduced_data.shape[1] == pca_dim, (
            f"[ERROR] Reduced dimension {reduced_data.shape[1]} does not match PCA dimension {pca_dim}"
        )

        row_bytes = pca_dim * 4
        pad = (align_bytes - (row_bytes % align_bytes)) % align_bytes

        with open(out_bin, 'wb') as fout:
            fout.write(struct.pack('<I', n))
            fout.write(struct.pack('<I', pca_dim))
            for vec in reduced_data:
                fout.write(vec.astype(np.float32).tobytes())
                if pad:
                    fout.write(b'\x00' * pad)

def export_gt_bin(h5_path, knns_key, dists_key, out_bin, index_base=1):
    """
    Export DiskANN groundtruth .bin format (n_query, K, n_query*K uint32, n_query*K float32)
    index_base: starting index of h5 knns (1 means need to subtract 1, 0 means no processing)
    """
    with h5py.File(h5_path, 'r') as f:
        knns = f[knns_key][:]
        dists = f[dists_key][:]
        n, k = knns.shape
        # If index starts from 1, subtract 1
        if index_base == 1:
            knns = knns - 1
        knns = knns.astype(np.uint32)
        dists = dists.astype(np.float32)
        # Sort by distance ascending (if not sorted)
        order = np.argsort(dists, axis=1)
        knns_sorted = np.take_along_axis(knns, order, axis=1)
        dists_sorted = np.take_along_axis(dists, order, axis=1)
        with open(out_bin, 'wb') as fout:
            fout.write(struct.pack('<i', n))
            fout.write(struct.pack('<i', k))
            fout.write(knns_sorted.tobytes())
            fout.write(dists_sorted.tobytes())
#---------------------------------------------------------------------------------

def main(args):
    DATASET_NAME = args.dataname
    H5_DIR = f'./data/{DATASET_NAME}/task1/'
    H5_FILE = f'benchmark-dev-{DATASET_NAME}.h5'
    h5_file = os.path.join(H5_DIR, H5_FILE)
    
    if not os.path.exists(h5_file):
        download_h5_if_needed(h5_file, DATASET_NAME)
           
    BUILD_DIR = os.path.join(H5_DIR, 'build')
    SEARCH_DIR = os.path.join(H5_DIR, 'search_results')
    print(f"[INFO] H5_FILE: {h5_file}")
    DATA_KEYS = {
        'train': 'train',
        'itest_queries': 'itest/queries',
        'otest_queries': 'otest/queries',
        'itest_knns': 'itest/knns',
        'otest_knns': 'otest/knns',
        'itest_dists': 'itest/dists',
        'otest_dists': 'otest/dists'
    }
    OUTPUTS = {
        'train_bin': os.path.join(BUILD_DIR, DATASET_NAME+'_train.bin'),
        'itest_queries_bin': os.path.join(BUILD_DIR, DATASET_NAME+'_itest_queries.bin'),
        'otest_queries_bin': os.path.join(BUILD_DIR, DATASET_NAME+'_otest_queries.bin'),
        'itest_knns_bin': os.path.join(BUILD_DIR, DATASET_NAME+'_itest_knns.bin'),
        'otest_knns_bin': os.path.join(BUILD_DIR, DATASET_NAME+'_otest_knns.bin'),
    }
    os.makedirs(BUILD_DIR, exist_ok=True)
    distance_metric = "l2"  # "l2", "mips", "cosine"
    BUILD_DISK_INDEX = './build/apps/build_disk_index'
    SEARCH_DISK_INDEX = './build/apps/search_disk_index'

    # Set PCA model save path
    PCA_MODEL_PATH = os.path.join(BUILD_DIR, DATASET_NAME + '_pca_model.joblib')

    # 1. Export otest/knns
    if not os.path.exists(OUTPUTS['otest_knns_bin']):
        print(f"[INFO] Exporting {DATA_KEYS['otest_knns']} → {OUTPUTS['otest_knns_bin']}")
        export_gt_bin(h5_file, DATA_KEYS['otest_knns'], DATA_KEYS['otest_dists'], OUTPUTS['otest_knns_bin'])
    else:
        print(f"[INFO] {OUTPUTS['otest_knns_bin']} already exists, skipping export")

    build_start = time.time()
    # 2. Export train
    if not os.path.exists(OUTPUTS['train_bin']):
        print(f"[INFO] Exporting {DATA_KEYS['train']} → {OUTPUTS['train_bin']}")
        # export_full_data(h5_file, DATA_KEYS['train'], OUTPUTS['train_bin'])
        export_full_data_with_pca(h5_file, DATA_KEYS['train'], OUTPUTS['train_bin'], PCA_MODEL_PATH)
    else:
        print(f"[INFO] {OUTPUTS['train_bin']} already exists, skipping export")

    # 3. Export otest/queries
    if not os.path.exists(OUTPUTS['otest_queries_bin']):
        print(f"[INFO] Exporting {DATA_KEYS['otest_queries']} → {OUTPUTS['otest_queries_bin']}")
        export_queries_with_pca(h5_file, DATA_KEYS['otest_queries'], OUTPUTS['otest_queries_bin'], PCA_MODEL_PATH)
        # export_full_data(h5_file, DATA_KEYS['otest_queries'], OUTPUTS['otest_queries_bin'])
    else:
        print(f"[INFO] {OUTPUTS['otest_queries_bin']} lready exists, skipping export")

    # 4. Build index (using train.bin)
    index_path_prefix = os.path.join(BUILD_DIR, DATASET_NAME)
    print(f"[INFO] Building index {index_path_prefix}")
    if not os.path.exists(index_path_prefix + '_disk.index'):
        print(f"[INFO] Building index {index_path_prefix}")
        build_cmd = [
            BUILD_DISK_INDEX,
            "--data_type", "float",
            "--dist_fn", distance_metric,
            "--data_path", OUTPUTS['train_bin'],
            "--index_path_prefix", index_path_prefix,
            "-R", str(args.R),
            "-L", str(args.LB),
            "-B", str(args.B),
            "-M", str(args.M),
            "-T", str(args.T),
        ]
        print(" ".join(build_cmd))
        result = run(build_cmd)
        if result.returncode != 0:
            print(f"[ERROR] build_disk_index failed")
            exit(1)
    else:
        print(f"[INFO] {index_path_prefix}_disk.index already exists, skipping build")
    build_end = time.time()
    build_time = build_end - build_start

    # 5. Search otest_queries
    os.makedirs(SEARCH_DIR, exist_ok=True)
    print(f"[INFO] Output results to {SEARCH_DIR}")
    SEARCH_DIR_PREFIX = os.path.join(SEARCH_DIR, DATASET_NAME)
    search_start = time.time()  
    if os.path.exists(OUTPUTS['otest_queries_bin']):
        print(f"[INFO] Searching {OUTPUTS['otest_queries_bin']} in {index_path_prefix}")
        search_cmd = [
            SEARCH_DISK_INDEX,
            "--data_type", "float",
            "--dist_fn", distance_metric,
            "--index_path_prefix", index_path_prefix,
            "--query_file", OUTPUTS['otest_queries_bin'],
            "--gt_file", OUTPUTS['otest_knns_bin'],
            "-K", str(args.K),
            "-L", *args.LS,  
            "--result_path", SEARCH_DIR_PREFIX,
            "--num_nodes_to_cache", "10000"
        ]
        print(" ".join(map(str, search_cmd)))
        result = run(search_cmd)
        if result.returncode != 0:
            print(f"[ERROR] search_disk_index failed")
            exit(1)
        print(f"[INFO] Search completed, results saved in {SEARCH_DIR}")
    else:
        print(f"[WARN] {OUTPUTS['otest_queries_bin']} not found, skipping search")
    search_end = time.time()  
    search_time = search_end - search_start

    # 6. Calculate recall
    if os.path.exists(OUTPUTS['otest_knns_bin']):
        print(f"[INFO] Calculating recall")
        gt_I = load_uint32_bin(OUTPUTS['otest_knns_bin'])
        search_results = load_uint32_bin(os.path.join(SEARCH_DIR,DATASET_NAME+'_200_idx_uint32.bin')) #SEARCH_DIR+DATASET_NAME+'_200_idx_uint32.bin'
        recall = get_recall(search_results, gt_I, k=30)
        print(f"[INFO] Recall: {recall:.4f}")
    else:
        print(f"[WARN] {OUTPUTS['otest_knns_bin']} not found, cannot calculate recall")

    # 7. Save results to CSV file
    params = {
    "R": args.R,
    "LB": args.LB,
    "B": args.B,
    "M": args.M,
    "T": args.T,
    "LS": args.LS,
    "K": args.K,
    "distance_metric": distance_metric,
    }
    params_str = str(params)
    d = {
    "dataset": args.dataname,
    "task": "task1",
    "algo": "diskann",
    "buildtime": build_time,
    "querytime": search_time,
    "params": params_str,
    "recall": recall
    }
    columns = ["dataset", "task", "algo", "buildtime", "querytime", "params", "recall"]
    with open(args.results, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()      
        print(d["dataset"], d["task"], d["algo"], d["params"], "=>", recall)
        writer.writerow(d)

if __name__ == "__main__":
    
    LOG_FILE = "sys_usage.log"
    FILE_CHANGE_LOG = "file_changes.log"
    INTERVAL = 2
    WATCH_DIR = os.path.expanduser("./data/build")
    os.makedirs(WATCH_DIR, exist_ok=True)

    stop_event = Event()
    monitor_proc = Process(target=monitor_resources, args=(LOG_FILE, INTERVAL, stop_event, WATCH_DIR))
    file_monitor_proc = Process(target=monitor_file_changes, args=(WATCH_DIR, FILE_CHANGE_LOG, INTERVAL, stop_event))
    monitor_proc.start()
    file_monitor_proc.start()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help='directory in which results are stored',
        default="./results.csv"
    )
    parser.add_argument(
        "--dataname",
        help='dataset name, e.g., ccnews, ccnews-fp16, pubmed23, etc.',
        default="ccnews-fp16"
    )
    parser.add_argument(
        "--R",
        help='parameter R, max_degree',
        default=64
    )
    parser.add_argument(
        "--LB",
        help='Parameter L_build, the size of search list during index build',
        default=100
    )   
    parser.add_argument(
        "--B",
        help='Parameter B, search_DRAM_budget for DiskANN',
        default=3
    )  
    parser.add_argument(
        "--M",
        help='Parameter M, build_DRAM_budget',
        default=4
    ) 
    parser.add_argument(
        "--T",
        help='Parameter T, num_threads',
        default=8
    )  
    parser.add_argument(
        "-LS",
        nargs='+',
        type=str,
        help='Parameter LS, the size of search list during search, e.g. -L 30 40 50 100',
        default=["200"]
    )  
    parser.add_argument(
        "--K",
        help='Parameter K, the number of nearest neighbors search',
        default=30
    )      
    args = parser.parse_args()

    print("[INFO] Resource monitoring started, main program will start in 10 seconds...")
    time.sleep(10)

    try:
        main(args)
    finally:
        stop_event.set()
        monitor_proc.join()
        file_monitor_proc.join()
        print("[INFO] Resource monitoring stopped")
    # main()