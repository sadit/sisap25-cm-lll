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


##########################download数据集###############################
def download_h5_if_needed(h5_file, dataset_name):
    # 你可以根据实际情况修改下载链接
    url = f"https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-{dataset_name}.h5?download=true"
    print(f"[INFO] {h5_file} 不存在，尝试从 {url} 下载...")
    os.makedirs(os.path.dirname(h5_file), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, h5_file)
        print(f"[INFO] 下载完成: {h5_file}")
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        exit(1)
###########################download数据集###############################

##########################结果读取以及recall计算###############################
def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)

def load_uint32_bin(bin_path):
    """
    读取 DiskANN .bin 文件（uint32类型），返回 shape=(n, k) 的 numpy 数组
    文件格式: int32 n, int32 k, n*k uint32
    """
    with open(bin_path, 'rb') as f:
        n, k = struct.unpack('<ii', f.read(8))
        arr = np.frombuffer(f.read(n * k * 4), dtype=np.uint32).reshape(n, k)
    return arr
##########################结果读取以及recall计算###############################

########################数据读取以及预处理#############################
def export_full_data(h5_path, data_key, out_bin, align_bytes=64):
    """
    导出数据为 .bin 格式, 不做PCA降维, 只做对齐。
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
    导出查询数据为 .bin 格式, 不做PCA降维, 只做对齐。
    """
    export_full_data(h5_path, data_key, out_bin, align_bytes)

from sklearn.preprocessing import normalize

def export_full_data_with_pca(h5_path, data_key, out_bin, pca_model_path, align_bytes=64, max_sample_size=100000, pca_dim=192):
    """
    导出数据为 .bin 格式, PCA降维后归一化, 分批处理。
    """
    with h5py.File(h5_path, 'r') as f:
        dataset = f[data_key]
        n, d = dataset.shape
        print(f"[INFO] Dataset shape: {n} x {d}")

        # 确定抽样数量
        sample_size = min(n, max_sample_size)
        print(f"[INFO] Sampling {sample_size} points from dataset")

        # 随机选择起始索引并连续读取 sample_size 个数据点
        if sample_size < n:
            start_idx = np.random.randint(0, n - sample_size + 1)
            sampled_data = dataset[start_idx:start_idx + sample_size].astype(np.float32)
        else:
            sampled_data = dataset[:].astype(np.float32)

        # PCA
        print(f"[INFO] Performing PCA on sampled data with target dimension {pca_dim}")
        pca = PCA(n_components=pca_dim)
        pca.fit(sampled_data)

        # 保存 PCA 模型
        print(f"[INFO] Saving PCA model to {pca_model_path}")
        dump(pca, pca_model_path)

        # 分批降维+归一化并写入
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
                # 归一化
                reduced_batch = normalize(reduced_batch, axis=1)
                for vec in reduced_batch:
                    fout.write(vec.astype(np.float32).tobytes())
                    if pad:
                        fout.write(b'\x00' * pad)

def export_queries_with_pca(h5_path, data_key, out_bin, pca_model_path, align_bytes=64):
    """
    查询数据PCA降维后归一化。
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
        # 归一化
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
    导出 DiskANN groundtruth .bin 格式(n_query, K, n_query*K uint32, n_query*K float32)
    index_base: h5 knns的起始编号(1表示需要-1, 0表示无需处理)
    """
    with h5py.File(h5_path, 'r') as f:
        knns = f[knns_key][:]
        dists = f[dists_key][:]
        n, k = knns.shape
        # 下标从1开始要-1
        if index_base == 1:
            knns = knns - 1
        knns = knns.astype(np.uint32)
        dists = dists.astype(np.float32)
        # 按距离升序排序（如果未排序）
        order = np.argsort(dists, axis=1)
        knns_sorted = np.take_along_axis(knns, order, axis=1)
        dists_sorted = np.take_along_axis(dists, order, axis=1)
        with open(out_bin, 'wb') as fout:
            fout.write(struct.pack('<i', n))
            fout.write(struct.pack('<i', k))
            fout.write(knns_sorted.tobytes())
            fout.write(dists_sorted.tobytes())

########################数据预处理#############################
def main(args):
    # 配置参数
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
    distance_metric = "l2"  # 可选: "l2", "mips", "cosine"
    BUILD_DISK_INDEX = './build/apps/build_disk_index'
    SEARCH_DISK_INDEX = './build/apps/search_disk_index'

    # 配置 PCA 模型保存路径
    PCA_MODEL_PATH = os.path.join(BUILD_DIR, DATASET_NAME + '_pca_model.joblib')

    # 1. 导出 otest/knns
    if not os.path.exists(OUTPUTS['otest_knns_bin']):
        print(f"[INFO] 导出 {DATA_KEYS['otest_knns']} → {OUTPUTS['otest_knns_bin']}")
        export_gt_bin(h5_file, DATA_KEYS['otest_knns'], DATA_KEYS['otest_dists'], OUTPUTS['otest_knns_bin'])
    else:
        print(f"[INFO] 已存在 {OUTPUTS['otest_knns_bin']}，跳过导出")

    build_start = time.time()

    # 2. 计算召回率
    if os.path.exists(OUTPUTS['otest_knns_bin']):
        print(f"[INFO] 计算召回率")
        gt_I = load_uint32_bin(OUTPUTS['otest_knns_bin'])
        search_results = load_uint32_bin(os.path.join(SEARCH_DIR,DATASET_NAME+'_200_idx_uint32.bin')) #SEARCH_DIR+DATASET_NAME+'_200_idx_uint32.bin'
        recall = get_recall(search_results, gt_I, k=30)
        print(f"[INFO] 召回率: {recall:.4f}")
    else:
        print(f"[WARN] 未找到 {OUTPUTS['otest_knns_bin']}，无法计算召回率")

    # 保存结果到 CSV 文件
    params = {
    "R": args.R,
    "LB": args.LB,
    "B": args.B,
    "M": args.M,
    "T": args.T,
    "LS": args.LS,
    "K": args.K,
    "distance_metric": distance_metric,
    "pca_dim": 192,  # 如有其它参数也可加进来
    # 你可以继续添加其它关键参数
    }
    params_str = str(params)
    d = {
    "dataset": args.dataname,
    "task": "task1",
    "algo": "diskann",
    "params": params_str,
    "recall": recall
    }
    columns = ["dataset", "task", "algo", "params", "recall"]
    # with open(args.results, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=columns)
    #     writer.writeheader()      
    #     print(d["dataset"], d["task"], d["algo"], d["params"], "=>", recall)
    #     writer.writerow(d)

if __name__ == "__main__":
    LOG_FILE = "sys_usage.log"
    FILE_CHANGE_LOG = "file_changes.log"

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
        "-K",
        help='Parameter K, the number of nearest neighbors search',
        default=30
    )      
    args = parser.parse_args()
    
    main(args)
    
    # main()