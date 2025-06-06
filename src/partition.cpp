// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include <omp.h>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "utils.h"
#include "math_utils.h"
#include "index.h"
#include "parameters.h"
#include "memory_mapper.h"
#include "partition.h"
#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include <thread>
#include <chrono>

// block size for reading/ processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

// #define SAVE_INFLATED_PQ true

template <typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(base_file.c_str(), read_blk_size);
    std::ofstream sample_writer(std::string(output_prefix + "_data.bin").c_str(), std::ios::binary);
    std::ofstream sample_id_writer(std::string(output_prefix + "_ids.bin").c_str(), std::ios::binary);

    std::random_device rd; // Will be used to obtain a seed for the random number engine
    auto x = rd();
    std::mt19937 generator(x); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> distribution(0, 1);

    size_t npts, nd;
    uint32_t npts_u32, nd_u32;
    uint32_t num_sampled_pts_u32 = 0;
    uint32_t one_const = 1;

    base_reader.read((char *)&npts_u32, sizeof(uint32_t));
    base_reader.read((char *)&nd_u32, sizeof(uint32_t));
    diskann::cout << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << "."
                  << std::endl;
    sample_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
    sample_writer.write((char *)&nd_u32, sizeof(uint32_t));
    sample_id_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
    sample_id_writer.write((char *)&one_const, sizeof(uint32_t));

    npts = npts_u32;
    nd = nd_u32;
    std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd);

    for (size_t i = 0; i < npts; i++)
    {
        base_reader.read((char *)cur_row.get(), sizeof(T) * nd);
        float sample = distribution(generator);
        if (sample < sampling_rate)
        {
            sample_writer.write((char *)cur_row.get(), sizeof(T) * nd);
            uint32_t cur_i_u32 = (uint32_t)i;
            sample_id_writer.write((char *)&cur_i_u32, sizeof(uint32_t));
            num_sampled_pts_u32++;
        }
    }
    sample_writer.seekp(0, std::ios::beg);
    sample_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
    sample_id_writer.seekp(0, std::ios::beg);
    sample_id_writer.write((char *)&num_sampled_pts_u32, sizeof(uint32_t));
    sample_writer.close();
    sample_id_writer.close();
    diskann::cout << "Wrote " << num_sampled_pts_u32 << " points to sample file: " << output_prefix + "_data.bin"
                  << std::endl;
}

// streams data from the file, and samples each vector with probability p_val
// and returns a matrix of size slice_size* ndims as floating point type.
// the slice_size and ndims are set inside the function.

/***********************************
 * Reimplement using gen_random_slice(const T* inputdata,...)
 ************************************/

template <typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims)
{
    size_t npts;
    uint32_t npts32, ndims32;
    std::vector<std::vector<float>> sampled_vectors;

    // amount to read in one shot
    size_t read_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file.c_str(), read_blk_size);

    // metadata: npts, ndims
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&ndims32, sizeof(uint32_t));
    npts = npts32;
    ndims = ndims32;

    std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(ndims);
    p_val = p_val < 1 ? p_val : 1;

    std::random_device rd; // Will be used to obtain a seed for the random number
    size_t x = rd();
    std::mt19937 generator((uint32_t)x);
    std::uniform_real_distribution<float> distribution(0, 1);

    for (size_t i = 0; i < npts; i++)
    {
        base_reader.read((char *)cur_vector_T.get(), ndims * sizeof(T));
        float rnd_val = distribution(generator);
        if (rnd_val < p_val)
        {
            std::vector<float> cur_vector_float;
            for (size_t d = 0; d < ndims; d++)
                cur_vector_float.push_back(cur_vector_T[d]);
            sampled_vectors.push_back(cur_vector_float);
        }
    }
    slice_size = sampled_vectors.size();
    sampled_data = new float[slice_size * ndims];
    for (size_t i = 0; i < slice_size; i++)
    {
        for (size_t j = 0; j < ndims; j++)
        {
            sampled_data[i * ndims + j] = sampled_vectors[i][j];
        }
    }
}

// same as above, but samples from the matrix inputdata instead of a file of
// npts*ndims to return sampled_data of size slice_size*ndims.
template <typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims, double p_val, float *&sampled_data,
                      size_t &slice_size)
{
    std::vector<std::vector<float>> sampled_vectors;
    const T *cur_vector_T;

    p_val = p_val < 1 ? p_val : 1;

    std::random_device rd; // Will be used to obtain a seed for the random number engine
    size_t x = rd();
    std::mt19937 generator((uint32_t)x); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> distribution(0, 1);

    for (size_t i = 0; i < npts; i++)
    {
        cur_vector_T = inputdata + ndims * i;
        float rnd_val = distribution(generator);
        if (rnd_val < p_val)
        {
            std::vector<float> cur_vector_float;
            for (size_t d = 0; d < ndims; d++)
                cur_vector_float.push_back(cur_vector_T[d]);
            sampled_vectors.push_back(cur_vector_float);
        }
    }
    slice_size = sampled_vectors.size();
    sampled_data = new float[slice_size * ndims];
    for (size_t i = 0; i < slice_size; i++)
    {
        for (size_t j = 0; j < ndims; j++)
        {
            sampled_data[i * ndims + j] = sampled_vectors[i][j];
        }
    }
}

int estimate_cluster_sizes(float *test_data_float, size_t num_test, float *pivots, const size_t num_centers,
                           const size_t test_dim, const size_t k_base, std::vector<size_t> &cluster_sizes)
{
    cluster_sizes.clear();

    size_t *shard_counts = new size_t[num_centers];

    for (size_t i = 0; i < num_centers; i++)
    {
        shard_counts[i] = 0;
    }

    size_t block_size = num_test <= BLOCK_SIZE ? num_test : BLOCK_SIZE;
    uint32_t *block_closest_centers = new uint32_t[block_size * k_base];
    float *block_data_float;

    size_t num_blocks = DIV_ROUND_UP(num_test, block_size);

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_test);
        size_t cur_blk_size = end_id - start_id;

        block_data_float = test_data_float + start_id * test_dim;

        math_utils::compute_closest_centers(block_data_float, cur_blk_size, test_dim, pivots, num_centers, k_base,
                                            block_closest_centers);

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            for (size_t p1 = 0; p1 < k_base; p1++)
            {
                size_t shard_id = block_closest_centers[p * k_base + p1];
                shard_counts[shard_id]++;
            }
        }
    }

    diskann::cout << "Estimated cluster sizes: ";
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i];
        cluster_sizes.push_back((size_t)cur_shard_count);
        diskann::cout << cur_shard_count << " ";
    }
    diskann::cout << std::endl;
    delete[] shard_counts;
    delete[] block_closest_centers;
    return 0;
}

template <typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        return -1;
    }

    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {
        std::string data_filename = prefix_path + "_subshard-" + std::to_string(i) + ".bin";
        std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i] = 0;
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * k_base);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                            block_closest_centers.get());

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            for (size_t p1 = 0; p1 < k_base; p1++)
            {
                size_t shard_id = block_closest_centers[p * k_base + p1];
                uint32_t original_point_map_id = (uint32_t)(start_id + p);
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
                shard_counts[shard_id]++;
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i];
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
    return 0;
}

// useful for partitioning large dataset. we first generate only the IDS for
// each shard, and retrieve the actual vectors on demand.
template <typename T>
int shard_data_into_clusters_only_ids(const std::string data_file, float *pivots, const size_t num_centers,
                                      const size_t dim, const size_t k_base, std::string prefix_path)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        return -1;
    }

    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);

    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {
        std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i] = 0;
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * k_base);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);


    // 初始化存储最近距离的数组
    std::unique_ptr<float[]> block_closest_distances = std::make_unique<float[]>(block_size * k_base);


    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_float.get(), sizeof(float) * (cur_blk_size * dim));

        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                            block_closest_centers.get(), nullptr, nullptr, block_closest_distances.get());
        

        std::vector<float> second_closest_distances; // 用于存储次近距离

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            // 收集每个点的次近距离
            second_closest_distances.push_back(block_closest_distances[p * k_base + 1]);
        }

        // 计算这一批次的次近距离的中位数
        float median_distance = 0.0f;
        if (!second_closest_distances.empty())
        {
            size_t mid_index = second_closest_distances.size() / 2;
            std::nth_element(second_closest_distances.begin(), second_closest_distances.begin() + mid_index, second_closest_distances.end());
            median_distance = second_closest_distances[mid_index];

            // 如果元素数量是偶数，取中间两个元素的平均值
            if (second_closest_distances.size() % 2 == 0)
            {
                std::nth_element(second_closest_distances.begin(), second_closest_distances.begin() + mid_index - 1, second_closest_distances.end());
                median_distance = (median_distance + second_closest_distances[mid_index - 1]) / 2.0f;
            }
        }

        diskann::cout << "Median of second closest distances: " << median_distance << std::endl;

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            for (size_t p1 = 0; p1 < k_base; p1++)
            {
                if (p1 == 1 && block_closest_distances[p * k_base + p1] >= median_distance)
                {
                    continue; // 如果次近距离 中位数，则跳过
                }
                size_t shard_id = block_closest_centers[p * k_base + p1];
                uint32_t original_point_map_id = (uint32_t)(start_id + p);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
                shard_counts[shard_id]++;
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i];
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
    return 0;
}

template <typename T>
int retrieve_shard_data_from_ids(const std::string data_file, std::string idmap_filename, std::string data_filename)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;

    uint32_t dummy_size = 0;

    std::ofstream shard_data_writer(data_filename.c_str(), std::ios::binary);
    shard_data_writer.write((char *)&dummy_size, sizeof(uint32_t));
    shard_data_writer.write((char *)&basedim32, sizeof(uint32_t));

    uint32_t *shard_ids;
    uint64_t shard_size, tmp;
    diskann::load_bin<uint32_t>(idmap_filename, shard_ids, shard_size, tmp);

    uint32_t cur_pos = 0;
    uint32_t num_written = 0;
    std::cout << "Shard has " << shard_size << " points" << std::endl;

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            uint32_t original_point_map_id = (uint32_t)(start_id + p);
            if (cur_pos == shard_size)
                break;
            if (original_point_map_id == shard_ids[cur_pos])
            {
                cur_pos++;
                shard_data_writer.write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                num_written++;
            }
        }
        if (cur_pos == shard_size)
            break;
    }

    diskann::cout << "Written file with " << num_written << " points" << std::endl;

    shard_data_writer.seekp(0);
    shard_data_writer.write((char *)&num_written, sizeof(uint32_t));
    shard_data_writer.close();
    delete[] shard_ids;
    return 0;
}

// partitions a large base file into many shards using k-means hueristic
// on a random sample generated using sampling_rate probability. After this, it
// assignes each base point to the closest k_base nearest centers and creates
// the shards.
// The total number of points across all shards will be k_base * num_points.

template <typename T>
int partition(const std::string data_file, const float sampling_rate, size_t num_parts, size_t max_k_means_reps,
              const std::string prefix_path, size_t k_base)
{
    size_t train_dim;
    size_t num_train;
    float *train_data_float;

    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    float *pivot_data;

    std::string cur_file = std::string(prefix_path);
    std::string output_file;

    // kmeans_partitioning on training data

    //  cur_file = cur_file + "_kmeans_partitioning-" +
    //  std::to_string(num_parts);
    output_file = cur_file + "_centroids.bin";

    pivot_data = new float[num_parts * train_dim];

    // Process Global k-means for kmeans_partitioning Step
    diskann::cout << "Processing global k-means (kmeans_partitioning Step)" << std::endl;
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);

    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

    diskann::cout << "Saving global k-center pivots" << std::endl;
    diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

    // now pivots are ready. need to stream base points and assign them to
    // closest clusters.

    shard_data_into_clusters<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
    delete[] pivot_data;
    delete[] train_data_float;
    return 0;
}

size_t calculate_num_parts(double ram_budget, size_t total_points, size_t dim, size_t point_size) {
    // 转换 ram_budget 为字节
    double max_ram_per_shard = ram_budget * 1024 * 1024 * 1024;

    // 初始 num_part 的范围
    size_t low = 3; // 最少 3 个 shard
    size_t high = total_points; // 最多每个点一个 shard

    while (low < high) {
        size_t mid = (low + high) / 2;

        // 计算每个 shard 的内存占用
        double shard_ram_usage = (static_cast<double>(total_points) / mid) * dim * point_size;

        // 如果内存占用小于等于 ram_budget，则尝试减少 num_part
        if (shard_ram_usage <= max_ram_per_shard) {
            high = mid; // 缩小范围，尝试更小的 num_part
        } else {
            low = mid + 1; // 增大范围，尝试更大的 num_part
        }
    }

    return low; // 返回满足条件的最小 num_part
}

template <typename T>
void load_global_pivots(const std::string &filename, T *&pivot_data, size_t &num_centers, size_t &dim) {
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // 读取聚类中心的数量和维度
    int num_centers_i32, dim_i32;
    reader.read(reinterpret_cast<char *>(&num_centers_i32), sizeof(int));
    reader.read(reinterpret_cast<char *>(&dim_i32), sizeof(int));

    num_centers = static_cast<size_t>(num_centers_i32);
    dim = static_cast<size_t>(dim_i32);

    // 分配内存并读取聚类中心数据
    pivot_data = new T[num_centers * dim];
    reader.read(reinterpret_cast<char *>(pivot_data), num_centers * dim * sizeof(T));

    reader.close();
    diskann::cout << "Loaded global pivots from " << filename << ": " << num_centers
                  << " centers, dimension " << dim << "." << std::endl;
}

// helper: 计算一个点到多个 pivots 的平方欧氏距离并获取 k 最近的索引列表
static inline void knn_indices(const float* point,
                               const float* pivots,
                               size_t pivots_count,
                               size_t dim,
                               size_t k,
                               std::vector<size_t>& out_idxs) {
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(pivots_count);
    for (size_t j = 0; j < pivots_count; ++j) {
        float dist = 0;
        const float* pj = pivots + j * dim;
        for (size_t d = 0; d < dim; ++d)
            dist += (point[d] - pj[d]) * (point[d] - pj[d]);
        dists.emplace_back(dist, j);
    }
    // 部分排序找到前 k
    size_t kk = std::min(k, pivots_count);
    std::nth_element(dists.begin(), dists.begin() + kk, dists.end(),
                     [](auto &a, auto &b) { return a.first < b.first; });
    out_idxs.clear(); out_idxs.reserve(kk);
    for (size_t t = 0; t < kk; ++t)
        out_idxs.push_back(dists[t].second);
}

// // 主函数
// template <typename T>
// int partition_with_ram_budget(const std::string data_file,
//                               const double sampling_rate,
//                               double ram_budget,
//                               size_t graph_degree,
//                               const std::string prefix_path,
//                               size_t k_base) {

//     size_t total_points, dim;
//     diskann::get_bin_metadata(data_file, total_points, dim);

//     // 采样一小部分点
//     float* sample_data = nullptr;
//     size_t num_sample = 0, sample_dim = 0;
//     gen_random_slice<T>(data_file, sampling_rate, sample_data, num_sample, sample_dim);
//     if (sample_dim != dim) throw std::runtime_error("Sample dimension mismatch.");

//     // 加载全局 pivots
//     float* pivot_data = nullptr;
//     size_t total_pivots = 0, pivot_dim = 0;
//     std::string global_pivots_file =  "global_pivot_vec.bin";
//     load_global_pivots<float>(global_pivots_file, pivot_data, total_pivots, pivot_dim);
//     if (pivot_dim != dim) throw std::runtime_error("Dimension mismatch.");
//     // // 删除临时文件全局 pivots
//     // if (std::remove(global_pivots_file.c_str()) != 0) {
//     //     diskann::cerr << "Warning: failed to remove file " << global_pivots_file << std::endl;
//     // }

//     // 初始化变量
//     std::vector<size_t> chosen_idxs;
//     std::vector<size_t> buffer_idxs;
//     std::vector<size_t> sample_shard_counts;
//     size_t num_parts = 3;

//     // 循环选择最均衡的 pivots
//     while (true) {
//         // 1) 全局 k_base 计数
//         std::vector<size_t> counts(total_pivots, 0);
//         for (size_t i = 0; i < num_sample; ++i) {
//             knn_indices(sample_data + i * dim, pivot_data, total_pivots, dim, k_base, buffer_idxs);
//             for (auto idx : buffer_idxs) ++counts[idx];
//         }
//         double avg_load = double(num_sample) * k_base / num_parts;
//         std::vector<std::pair<double, size_t>> dev;
//         dev.reserve(total_pivots);
//         for (size_t j = 0; j < total_pivots; ++j)
//             dev.emplace_back(std::abs(double(counts[j]) - avg_load), j);
//         std::nth_element(dev.begin(), dev.begin() + num_parts, dev.end(),
//                         [](auto &a, auto &b) { return a.first < b.first; });
//         // 更新 chosen_idxs
//         chosen_idxs.clear();
//         chosen_idxs.reserve(num_parts);
//         for (size_t t = 0; t < num_parts; ++t)
//             chosen_idxs.push_back(dev[t].second);

//         // 2) 估算 sample_shard_counts
//         sample_shard_counts.assign(num_parts, 0);
//         for (size_t i = 0; i < num_sample; ++i) {
//             const float *pt = sample_data + i * dim;
//             std::vector<std::pair<float, size_t>> ld;
//             ld.reserve(num_parts);
//             for (size_t t = 0; t < num_parts; ++t) {
//                 size_t pj = chosen_idxs[t];
//                 float dist = 0;
//                 const float *q = pivot_data + pj * dim;
//                 for (size_t d = 0; d < dim; ++d)
//                     dist += (pt[d] - q[d]) * (pt[d] - q[d]);
//                 ld.emplace_back(dist, t);
//             }
//             size_t kk2 = std::min(k_base, num_parts);
//             std::nth_element(ld.begin(), ld.begin() + kk2, ld.end(),
//                             [](auto &a, auto &b) { return a.first < b.first; });
//             for (size_t t = 0; t < kk2; ++t)
//                 ++sample_shard_counts[ld[t].second];
//         }
//         // 3) 估算 RAM
//         double max_ram = 0;
//         for (size_t t = 0; t < num_parts; ++t) {
//             double est_cnt = sample_shard_counts[t] / sampling_rate;
//             max_ram = std::max(max_ram,
//                             diskann::estimate_ram_usage(uint64_t(std::ceil(est_cnt)),
//                                                         (uint32_t)dim, sizeof(T), (uint32_t)graph_degree));
//         }
//         diskann::cout << "#parts=" << num_parts << " est_ram=" << max_ram / (1e9)
//                     << "GB budget=" << ram_budget << "GB" << std::endl;

//         // 如果满足内存预算，或者 num_parts 达到最大值 10，则退出循环
//         if (max_ram <= ram_budget * 1024 * 1024 * 1024 || num_parts >= 10) break;

//         num_parts += 2;
//     }
//     // 输出最终结果在采样上的分片详情
//     diskann::cout << "Final chosen pivots indices: ";
//     for (auto idx : chosen_idxs) diskann::cout << idx << " ";
//     diskann::cout << std::endl;
//     diskann::cout << "Sample shard counts: ";
//     for (auto cnt : sample_shard_counts) diskann::cout << cnt << " ";
//     diskann::cout << std::endl;

//     // 构建 selected_pivots 并分片
//     float* selected_pivots = new float[num_parts * dim];
//     for (size_t t = 0; t < num_parts; ++t) {
//         std::memcpy(selected_pivots + t*dim,
//                     pivot_data + chosen_idxs[t]*dim,
//                     dim * sizeof(float));
//     }
//     shard_data_into_clusters_only_ids<T>(data_file, selected_pivots,
//                                          num_parts, dim, k_base, prefix_path);

//     // 清理
//     delete[] sample_data;
//     delete[] pivot_data;
//     delete[] selected_pivots;
//     return int(num_parts);
// }

template <typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base)
{
    size_t train_dim;
    size_t num_train;
    float *train_data_float;
    size_t max_k_means_reps = 10;

    int num_parts = 3;
    bool fit_in_ram = false;

    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    size_t test_dim;
    size_t num_test;
    float *test_data_float;
    gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test, test_dim);

    float *pivot_data = nullptr;

    std::string cur_file = std::string(prefix_path);
    std::string output_file;

    // kmeans_partitioning on training data

    //  cur_file = cur_file + "_kmeans_partitioning-" +
    //  std::to_string(num_parts);
    output_file = cur_file + "_centroids.bin";

    while (!fit_in_ram)
    {
        fit_in_ram = true;

        double max_ram_usage = 0;
        if (pivot_data != nullptr)
            delete[] pivot_data;

        pivot_data = new float[num_parts * train_dim];
        // Process Global k-means for kmeans_partitioning Step
        diskann::cout << "Processing global k-means (kmeans_partitioning Step)" << std::endl;
        kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);

        kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

        // now pivots are ready. need to stream base points and assign them to
        // closest clusters.

        std::vector<size_t> cluster_sizes;
        estimate_cluster_sizes(test_data_float, num_test, pivot_data, num_parts, train_dim, k_base, cluster_sizes);

        for (auto &p : cluster_sizes)
        {
            // to account for the fact that p is the size of the shard over the
            // testing sample.
            p = (uint64_t)(p / sampling_rate);
            double cur_shard_ram_estimate =
                diskann::estimate_ram_usage(p, (uint32_t)train_dim, sizeof(T), (uint32_t)graph_degree);

            if (cur_shard_ram_estimate > max_ram_usage)
                max_ram_usage = cur_shard_ram_estimate;
        }
        diskann::cout << "With " << num_parts
                      << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
                      << "GB, budget given is " << ram_budget << std::endl;
        if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget)
        {
            fit_in_ram = false;
            num_parts += 2;
        }
    }

    diskann::cout << "Saving global k-center pivots" << std::endl;
    diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

    shard_data_into_clusters_only_ids<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
    delete[] pivot_data;
    delete[] train_data_float;
    delete[] test_data_float;
    return num_parts;
}

// Instantations of supported templates

template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(const std::string base_file, const std::string output_prefix,
                                                         double sampling_rate);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(const std::string base_file, const std::string output_prefix,
                                                          double sampling_rate);
template void DISKANN_DLLEXPORT gen_random_slice<float>(const std::string base_file, const std::string output_prefix,
                                                        double sampling_rate);

template void DISKANN_DLLEXPORT gen_random_slice<float>(const float *inputdata, size_t npts, size_t ndims, double p_val,
                                                        float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(const uint8_t *inputdata, size_t npts, size_t ndims,
                                                          double p_val, float *&sampled_data, size_t &slice_size);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(const int8_t *inputdata, size_t npts, size_t ndims,
                                                         double p_val, float *&sampled_data, size_t &slice_size);

template void DISKANN_DLLEXPORT gen_random_slice<float>(const std::string data_file, double p_val, float *&sampled_data,
                                                        size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<uint8_t>(const std::string data_file, double p_val,
                                                          float *&sampled_data, size_t &slice_size, size_t &ndims);
template void DISKANN_DLLEXPORT gen_random_slice<int8_t>(const std::string data_file, double p_val,
                                                         float *&sampled_data, size_t &slice_size, size_t &ndims);

template DISKANN_DLLEXPORT int partition<int8_t>(const std::string data_file, const float sampling_rate,
                                                 size_t num_centers, size_t max_k_means_reps,
                                                 const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<uint8_t>(const std::string data_file, const float sampling_rate,
                                                  size_t num_centers, size_t max_k_means_reps,
                                                  const std::string prefix_path, size_t k_base);
template DISKANN_DLLEXPORT int partition<float>(const std::string data_file, const float sampling_rate,
                                                size_t num_centers, size_t max_k_means_reps,
                                                const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int partition_with_ram_budget<int8_t>(const std::string data_file,
                                                                 const double sampling_rate, double ram_budget,
                                                                 size_t graph_degree, const std::string prefix_path,
                                                                 size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<uint8_t>(const std::string data_file,
                                                                  const double sampling_rate, double ram_budget,
                                                                  size_t graph_degree, const std::string prefix_path,
                                                                  size_t k_base);
template DISKANN_DLLEXPORT int partition_with_ram_budget<float>(const std::string data_file, const double sampling_rate,
                                                                double ram_budget, size_t graph_degree,
                                                                const std::string prefix_path, size_t k_base);

template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<float>(const std::string data_file,
                                                                   std::string idmap_filename,
                                                                   std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<uint8_t>(const std::string data_file,
                                                                     std::string idmap_filename,
                                                                     std::string data_filename);
template DISKANN_DLLEXPORT int retrieve_shard_data_from_ids<int8_t>(const std::string data_file,
                                                                    std::string idmap_filename,
                                                                    std::string data_filename);