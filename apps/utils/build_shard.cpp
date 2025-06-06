// build_shard.cpp
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <boost/program_options.hpp>

#include "disk_utils.h" // get_file_size etc.
#include "utils.h"      // get_bin_metadata, copy_file
#include "cached_io.h"  // cached_ifstream/ofstream
#include "index.h"      // diskann::Index, IndexWriteParametersBuilder
#include "program_options_utils.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string input_bin, output_index;
    uint32_t R, L, num_threads, build_pq_bytes;
    bool use_opq = false;

    po::options_description desc("build_shard options");
    desc.add_options()
        ("help,h", "Print usage")
        ("input,i",  po::value<std::string>(&input_bin)->required(), "Input .bin file")
        ("output,o", po::value<std::string>(&output_index)->required(),"Output shard index (.mem.index)")
        ("R",        po::value<uint32_t>(&R)->default_value(64),      "Max graph degree")
        ("L",        po::value<uint32_t>(&L)->default_value(100),     "Candidate list size")
        ("T",        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()), "Thread count")
        ("pq_bytes", po::value<uint32_t>(&build_pq_bytes)->default_value(0), "PQ bytes for build (0=full precision)")
        ("opq",      po::bool_switch(&use_opq)->default_value(false), "Enable OPQ");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    po::notify(vm);

    // 1. 读取元数据
    size_t npts, dim;
    diskann::get_bin_metadata(input_bin, npts, dim);

    // 2. 构建 IndexWriteParameters
    auto params = diskann::IndexWriteParametersBuilder(L, R)
                      .with_num_threads(num_threads)
                      .build();

    // 3. 实例化 diskann::Index
    // 与原 build_merged_vamana_index 单机分支一致
    using T = float;
    using TagT = uint32_t;
    diskann::Index<T, TagT> index(
        diskann::Metric::L2,
        /* dim */ dim,
        /* npts */ npts,
        std::make_shared<diskann::IndexWriteParameters>(params),
        /* tags */ nullptr,
        /* num_frozen_pts */ diskann::defaults::NUM_FROZEN_POINTS_STATIC,
        /* use_threads */ false,
        /* use_pq */ (build_pq_bytes > 0),
        /* use_opq */ use_opq,
        /* use_filters */ false,
        /* build_pq_bytes */ build_pq_bytes,
        /* reorder_data */ false,
        /* dummy */ false
    );

    // 4. 构图
    index.build(input_bin.c_str(), npts);

    // 5. 保存
    index.save(output_index.c_str());

    std::cout << "Built shard index: " << output_index << "\n";
    return 0;
}
