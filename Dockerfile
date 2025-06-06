#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

# FROM ubuntu:jammy

# RUN apt update
# RUN apt install -y software-properties-common
# RUN add-apt-repository -y ppa:git-core/ppa
# RUN apt update
# RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10

# WORKDIR /app
# RUN git clone https://github.com/microsoft/DiskANN.git 
# WORKDIR /app/DiskANN
# RUN mkdir build
# RUN cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release
# RUN cmake --build build -- -j

FROM ubuntu:jammy
# 安装依赖
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        git make cmake g++ libaio-dev libgoogle-perftools-dev \
        libunwind-dev clang-format libboost-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev python3.10 python3-pip python3.10-venv \
        hdf5-tools libhdf5-dev

# 设置工作目录
WORKDIR /app

# 复制本地 DiskANN 项目到容器（假设你本地已开发/修改）
COPY . /app/DiskANN

WORKDIR /app/DiskANN

RUN mkdir -p data

# 构建 C++ 可执行文件
RUN mkdir -p build && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -- -j

# 安装 Python 依赖（如有 requirements.txt）
RUN python3.10 -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then python3.10 -m pip install -r requirements.txt; fi

# 默认入口：可交互或直接运行脚本
# 本机数据存放位置 mnt/python/PCAdr/data/pubmed23/benchmark-dev-pubmed23.h5
CMD ["python3","RunAndEval.py"]
