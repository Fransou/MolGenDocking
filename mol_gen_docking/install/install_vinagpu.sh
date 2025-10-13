#!/bin/bash

# Install QuickVina2-GPU-2.1
cd ..
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git

# Compile all 3 versions and modify each Makefile to reflect the correct paths
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
WORKSPACE=$HOME/MolGenDocking/external_repositories
cd Vina-GPU-2.1
for dir in QuickVina2-GPU-2.1 QuickVina-W-GPU-2.1 AutoDock-Vina-GPU-2.1; do
    cd $dir

    # 1. Ensure WORK_DIR points to the current directory
    sed -i "s|^WORK_DIR=.*|WORK_DIR=$(pwd)|" Makefile

    # 2. Ensure Boost and CUDA paths are set correctly
    sed -i "s|^BOOST_LIB_PATH=.*|BOOST_LIB_PATH=$BOOST_ROOT/lib|" Makefile
    sed -i "s|^OPENCL_LIB_PATH=.*|OPENCL_LIB_PATH=$CUDA_PATH|" Makefile

    # 3. Remove Boost thread source references
    sed -i "s|SRC=.*thread.cpp .*once.cpp|SRC=./lib/*.cpp ./OpenCL/src/wrapcl.cpp|" Makefile

    # 4. Add -lboost_thread to LIB1
    sed -i "s|^LIB1=.*|LIB1=-lboost_thread -lboost_program_options -lboost_system -lboost_filesystem -lOpenCL|" Makefile

    make clean
    make source
    cd ..
done
