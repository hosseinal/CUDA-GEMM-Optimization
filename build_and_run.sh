#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 05:00:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf

# CUDA GEMM Optimization - Build and Run Script
# This script builds the project and runs both FP32 and FP16 GEMM profiling

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CUDA GEMM Optimization - Build and Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Configure with CMake
echo -e "${GREEN}[1/4] Configuring project with CMake...${NC}"
cmake -B build

echo ""

# Step 2: Build the project
echo -e "${GREEN}[2/3] Building project (Release mode, parallel)...${NC}"
cmake --build build --config Release --parallel

echo ""

# Step 3: Run the executables
echo -e "${GREEN}[3/3] Running GEMM profiling...${NC}"
echo ""

echo -e "${BLUE}Running FP32 GEMM profiling:${NC}"
echo "----------------------------------------"
echo "FP 32" > ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 512 512 32" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 512 512 32 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 512 512 64" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 512 512 64 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 512 512 128" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 512 512 128 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 1024 256 32" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 1024 256 32 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 1024 256 64" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 1024 256 64 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 1024 256 128" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 1024 256 128 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 2048 2048 32" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 2048 256 32 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 2048 2048 64" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 2048 256 64 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt
echo "Matrix Size: 2048 2048 128" >> ./profile_cuda_gemm_fp32_results.txt
./build/src/profile_cuda_gemm_fp32 2048 256 128 >> ./profile_cuda_gemm_fp32_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp32_results.txt

echo -e "${BLUE}Running FP16 GEMM profiling:${NC}"
echo "----------------------------------------"
echo "FP 16" > ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 512 512 32" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 512 512 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 512 512 64" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 512 512 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 512 512 128" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 512 512 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 1024 256 32" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 1024 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 1024 256 64" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 1024 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 1024 256 128" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 1024 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 2048 256 32" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 2048 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 2048 256 64" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 2048 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo "--------------------------------------------------------------------------------">> ./profile_cuda_gemm_fp16_results.txt
echo "Matrix Size: 2048 256 128" >> ./profile_cuda_gemm_fp16_results.txt
./build/src/profile_cuda_gemm_fp16 2048 256 128 >> ./profile_cuda_gemm_fp16_results.txt
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build and run completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
