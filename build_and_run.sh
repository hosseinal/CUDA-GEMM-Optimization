#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 01:00:00
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
echo -e "${GREEN}[2/4] Building project (Release mode, parallel)...${NC}"
cmake --build build --config Release --parallel

echo ""

# Step 3: Install the project
echo -e "${GREEN}[3/4] Installing project...${NC}"
cmake --install build

echo ""

# Step 4: Run the executables
echo -e "${GREEN}[4/4] Running GEMM profiling...${NC}"
echo ""

echo -e "${BLUE}Running FP32 GEMM profiling:${NC}"
echo "----------------------------------------"
./build/src/profile_cuda_gemm_fp32

echo ""
echo -e "${BLUE}Running FP16 GEMM profiling:${NC}"
echo "----------------------------------------"
./build/src/profile_cuda_gemm_fp16

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build and run completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
