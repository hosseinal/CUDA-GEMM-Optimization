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
echo -e "${GREEN}[2/3] Building project (Release mode, parallel)...${NC}"
cmake --build build --config Release --parallel

echo ""

# Step 3: Run the executables over requested parameter sweep
echo -e "${GREEN}[3/3] Running GEMM profiling parameter sweep...${NC}"
echo ""

# Sizes to test: (M K N) tuples
SIZES=("512 512 512" "1024 256 1024" "2048 256 2048")

# Block columns to test
BCOLS=(32 64 128)

# Create logs directory
LOG_DIR=logs
mkdir -p ${LOG_DIR}

FP32_BIN=./build/src/profile_cuda_gemm_fp32
FP16_BIN=./build/src/profile_cuda_gemm_fp16

if [ ! -x "${FP32_BIN}" ] || [ ! -x "${FP16_BIN}" ]; then
	echo -e "${RED}One or both profile executables not found or not executable:${NC}"
	echo "  Expected: ${FP32_BIN} and ${FP16_BIN}"
	exit 1
fi

for size in "${SIZES[@]}"; do
	read -r M K N <<< "${size}"
	for b in "${BCOLS[@]}"; do
		echo -e "${BLUE}Running FP32: M=${M} K=${K} N=${N} bCols=${b}${NC}"
		echo "----------------------------------------"
		LOGFILE=${LOG_DIR}/fp32_M${M}_K${K}_N${N}_b${b}.log
		# Pass bCols as a fourth argument (program currently ignores extra args unless implemented)
		${FP32_BIN} ${M} ${K} ${N} ${b} | tee "${LOGFILE}"

		echo -e "${BLUE}Running FP16: M=${M} K=${K} N=${N} bCols=${b}${NC}"
		echo "----------------------------------------"
		LOGFILE=${LOG_DIR}/fp16_M${M}_K${K}_N${N}_b${b}.log
		${FP16_BIN} ${M} ${K} ${N} ${b} | tee "${LOGFILE}"

		echo ""
	done
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build and run completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
