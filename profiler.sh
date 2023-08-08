#!/usr/bin/env zsh

# Activate spack environment to start python
echo 'Starting spack environment...'
source $MPHOME/hpc_project/spack/share/spack/setup-env.sh
spack env activate hpc_project
echo 'Environment loaded. Starting python script...'

# # Run script initially
# python3 main.py

# Run GEMM script through profiler
echo 'Running script through profiler...'
mprof run main.py

# Generate memory usage plot
mprof plot -s -o memory_usage_plot.jpeg

# Generate flame plot
mprof plot -f -o flame-graph.jpeg

