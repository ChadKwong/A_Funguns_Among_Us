#!/usr/bin/env bash

#Retrieve spack from repository and load gcc 11.2.0
git clone --depth=100 --branch=releases/v0.20 https://github.com/spack/spack.git 
module load gcc/11.2.0 python/3.10.8-dkpz5k5
export SPACK_PYTHON=$(which python3)

#This line needs to be ran everytime to initiate spack
source $MPHOME/hpc_project/spack/share/spack/setup-env.sh

#Create environment
spack env create hpc_project packages.yaml
spack env activate hpc_project

#Install packages
spack install
pip install dask
pip install dask-ml