#!/bin/bash
source /home/yshen/.bashrc
source /home/yshen/anaconda3/etc/profile.d/conda.sh
source /home/wangzhe/TriH-ANNS/conda/net.sh

export RAFT_ROOT=/home/wangzhe/TriH-ANNS/raft_install
export CUDACXX=/home/wangzhe/.conda/envs/TriH/bin/nvcc

conda activate TriH
cmake --version
cmake "${@}"