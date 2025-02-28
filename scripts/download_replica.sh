#!/bin/bash

# 安装基础工具
sudo apt update && sudo apt install -y unzip wget

# 主数据下载逻辑
mkdir -p data
cd data

# 下载Replica数据
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip

# 下载并处理剔除后的网格
cd Replica
wget https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip
unzip cull_replica_mesh.zip
rm cull_replica_mesh.zip
mv cull_replica_mesh gt_mesh_culled

# 移动ply文件到目标目录
mkdir -p gt_mesh
mv *.ply gt_mesh/

test