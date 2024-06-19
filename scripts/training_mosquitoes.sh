#!/bin/bash

cat << "EOF"
  __  __ ____   _____ _____  
 |  \/  |  _ \ / ____|  __ \ 
 | \  / | |_) | |  __| |  | |
 | |\/| |  _ <| | |_ | |  | |
 | |  | | |_) | |__| | |__| |
 |_|  |_|____/ \_____|_____/ 
                             
                             
EOF
echo " --- Welcome to MBGd workflow ! ---"
echo " content creator: Isabelle Rodrigues Vaz de Melo "

# Load configurations from config.yaml
CONFIG_FILE="configs/config.yaml"

# Path to Local yq (linux)
YQ_LOCAL="./yq"

# Verifica se o yq local existe
if [ ! -f "$YQ_LOCAL" ]; then
    echo "yq not found. Downloading yq..."
    wget https://github.com/mikefarah/yq/releases/download/v4.35.2/yq_linux_amd64 -O yq
    chmod +x yq
fi

# Extract values from config.yaml usando yq local
OBJ=$($YQ_LOCAL e '.TRAINING.OBJECT' $CONFIG_FILE)
FOLDS=$($YQ_LOCAL e '.REGISTER_DATASETS.FOLDS' $CONFIG_FILE)
MAX_ITER=$($YQ_LOCAL e '.TRAINING.MAX_ITER' $CONFIG_FILE)
CUDA_DEVICE=$($YQ_LOCAL e '.TRAINING.CUDA_DEVICE // 0' $CONFIG_FILE)

# START MOSQUITOES WORKFLOW
echo "Starting training workflow for ${OBJ} detection, ${FOLDS} folds."

for ((fold=0; fold<FOLDS; fold++)); do
    data_train="mbg_train${fold}_${OBJ}"
    data_val="mbg_val${fold}_${OBJ}"

    #echo "Running train.py for fold $fold..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/train.py --config-file $CONFIG_FILE --data-train "$data_train" --data-val "$data_val"
    
    echo "Running test.py for fold $fold..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/test.py --config-file $CONFIG_FILE --data-train "$data_train" --data-val "$data_val"
done

# FINISH MOSQUITOES WORKFLOW
echo "Training workflow completed."


