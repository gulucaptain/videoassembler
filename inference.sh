CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=10086 NODE_RANK=0 WORLD_SIZE=1 HYDRA_FULL_ERROR=1 \
python main.py --config-name=_meta_/inference.yaml