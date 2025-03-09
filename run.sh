#!/bin/bash

# 5 tasks Split-UT-Zappos
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py ut_zappos_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output/ut-5 --size 20 --epoch 10 --lr 0.02 --batchwise_prompt True --alpha 0.7  --beta 0.02 --dll_coeff 1.0 --ortho 0.000003 --ce 1.0 --rce 0.01 --pull_constraint_coeff_1 0.0001 --pull_constraint_coeff 0.7
# 10 tasks Split-UT-Zappos
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py ut_zappos_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output/ut-10 --size 20 --epoch 3 --lr 0.03 --batchwise_prompt True --alpha 0.4 --beta 0.03 --dll_coeff 0.5 --ortho 0.0000001 --num_tasks 10 --pull_constraint_coeff_1 0.1
# Split-Clothing
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py clothing16k --model vit_base_patch16_224 --batch-size 16 --data-path ./local_datasets/ --output_dir ./output/clothing16k --size 20 --epoch 25 --lr 0.03 --batchwise_prompt True --alpha 0.3 --beta 0.5 --dll_coeff 0.1 --ortho 0.0000001 --ce 1.0 --rce 0.006 --pull_constraint_coeff_1 0.3