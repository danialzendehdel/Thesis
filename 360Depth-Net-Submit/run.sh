#!/bin/bash

# Dataset: Rec_Street
# images size: torch.Size([8, 4, 256, 256]) torch.Size([8, 4, 256, 256]) torch.Size([8, 64, 256])
python main.py --datapath REC_STREET[16-06-2022]-16byte/ \
               --datapath2 new_street/ \
               --save_checkpoint checkpoints/test_2004/ \
               --tensorboard_path Logs_all/test_2004 \
               --batch 8 \
               --Dours \
               --epochs 500 \
               --save_path save_path/test_2004/
