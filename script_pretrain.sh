mkdir -p ./checkpoints/ntu60_xsub
# mkdir -p ./checkpoints/ntu60_xview
# mkdir -p ./checkpoints/ntu120_xsub
# mkdir -p ./checkpoints/ntu120_xset

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node=4 pretrain.py   \
  --lr 0.0005   --batch-size 512  \
  --checkpoint-path ./checkpoints/ntu60_xsub \
  --schedule 351  -