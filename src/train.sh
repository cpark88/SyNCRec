torchrun --nproc_per_node=8 --master_port=1234 run_pretrain.py \
--batch_size=2 \
--hidden_size=512 \
--num_hidden_layers=2 \
--num_attention_head=1 \
--epoch=1000 \
--max_seq_length=50 \
--data_name='amazon' \
--cross_expert_ratio=0.8 \
--strd_ym='202312' \
--cross_detach='y' \
--single_detach='y' \
--expert_layer='transformer' \
--lrb='y' \
--mip='y' \
--expert_num=5 \
--task_num=6 \
--lr=0.001



