CUDA_VISIBLE_DEVICES=0 python3 cdmc/train.py \
	--algorithm curl \
	--aux_update_freq 1 \
	--seed 0 \
	--train_context_file empty.json \
	--test_context_file empty.json