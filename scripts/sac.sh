CUDA_VISIBLE_DEVICES=0 python3 cdmc/train.py \
	--algorithm sac \
	--seed 0 \
	--train_context_file empty.json \
	--test_context_file empty.json