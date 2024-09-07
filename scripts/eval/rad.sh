CUDA_VISIBLE_DEVICES=0 python3 cdmc/eval.py \
    --algorithm rad \
	--eval_episodes 100 \
	--seed 0 \
	--train_context_file empty.json \
	--test_context_file empty.json