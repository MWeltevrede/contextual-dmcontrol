from cdmc.arguments import parse_args
import json
from pyvirtualdisplay import Display
import torch
from inspect import getsourcefile
from os.path import abspath
import os
import numpy as np
import copy

with Display() as disp:
	from env.wrappers import make_env
	args = parse_args()
	
	current_file_dir = os.path.dirname(abspath(getsourcefile(lambda:0)))
	all_colors = torch.load(f'{current_file_dir}/env/data/color_hard.pt')
	num_colors = len(all_colors)
	all_video_paths = [f'{current_file_dir}/env/data/video_hard/video{i}.mp4' for i in range(100)]
	num_video_paths = len(all_video_paths)

	# generate physics states by resetting the original DMC env
	physics_seeds = []
	num_physics_seeds = 20
	num_colors_and_videos_to_sample = 10

	np.random.seed(args.seed)
	random_color_indices = np.random.choice(np.arange(num_colors), size=num_colors_and_videos_to_sample, replace=False)
	train_colors = []
	test_colors = []
	for i in range(num_colors):
		if i in random_color_indices:
			train_colors.append(all_colors[i])
		else:
			test_colors.append(all_colors[i])
	random_video_indices = np.random.choice(np.arange(num_video_paths), size=num_colors_and_videos_to_sample, replace=False)
	train_video_paths = []
	test_video_paths = []
	for i in range(num_video_paths):
		if i in random_video_indices:
			train_video_paths.append(all_video_paths[i])
		else:
			test_video_paths.append(all_video_paths[i])

	colors = []
	video_paths = []
	for i in range(num_physics_seeds):
		current_seed = args.seed + i
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=current_seed,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			image_size=args.image_size,
		)
		env.reset()
		physics_seeds.append((current_seed, env.get_state().tolist()))

		color = train_colors[i%num_colors_and_videos_to_sample]
		colors.append(dict([(k, v.tolist()) for k,v in color.items()]))

		video_paths.append(train_video_paths[i%num_colors_and_videos_to_sample])

	with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts{num_physics_seeds}_{args.seed}.json', 'w') as file:
		json.dump({'physics_seeds': physics_seeds, 'colors': colors, 'video_paths': video_paths}, file)

	# the remaining colours and videos are used for the test set
	# physics states for the test set are just randomly sampled from the full distribution (indicated by giving an empty states list)
	with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/test_contexts{num_physics_seeds}_{args.seed}.json', 'w') as file:
		json.dump({'physics_seeds': num_physics_seeds, 'colors': [dict([(k, v.tolist()) for k,v in color_dict.items()]) for color_dict in test_colors], 'video_paths': test_video_paths}, file)
