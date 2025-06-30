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
	from cdmc.env.wrappers import make_env
	args = parse_args()
	
	current_file_dir = os.path.dirname(abspath(getsourcefile(lambda:0)))
	all_colors = torch.load(f'{current_file_dir}/env/data/color_hard.pt')
	num_colors = len(all_colors)
	# all_video_paths = [f'{current_file_dir}/env/data/video_hard/video{i}.mp4' for i in range(100)]
	# num_video_paths = len(all_video_paths)

	# generate physics states by resetting the original DMC env
	physics_seeds = []
	num_physics_seeds = 5
	num_colors_and_videos_to_sample = 5

	np.random.seed(args.seed)
	random_color_indices = np.random.choice(np.arange(num_colors), size=num_colors_and_videos_to_sample, replace=False)
	train_colors = []
	test_colors = []
	for i in range(num_colors):
		if i in random_color_indices:
			train_colors.append(all_colors[i])
		else:
			test_colors.append(all_colors[i])
	# random_video_indices = np.random.choice(np.arange(num_video_paths), size=num_colors_and_videos_to_sample, replace=False)
	# train_video_paths = []
	test_video_paths = []
	# for i in range(num_video_paths):
	# 	if i in random_video_indices:
	# 		train_video_paths.append(all_video_paths[i])
	# 	else:
	# 		test_video_paths.append(all_video_paths[i])

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

		# video_paths.append(train_video_paths[i%num_colors_and_videos_to_sample])

	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts_base{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': physics_seeds, 'colors': colors, 'video_paths': video_paths}, file)

	with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts_dmc{num_physics_seeds}_{args.seed}.json', 'w') as file:
		json.dump({'states': physics_seeds, 'colors': [], 'video_paths': []}, file)

	# # the remaining colours and videos are used for the test set
	# # physics states for the test set are just randomly sampled from the full distribution (indicated by giving an empty states list)
	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/test_contexts_base{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': args.seed + num_physics_seeds, 'colors': [dict([(k, v.tolist()) for k,v in color_dict.items()]) for color_dict in test_colors], 'video_paths': test_video_paths}, file)

	# #### Generate unreachable tasks
	# unreachable_physics_seeds = copy.deepcopy(physics_seeds)
	# unreachable_colors = copy.deepcopy(colors)
	# unreachable_video_paths = copy.deepcopy(video_paths)

	# # Sample 40 additional colors and videos
	# unreachable_random_color_indices = np.random.choice(np.arange(len(test_colors)), size=4*num_colors_and_videos_to_sample, replace=False)
	# additional_train_colors = []
	# unreachable_test_colors = []
	# for i in range(len(test_colors)):
	# 	if i in unreachable_random_color_indices:
	# 		additional_train_colors.append(test_colors[i])
	# 	else:
	# 		unreachable_test_colors.append(test_colors[i])
	# unreachable_random_video_indices = np.random.choice(np.arange(len(test_video_paths)), size=4*num_colors_and_videos_to_sample, replace=False)
	# additional_train_video_paths = []
	# unreachable_test_video_paths = []
	# for i in range(len(test_video_paths)):
	# 	if i in unreachable_random_video_indices:
	# 		additional_train_video_paths.append(test_video_paths[i])
	# 	else:
	# 		unreachable_test_video_paths.append(test_video_paths[i])

	# for i in range(4*num_physics_seeds):
	# 	# env.reset()
	# 	# unreachable_physics_states.append(env.get_state().tolist())\
	# 	current_seed = args.seed + num_physics_seeds + i
	# 	env = make_env(
	# 		domain_name=args.domain_name,
	# 		task_name=args.task_name,
	# 		seed=current_seed,
	# 		episode_length=args.episode_length,
	# 		action_repeat=args.action_repeat,
	# 		image_size=args.image_size,
	# 	)
	# 	env.reset()
	# 	unreachable_physics_seeds.append((current_seed, env.get_state().tolist()))
	# 	# unreachable_physics_seeds.append(args.seed + num_physics_seeds + i)

	# 	color = additional_train_colors[i%(4*num_colors_and_videos_to_sample)]
	# 	unreachable_colors.append(dict([(k, v.tolist()) for k,v in color.items()]))

	# 	unreachable_video_paths.append(additional_train_video_paths[i%(4*num_colors_and_videos_to_sample)])


	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts_unreachable{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': unreachable_physics_seeds, 'colors': unreachable_colors, 'video_paths': unreachable_video_paths}, file)

	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/test_contexts_unreachable{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': num_physics_seeds + 4*num_physics_seeds, 'colors': [dict([(k, v.tolist()) for k,v in color_dict.items()]) for color_dict in unreachable_test_colors], 'video_paths': unreachable_test_video_paths}, file)

	# #### Generate random expl
	# random_physics_seeds = copy.deepcopy(physics_seeds)
	# random_colors = copy.deepcopy(colors)
	# random_video_paths = copy.deepcopy(video_paths)

	# # for each base context add 4 new reachable contexts 
	# for i in range(20):
	# 	for j in range(num_physics_seeds):
	# 		current_seed = args.seed + j
	# 		env = make_env(
	# 			domain_name=args.domain_name,
	# 			task_name=args.task_name,
	# 			seed=current_seed,
	# 			episode_length=args.episode_length,
	# 			action_repeat=args.action_repeat,
	# 			image_size=args.image_size,
	# 		)
	# 		env.reset()
	# 		# 100 random steps
	# 		np.random.seed(args.seed + (i+1)*num_physics_seeds + j)
	# 		for _ in range(100):
	# 			action = (np.random.rand(*env.action_space.shape) * (env.action_space.high - env.action_space.low)) + env.action_space.low
	# 			action = action.astype(np.float32)
	# 			_ = env.step(action)
	# 		random_physics_seeds.append((current_seed, env.get_state().tolist()))


	# 		# combined with the same colors and backgrounds from the base contexts
	# 		random_colors.append(colors[j%num_colors_and_videos_to_sample])
	# 		# random_video_paths.append(video_paths[j%num_colors_and_videos_to_sample])

	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts_random{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': random_physics_seeds, 'colors': random_colors, 'video_paths': random_video_paths}, file)
	
	# with open(f'{current_file_dir}/../{args.domain_name}_{args.task_name}/train_contexts_dmc_random{num_physics_seeds}_{args.seed}.json', 'w') as file:
	# 	json.dump({'states': random_physics_seeds, 'colors': [], 'video_paths': []}, file)
