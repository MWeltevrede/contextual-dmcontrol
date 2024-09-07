from cdmc.arguments  import parse_args
import json
from pyvirtualdisplay import Display
import torch
import random

with Display() as disp:
	from env.wrappers import make_env
	args = parse_args()
	env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			image_size=args.image_size,
			states=[],
			video_paths=[],
			colors=[]
		)
	
	all_colors = torch.load(f'cdmc/env/data/color_hard.pt')
	num_colors = len(all_colors)
	all_video_paths = [f'cdmc/env/data/video_hard/video{i}.mp4' for i in range(100)]
	num_video_paths = len(all_video_paths)

	# generate physics states by resetting the original DMC env
	physics_states = []
	num_physics_states = 10
	colors = []
	video_paths = []
	for i in range(num_physics_states):
		env.reset()
		physics_states.append(env.get_state().tolist())

		# sample random color
		rand_color_id = random.randint(0, num_colors-i-1)
		color = all_colors[rand_color_id]
		colors.append(dict([(k, v.tolist()) for k,v in color.items()]))
		del all_colors[rand_color_id]

		# sample random video path
		rand_video_id = random.randint(0, num_video_paths-i-1)
		video_paths.append(all_video_paths[rand_video_id])
		del all_video_paths[rand_video_id]

	with open(f'{args.domain_name}_{args.task_name}/train_contexts{num_physics_states}.json', 'w') as file:
		json.dump({'states': physics_states, 'colors': colors, 'video_paths': video_paths}, file)

	# the remaining colours and videos are used for the test set
	# physics states for the test set are just randomly sampled from the full distribution (indicated by giving an empty states list)
	with open(f'{args.domain_name}_{args.task_name}/test_contexts{num_physics_states}.json', 'w') as file:
		json.dump({'states': [], 'colors': [dict([(k, v.tolist()) for k,v in color_dict.items()]) for color_dict in all_colors], 'video_paths': all_video_paths}, file)