import torch
import os
import numpy as np
import gym
import cdmc.utils as utils
import time
import json
from cdmc.arguments import parse_args
from cdmc.env.wrappers import make_env
from cdmc.algorithms.factory import make_agent
from cdmc.logger import Logger
from cdmc.video import VideoRecorder
from pyvirtualdisplay import Display
import wandb


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	print("before making envs")
	with open(args.train_context_file, 'r') as file:
		train_contexts = json.load(file)
	with open(args.test_context_file, 'r') as file:
		test_contexts = json.load(file)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		states=train_contexts['states'],
		video_paths=train_contexts['video_paths'],
		colors=[dict([(k, np.array(v)) for k,v in color_dict.items()]) for color_dict in train_contexts['colors']],
	)
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		states=test_contexts['states'],
		video_paths=test_contexts['video_paths'],
		colors=[dict([(k, np.array(v)) for k,v in color_dict.items()]) for color_dict in test_contexts['colors']],
	)
	print("after making envs")
	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, os.path.split(args.train_context_file)[-1][:-5], str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step == args.init_steps:
			for i in range(1, args.init_steps + 1):
				agent.update(replay_buffer, L, i)	
		elif step > args.init_steps:
			agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	with open("wandb_info.txt") as file:
		lines = [line.rstrip() for line in file]
		os.environ["WANDB_API_KEY"] = lines[0]
		os.environ["WANDB_START_METHOD"] = "thread"
		wandb_project = "ContextualDMC"
	
	args_dict = vars(args)
	with open(args.train_context_file, 'r') as file:
		contexts = json.load(file)
		args_dict['num_contexts'] = max(max(len(contexts['states']) if isinstance(contexts['states'], list) else 0, len(contexts['video_paths'])), len(contexts['colors']))
		train_context_name_short = os.path.split(args.train_context_file)[-1]
		args_dict['train_context_name_short'] = '_'.join(train_context_name_short.split('_')[:-1])
	with wandb.init(project=wandb_project, entity=lines[1], config=args_dict, tags=[args.algorithm, args.domain_name, args.task_name]):
		with Display() as disp:
			main(args)
