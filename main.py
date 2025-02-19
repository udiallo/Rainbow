# -*- coding: utf-8 -*-

from obstacle_tower_env import ObstacleTowerEnv
import argparse
from datetime import datetime
import numpy as np
import torch

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
from tqdm import tqdm

import monodepth.monodepth_inference as depth_estimation
from obstacle_tower_od import ObjectDetection
import prepare_input
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('environment_filename', default='../obstacle-tower-challenge/ObstacleTower/obstacletower.x86_64', nargs='?')
parser.add_argument('--docker_training', action='store_true')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')


# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Environment
env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, retro=False, realtime_mode=True, timeout_wait=700)
env.seed(2)
env.floor(2)
#env = Env(args)
env.train()
action_space = env.action_space()

action_dict =	{
  #1 : [0, 0, 0, 0], #nothing
  0 : [1, 0, 0, 0], # forward
  1 : [0, 0, 0, 1], # right
  2 : [0, 0, 0, 2], # left
  3 : [1, 0, 1, 0], # forward jump
  4 : [0, 0, 1, 1], # right jump
  5 : [0, 0, 1, 2], # left jump
  6 : [0, 1, 0, 0], # camera cc
  7 : [0, 2, 0, 0], # camera c
}



#action_space = []  # add function action_space() to ObstacleTowerEnv

# Agent
dqn = Agent(args, env)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct placeholder for input values of object detection and past actions
y = torch.Tensor(1, 16)  # placeholder

# Object detection and monodepth
OD = ObjectDetection()
depth_model = depth_estimation.Monodepth_inference('monodepth/model/monodepth-30000')

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 500, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False



  states_list = []
  objects_list = []

  if done == False:

    random_action = np.random.randint(0, 8)
    next_state, rgb, _, done = env.step(action_dict[random_action])
    objects, depthmap = prepare_input.prepare_input(next_state[0], depth_model, OD)
    state = cv2.resize(depthmap, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)

    states_list.append(state)
    states_list.append(state)
    states_list.append(state)
    states_list.append(state)
    objects_list.append(objects)
    objects_list.append(objects)
    objects_list.append(objects)
    objects_list.append(objects)

    states_list = np.asarray(states_list)
    objects_list = np.asarray(objects_list)
    objects_list = np.concatenate([objects_list[0], objects_list[1], objects_list[2], objects_list[3]], axis=None)

  state = torch.tensor(states_list, dtype=torch.float32).div_(255)

  y = torch.Tensor(objects_list)
  y = y.unsqueeze(0)

  val_mem.append(state, None, y, None, done)

  #state = next_state
  T += 1



if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop

  # after reseting hard code 10 steps to the front
  # from that position reset every 100 setps, it should learn to find the door
  # make sure that we are in the same room -> use same seed




  dqn.train()
  T, done = 0, True
  y = torch.Tensor(1, 16)
  for T in tqdm(range(args.T_max)):
    if done:
      state, done = env.reset(), True
      state = state[0][0]
      objects, depthmap = prepare_input.prepare_input(state, depth_model, OD)
      state = cv2.resize(depthmap, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
      start_states = [state, state, state, state]
      state = np.asarray(start_states)
      state = torch.tensor(state, dtype=torch.float32).div_(255)

    states_list = []
    objects_list = []

    if done == False:


      objects, depthmap = prepare_input.prepare_input(state[0], depth_model, OD)
      state_temp = cv2.resize(depthmap, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)

      states_list.append(state_temp)
      states_list.append(state_temp)
      states_list.append(state_temp)
      states_list.append(state_temp)
      objects_list.append(objects)
      objects_list.append(objects)
      objects_list.append(objects)
      objects_list.append(objects)

      states_list = np.asarray(states_list)
      objects_list = np.asarray(objects_list)
      objects_list = np.concatenate([objects_list[0], objects_list[1],objects_list[2],objects_list[3]], axis=None)

      state = torch.tensor(states_list, dtype=torch.float32).div_(255)


      y = torch.Tensor(objects_list)
      y = y.unsqueeze(0)

    done = False


    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state, y)  # Choose an action greedily (with noisy weights)

    act_vector = action_dict[action]

    next_state, rgb, reward, done = env.step(act_vector)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, y,  reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

    state = next_state
    done = False

env.close()
