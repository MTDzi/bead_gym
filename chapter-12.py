from beads_gym.environment.beads_cart_pole_environment import BeadsCartPoleEnvironment
from beads_gym.environment.beads_quad_copter_environment import BeadsQuadCopterEnvironment


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import count
import moviepy.editor as mpy

import os.path
import tempfile
import random
import base64
import glob
import time
import json
import gym
import io
import os
import gc

from gym import wrappers

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56) #, 78, 90)


env = BeadsQuadCopterEnvironment()
env.reset()
plt.imshow(env.render())
plt.savefig("init_image.png")
plt.clf()


plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)


def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None, 
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env_name == "BeadsCartPoleEnvironment":
            env = BeadsCartPoleEnvironment()
            env.do_render = render
            env.do_record = record
            env.monitor_mode = monitor_mode
            return env
        if env_name == "BeadsQuadCopterEnvironment":
            env = BeadsQuadCopterEnvironment()
            env.do_render = render
            env.do_record = record
            env.monitor_mode = monitor_mode
            return env
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True, 
            mode=monitor_mode, 
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env
    return make_env_fn, kargs


def get_videos_html(env_videos, title, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm

def get_gif_html(video, title, video_id):
    video = np.array(video)
    num_frames = len(video)
    fps = 10

    # Create a VideoClip
    clip = mpy.VideoClip(
        # fun,
        lambda t: video[int(t * fps)],
        duration=num_frames / fps,
    )

    # Write the VideoClip to a file
    clip.write_videofile(f"{title}_{video_id}.mp4", fps=fps)


class RenderUint8(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def render(self, mode='rgb_array'):
        frame = self.env.render(mode=mode)
        return frame.astype(np.uint8)


class FCQV(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0: 
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, 
                             device=self.device, 
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class FCDP(nn.Module):
    def __init__(self, 
                 input_dim,
                 action_bounds,
                 hidden_dims=(32,32), 
                 activation_fc=F.selu,
                 out_activation_fc=F.tanh):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.act_min, self.act_max = action_bounds

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.act_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
        self.act_min = torch.tensor(self.act_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.act_max = torch.tensor(self.act_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = self.out_activation_fc(
            torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(
            torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.act_max - self.act_min) / \
                                    (self.nn_max - self.nn_min) + self.act_min

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        return self.rescale_fn(x)


class ReplayBuffer():
    def __init__(self, 
                 max_size=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size


class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)


class NormalNoiseStrategy():
    def __init__(self, bounds, exploration_noise_ratio=0.1, exploration_noise_amplitude=None, ou_process=False):
        self.low, self.high = bounds
        
        if exploration_noise_ratio is None: assert exploration_noise_amplitude is not None
        if exploration_noise_amplitude is None: assert exploration_noise_ratio is not None
        self.exploration_noise_ratio = exploration_noise_ratio
        self.exploration_noise_amplitude = exploration_noise_amplitude
        self.ou_process = ou_process
        self.prev_noise = np.zeros(len(self.high))
        self.ratio_noise_injected = 0
        
    def select_action(self, model, state, max_exploration=False):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            
        noise = self.prev_noise if self.ou_process else np.zeros(len(self.high))
        if max_exploration:
            noise += np.random.normal(loc=0, scale=self.high, size=len(self.high))
        else:
            if self.exploration_noise_ratio is not None:
                noise += np.random.normal(loc=0, scale=1, size=len(self.high)) * np.abs(greedy_action) * self.exploration_noise_ratio
            elif self.exploration_noise_amplitude is not None:
                noise += np.random.normal(loc=0, scale=self.exploration_noise_amplitude, size=len(self.high))
            else:
                raise ValueError("No exploration noise specified")

        noisy_action = greedy_action + noise
        self.prev_noise = noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action
    
    
class NormalNoiseDecayStrategy():
    def __init__(
        self,
        bounds,
        init_noise_ratio_mult=0.5, min_noise_ratio_mult=0.1,
        init_noise_ratio_add=0.5, min_noise_ratio_add=0.1,
        decay_steps=10000,
    ):
        self.t = 0
        self.low, self.high = bounds
        self.noise_ratio_mult = init_noise_ratio_mult
        self.init_noise_ratio_mult = init_noise_ratio_mult
        self.min_noise_ratio_mult = min_noise_ratio_mult
        self.noise_ratio_add = init_noise_ratio_add
        self.init_noise_ratio_add = init_noise_ratio_add
        self.min_noise_ratio_add = min_noise_ratio_add
        self.decay_steps = decay_steps
        self.ratio_noise_injected = 0

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio_mult = (self.init_noise_ratio_mult - self.min_noise_ratio_mult) * noise_ratio + self.min_noise_ratio_mult
        self.noise_ratio_mult = np.clip(noise_ratio_mult, self.min_noise_ratio_mult, self.init_noise_ratio_mult)
        noise_ratio_add = (self.init_noise_ratio_add - self.min_noise_ratio_add) * noise_ratio + self.min_noise_ratio_add
        self.noise_ratio_add = np.clip(noise_ratio_add, self.min_noise_ratio_add, self.init_noise_ratio_add)
        self.t += 1

    def select_action(self, model, state, max_exploration=False):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            
        noise = np.zeros(len(self.high))
        if max_exploration:
            noise += np.random.normal(loc=0, scale=self.high, size=len(self.high))
        else:
            mult_noise_scale = np.abs(greedy_action) * self.noise_ratio_mult
            noise += np.random.normal(loc=0, scale=mult_noise_scale, size=len(self.high))
            noise += np.random.normal(loc=0, scale=self.noise_ratio_add, size=len(self.high))

        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self._noise_ratio_update()
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action


class DDPG():
    def __init__(self, 
                 replay_buffer_fn,
                 policy_model_fn, 
                 policy_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau):
        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn

        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), 
                                       self.value_max_grad_norm)
        self.value_optimizer.step()

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), 
                                       self.policy_max_grad_norm)        
        self.policy_optimizer.step()

    def interaction_step(self, state, env, state_noise=None):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        noisy_state = state if state_noise is None else state + state_noise
        action = self.training_strategy.select_action(self.online_policy_model, 
                                                      noisy_state, 
                                                      len(self.replay_buffer) < min_samples)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
        return new_state, is_terminal
    
    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model.parameters(), 
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(), 
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        
        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, 
                                                       self.value_optimizer_lr)        
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, 
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = training_strategy_fn(action_bounds)
        self.evaluation_strategy = evaluation_strategy_fn(action_bounds)
                    
        result = np.empty((max_episodes, 6))
        result[:] = np.nan
        training_time = 0
        state_noise_scale = 0.05
        reasonable_bound = 0.1
        how_often_within_reasonable_bounds = (np.linalg.norm(np.random.normal(loc=0, scale=state_noise_scale, size=(10000, 3)), axis=1) < reasonable_bound).mean()
        print(
            f"With the scale of {state_noise_scale:.2f}, the noise vector will be "
            f"{100 * how_often_within_reasonable_bounds:.2f}% of the time within a ball "
            f"of radius: {reasonable_bound}"
        )
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                if np.random.uniform() > 1.0:
                    state_noise = np.random.normal(loc=0, scale=state_noise_scale, size=len(state))
                else:
                    state_noise = None
                state, is_terminal = self.interaction_step(state, env, state_noise)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break
            
            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.save_checkpoint(episode-1, self.online_policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, mean_100_exp_rat, training_time, wallclock_elapsed
            
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = (
                f'el {elapsed_str}, ep {episode-1:04}, ts {total_step:07}, '
                f'mean_10_reward: {mean_10_reward:.2f}\u00B1{std_10_reward:.2f}, '
                f'mean_100_reward: {mean_100_reward:.2f}\u00B1{std_100_reward:.2f}, '
                f'mean_100_exp_rat: {mean_100_exp_rat:.2f}\u00B1{std_100_exp_rat:.2f}, '
                f'mean_100_eval_score: {mean_100_eval_score:.2f}\u00B1{std_100_eval_score:.2f}'
            )
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}min training time,'
              ' {:.2f}min wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time / 60, wallclock_time / 60))
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if eval_env.do_render: eval_env.render()
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=2, max_n_videos=2):

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        for i in range(n_episodes):
            env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)
            self.evaluate(self.online_policy_model, env, n_episodes=1)
            env.close()
            get_gif_html(
                env.videos[0], 
                title.format(self.__class__.__name__),
                i,
            )
        del env

    def demo_progression(self, title='{} Agent progression', max_n_videos=4):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.online_policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in SEEDS:
    print(f'BEGIN seed: {seed}')
    environment_settings = {
        # 'env_name': 'BeadsCartPole',
        'env_name': 'BeadsQuadCopterEnvironment',
        'gamma': 0.99, # 0.99,
        'max_minutes': 360,
        'max_episodes': 1500,
        'goal_mean_100_reward': 140000.0,
    }

    policy_model_fn = lambda nS, bounds: FCDP(nS, bounds, hidden_dims=(300,300))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005

    value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(300,300))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    # training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1, exploration_noise_amplitude=1.0)
    training_strategy_fn = lambda bounds: NormalNoiseDecayStrategy(bounds,
                                                                   init_noise_ratio_mult=0.1,
                                                                   min_noise_ratio_mult=0.01,
                                                                   init_noise_ratio_add=1.5,
                                                                   min_noise_ratio_add=0.01,
                                                                   decay_steps=1_000_000)
    # training_strategy_fn = lambda bounds: NormalNoiseStrategy(
    #     bounds,
    #     exploration_noise_ratio=0.1,
    #     exploration_noise_amplitude=0.2,
    #     ou_process=True,    
    # )
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    replay_buffer_fn = lambda: ReplayBuffer(max_size=1_000_000, batch_size=256)
    n_warmup_batches = 5
    update_target_every_steps = 2
    tau = 0.005
    
    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()

    agent = DDPG(replay_buffer_fn,
                 policy_model_fn, 
                 policy_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    ddpg_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        print('best_agent assigned!\n')
        best_agent = agent
    print(f'END seed: {seed}')
        
ddpg_results = np.array(ddpg_results)
_ = BEEP()

del best_agent.replay_buffer
print(gc.collect())

ddpg_max_t, ddpg_max_r, ddpg_max_s, ddpg_max_exp, \
ddpg_max_sec, ddpg_max_rt = np.max(ddpg_results, axis=0).T
ddpg_min_t, ddpg_min_r, ddpg_min_s, ddpg_min_exp, \
ddpg_min_sec, ddpg_min_rt = np.min(ddpg_results, axis=0).T
ddpg_mean_t, ddpg_mean_r, ddpg_mean_s, ddpg_mean_exp, \
ddpg_mean_sec, ddpg_mean_rt = np.mean(ddpg_results, axis=0).T
ddpg_x = np.arange(len(ddpg_mean_s))

fig, axs = plt.subplots(3, 1, figsize=(15,10), sharey=False, sharex=True)

# DDPG
axs[0].plot(ddpg_max_r, 'r', linewidth=1)
axs[0].plot(ddpg_min_r, 'r', linewidth=1)
axs[0].plot(ddpg_mean_r, 'r:', label='DDPG', linewidth=2)
axs[0].fill_between(
    ddpg_x, ddpg_min_r, ddpg_max_r, facecolor='r', alpha=0.3)

axs[1].plot(ddpg_max_s, 'r', linewidth=1)
axs[1].plot(ddpg_min_s, 'r', linewidth=1)
axs[1].plot(ddpg_mean_s, 'r:', label='DDPG', linewidth=2)
axs[1].fill_between(
    ddpg_x, ddpg_min_s, ddpg_max_s, facecolor='r', alpha=0.3)

axs[2].plot(ddpg_max_exp, 'r', linewidth=1)
axs[2].plot(ddpg_min_exp, 'r', linewidth=1)
axs[2].plot(ddpg_mean_exp, 'r:', label='DDPG', linewidth=2)
axs[2].fill_between(
    ddpg_x, ddpg_min_exp, ddpg_max_exp, facecolor='r', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
axs[2].set_title('Moving Noise')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.savefig("progress.png")


del ddpg_results
print(gc.collect())

best_agent.demo_last(title="last")
print('done')
# best_agent.demo_progression(title="progress")


fig, axs = plt.subplots(3, 1, figsize=(15,15), sharey=False, sharex=True)

# DDPG
axs[0].plot(ddpg_max_t, 'r', linewidth=1)
axs[0].plot(ddpg_min_t, 'r', linewidth=1)
axs[0].plot(ddpg_mean_t, 'r:', label='DDPG', linewidth=2)
axs[0].fill_between(
    ddpg_x, ddpg_min_t, ddpg_max_t, facecolor='r', alpha=0.3)

axs[1].plot(ddpg_max_sec, 'r', linewidth=1)
axs[1].plot(ddpg_min_sec, 'r', linewidth=1)
axs[1].plot(ddpg_mean_sec, 'r:', label='DDPG', linewidth=2)
axs[1].fill_between(
    ddpg_x, ddpg_min_sec, ddpg_max_sec, facecolor='r', alpha=0.3)

axs[2].plot(ddpg_max_rt, 'r', linewidth=1)
axs[2].plot(ddpg_min_rt, 'r', linewidth=1)
axs[2].plot(ddpg_mean_rt, 'r:', label='DDPG', linewidth=2)
axs[2].fill_between(
    ddpg_x, ddpg_min_rt, ddpg_max_rt, facecolor='r', alpha=0.3)

# ALL
axs[0].set_title('Total Steps')
axs[1].set_title('Training Time')
axs[2].set_title('Wall-clock Time')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()

ddpg_root_dir = os.path.join(RESULTS_DIR, 'ddpg')
not os.path.exists(ddpg_root_dir) and os.makedirs(ddpg_root_dir)

np.save(os.path.join(ddpg_root_dir, 'x'), ddpg_x)

np.save(os.path.join(ddpg_root_dir, 'max_r'), ddpg_max_r)
np.save(os.path.join(ddpg_root_dir, 'min_r'), ddpg_min_r)
np.save(os.path.join(ddpg_root_dir, 'mean_r'), ddpg_mean_r)

np.save(os.path.join(ddpg_root_dir, 'max_s'), ddpg_max_s)
np.save(os.path.join(ddpg_root_dir, 'min_s'), ddpg_min_s )
np.save(os.path.join(ddpg_root_dir, 'mean_s'), ddpg_mean_s)

np.save(os.path.join(ddpg_root_dir, 'max_t'), ddpg_max_t)
np.save(os.path.join(ddpg_root_dir, 'min_t'), ddpg_min_t)
np.save(os.path.join(ddpg_root_dir, 'mean_t'), ddpg_mean_t)

np.save(os.path.join(ddpg_root_dir, 'max_sec'), ddpg_max_sec)
np.save(os.path.join(ddpg_root_dir, 'min_sec'), ddpg_min_sec)
np.save(os.path.join(ddpg_root_dir, 'mean_sec'), ddpg_mean_sec)

np.save(os.path.join(ddpg_root_dir, 'max_rt'), ddpg_max_rt)
np.save(os.path.join(ddpg_root_dir, 'min_rt'), ddpg_min_rt)
np.save(os.path.join(ddpg_root_dir, 'mean_rt'), ddpg_mean_rt)
