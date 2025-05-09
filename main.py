import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os.path

import os
import gc

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from beads_gym.rl.agents import DDPG
from beads_gym.rl.policies import FCDP


LEAVE_PRINT_EVERY_N_SECS = 300
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')


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


class FCQV(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_dims=(32, 32), 
        activation_fc=F.relu,
    ):
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


class ReplayBuffer:
    def __init__(
        self, 
        max_size=10000, 
        batch_size=64,
    ):
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


@hydra.main(config_path="./configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    # Let's first visualize the beads env used in this experiment
    env = instantiate(cfg.env)
    env.reset()
    plt.imshow(env.render())
    plt.savefig("init_image.png")
    plt.clf()
    
    ddpg_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in cfg.seeds:
        print(f'BEGIN seed: {seed}')

        policy_model_fn = partial(FCDP, hidden_dims=(300, 300))
        policy_max_grad_norm = float('inf')
        policy_optimizer_fn = partial(optim.Adam, lr=0.0005)

        value_model_fn = partial(FCQV, hidden_dims=(300, 300))
        value_max_grad_norm = float('inf')
        value_optimizer_fn = partial(optim.Adam, lr=0.0005)

        # training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1, exploration_noise_amplitude=1.0)
        # training_strategy_fn = partial(
        #     NormalNoiseDecayStrategy,
        #     init_noise_ratio_mult=0.1,
        #     min_noise_ratio_mult=0.01,
        #     init_noise_ratio_add=1.5,
        #     min_noise_ratio_add=0.01,
        #     decay_steps=1_000_000,
        # )
        training_strategy_fn = partial(instantiate, cfg.train_strategy)
        # training_strategy_fn = lambda bounds: NormalNoiseStrategy(
        #     bounds,
        #     exploration_noise_ratio=0.1,
        #     exploration_noise_amplitude=0.2,
        #     ou_process=True,    
        # )
        evaluation_strategy_fn = partial(instantiate, cfg.eval_strategy)

        replay_buffer_fn = partial(ReplayBuffer, max_size=1_000_000, batch_size=256)
        
        agent = DDPG(
            replay_buffer_fn=replay_buffer_fn,
            policy_model_fn=policy_model_fn, 
            policy_max_grad_norm=policy_max_grad_norm, 
            policy_optimizer_fn=policy_optimizer_fn, 
            value_model_fn=value_model_fn, 
            value_max_grad_norm=value_max_grad_norm, 
            value_optimizer_fn=value_optimizer_fn, 
            training_strategy_fn=training_strategy_fn,
            evaluation_strategy_fn=evaluation_strategy_fn,
            n_warmup_batches=cfg.agent.n_warmup_batches,
            update_target_every_steps=cfg.agent.update_target_every_steps,
            tau=cfg.agent.tau,
            leave_print_every_n_secs=cfg.agent.leave_print_every_n_secs,
        )

        make_env_fn = partial(instantiate, cfg.env)
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn=make_env_fn,
            seed=seed,
            gamma=cfg.agent.train.gamma,
            max_minutes=cfg.agent.train.max_minutes,
            max_episodes=cfg.agent.train.max_episodes,
            goal_mean_100_reward=cfg.agent.train.goal_mean_100_reward,
        )

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

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharey=False, sharex=True)

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

    best_agent.demo_last(title="last")
    print('done')

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


if __name__ == '__main__':
    main()