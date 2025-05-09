import tempfile
import random
import glob
import time
from itertools import count
import gc
import os

import moviepy.editor as mpy

import numpy as np
import torch

ERASE_LINE = '\x1b[2K'


def get_gif_html(video, title, video_id):
    video = np.array(video)
    num_frames = len(video)
    fps = 10

    # Create a VideoClip
    clip = mpy.VideoClip(
        lambda t: video[int(t * fps)],
        duration=num_frames / fps,
    )

    # Write the VideoClip to a file
    clip.write_videofile(f"{title}_{video_id}.mp4", fps=fps)


class DDPG:
    def __init__(
        self,
        *,
        replay_buffer_fn,
        policy_model_fn, 
        policy_max_grad_norm, 
        policy_optimizer_fn, 
        value_model_fn, 
        value_max_grad_norm, 
        value_optimizer_fn, 
        training_strategy_fn,
        evaluation_strategy_fn,
        n_warmup_batches,
        update_target_every_steps,
        tau,
        leave_print_every_n_secs,
    ):
        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        
        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn

        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau
        self.leave_print_every_n_secs = leave_print_every_n_secs

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

    def train(
        self,
        *,
        make_env_fn,
        seed,
        gamma, 
        max_minutes,
        max_episodes,
        goal_mean_100_reward,
    ):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.gamma = gamma
        
        env = self.make_env_fn()
        torch.manual_seed(seed) ; np.random.seed(seed) ; random.seed(seed)
    
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
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model.parameters())        
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model.parameters())

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)
                    
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
            
            # Stats
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
            
            reached_debug_time = time.time() - last_debug_time >= self.leave_print_every_n_secs
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
            env = self.make_env_fn(monitor_mode='evaluation', do_render=True, do_record=True)
            self.evaluate(self.online_policy_model, env, n_episodes=1)
            env.close()
            get_gif_html(
                env.videos[0], 
                title.format(self.__class__.__name__),
                i,
            )
        del env

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

