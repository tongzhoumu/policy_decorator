ALGO_NAME = 'Offline-BeT'

import argparse
import os
import random
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict
from utils.profiling import NonOverlappingTimeProfiler

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from utils.sampler import IterationBasedBatchSampler
from utils.torch_utils import worker_init_fn

from nets.behavior_transformer import BehaviorTransformer, GPT, GPTConfig


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='test',
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="policy_decorator",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Imitation Learning arguments
    parser.add_argument("--env-id", type=str, default="TurnFaucet-v2",
        help="the id of the environment")
    parser.add_argument("--demo-path", type=str, default='data/TurnFaucet/trajectory.h5',
        help="the path of demo dataset (pkl or h5)")
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=200_000,
        help="total timesteps of the experiments")
    parser.add_argument("--batch-size", type=int, default=2048, # marginal improvements beyond 2048
        help="the batch size of sample from the replay memory")
    
    # BeT arguments
    parser.add_argument("--lr", type=float, default=1e-4) # Important, should be jointly tuned with batch-size
    parser.add_argument("--context-window", type=int, default=20) # Important
    parser.add_argument("--n-clusters", type=int, default=8) # Very important
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embedding", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=2e-4)

    # Eval, logging, and others
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--num-eval-envs", type=int, default=10) # NOTE: should not be too large, otherwise bias to short episodes
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-dataload-workers", type=int, default=0)
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pose')
    parser.add_argument("--obj-ids", metavar='N', type=str, nargs='+', default=[])

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    if args.num_eval_envs == 1:
        args.sync_venv = True
    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, 'Control mode mismatched'
    # fmt: on
    return args

import mani_skill2.envs
import envs.maniskill_fixed # register the environments for policy decorator
from mani_skill2.utils.wrappers import RecordEpisode

def make_env(env_id, seed, control_mode=None, video_dir=None, other_kwargs={}):
    def thunk():
        env_kwargs = {'model_ids': other_kwargs['obj_ids']} if len(other_kwargs['obj_ids']) > 0 else {}
        env = gym.make(env_id, reward_mode='sparse', obs_mode='state', control_mode=control_mode,
                        render_mode='cameras' if video_dir else None, **env_kwargs)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FrameStack(env, other_kwargs['context_window'])

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class SmallDemoDataset_with_history(Dataset):
    def __init__(self, data_path, device, history_len, num_traj):
        from utils.ms2_data import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        
        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        self.slices = []
        num_traj = len(trajectories['actions'])
        min_traj_length = np.inf
        ignore_cnt = 0
        total_transitions = 0
        for i in range(num_traj):
            L = trajectories['actions'][i].shape[0]
            assert trajectories['observations'][i].shape[0] == L + 1
            total_transitions += L
            min_traj_length = min(L, min_traj_length)
            if L - history_len < 0:
                print(f"Ignored short trajectory #{i}: len={L}, history={history_len}")
                ignore_cnt += 1
            else:
                self.slices += [
                    (i, start, start + history_len) for start in range(L - history_len)
                ]  # slice indices follow convention [start, end)

        if min_traj_length < history_len:
            print(f"Ignored {ignore_cnt} short trajectories out of {num_traj}. To include all, set window <= {min_traj_length}.")
        
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        i, start, end = self.slices[index]
        return {k: v[i][start:end] for k, v in self.trajectories.items()}

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space.shape) == 2 # (context_window, obs_dim)
        obs_dim = env.single_observation_space.shape[1]
        self.net = BehaviorTransformer(
            obs_dim=obs_dim,
            act_dim=np.prod(env.single_action_space.shape),
            goal_dim=-1,
            gpt_model=GPT(
                GPTConfig(
                    block_size=args.context_window,
                    input_dim=obs_dim,
                    n_layer=args.n_layers,
                    n_head=args.n_heads,
                    n_embd=args.n_embedding,
                )
            ),  # The sequence model to use.
            n_clusters=args.n_clusters,  # Number of clusters to use for k-means discretization.
            kmeans_fit_steps=-1, # manually call _fit_kmeans() later
        )
        self.get_eval_action = self.get_action

    def forward(self, obs_seq, action_seq=None):
        return self.net(obs_seq, None, action_seq)
    
    def get_action(self, obs_seq):
        # obs_seq: (B, context_window, obs_dim)
        with torch.no_grad():
            action_seq, _, _ = self.net(obs_seq, None, None)
        return action_seq[:, -1] # (B, act_dim)

def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            print(f"global_step={cur_iter}, ep_return={ep['r'][0]:.2f}, ep_len={ep['l'][0]}, success={info['success']}")
            result['return'].append(ep['r'][0])
            result['len'].append(ep["l"][0])
            result['success'].append(info['success'])
    return result

def evaluate(n, agent, eval_envs, device):
    # agent.eval() #  NOTE: turning off dropout leads to worse performance, don't know the reason
    print('======= Evaluation Starts =========')
    result = defaultdict(list)
    obs, info = eval_envs.reset() # don't seed here
    while len(result['return']) < n:
        with torch.no_grad():
            action = agent.get_eval_action(torch.Tensor(obs).to(device))
        obs, rew, terminated, truncated, info = eval_envs.step(action.cpu().numpy())
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    # agent.train()
    return result

def save_ckpt(tag):
    os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
    torch.save({
        'agent': agent.state_dict(),
    }, f'{log_path}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json
    with open(f'{log_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv or args.num_eval_envs == 1 \
        else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i, args.control_mode,
                f'{log_path}/videos' if args.capture_video and i == 0 else None,
                other_kwargs=args.__dict__,
                )
        for i in range(args.num_eval_envs)]
    )
    eval_envs.reset(seed=args.seed+1000) # seed eval_envs here, and no more seeding during evaluation
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    # dataloader setup
    dataset = SmallDemoDataset_with_history(args.demo_path, device, 
                    history_len=args.context_window, num_traj=args.num_demo_traj)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )

    # agent setup
    agent = Agent(envs, args).to(device)
    optimizer = agent.net.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=[0.9, 0.999],
    )
    action_dataset = []
    for action_traj in dataset.trajectories['actions']:
        action_dataset += action_traj
    action_dataset = torch.stack(action_dataset, dim=0) # (N, act_dim)
    agent.net._fit_kmeans_from_action_dataset(action_dataset)
    model = agent


    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    loss_fn = F.mse_loss
    best_success_rate = -1

    timer = NonOverlappingTimeProfiler()

    for iteration, data_batch in enumerate(train_dataloader):
        cur_iter = iteration + 1
        timer.end('data')

        # forward and compute loss
        pred_actions, total_loss, loss_dict = model(
            obs_seq=data_batch['observations'], # (B, L, obs_dim)
            action_seq=data_batch['actions'], # (B, L, act_dim)
        )
        timer.end('forward')

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        timer.end('backward')

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if cur_iter % args.log_freq == 0: 
            print(cur_iter, total_loss.item())
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            for k, v in loss_dict.items():
                writer.add_scalar(f"losses/{k}", v, cur_iter)
            timer.dump_to_writer(writer, cur_iter)

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), cur_iter)
            timer.end('eval')
            sr = np.mean(result['success'])
            if sr > best_success_rate:
                best_success_rate = sr
                save_ckpt('best_eval_success_rate')
                print(f'### Update best success rate: {sr:.4f}')

        # Checkpoint
        if args.save_freq and cur_iter % args.save_freq == 0:
            save_ckpt(str(cur_iter))

    envs.close()
    writer.close()
