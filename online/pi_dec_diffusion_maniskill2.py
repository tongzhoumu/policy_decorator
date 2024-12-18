ALGO_NAME = 'PolicyDecorator-DiffusionPolicy'

import os
import argparse
import random
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict
from utils.profiling import NonOverlappingTimeProfiler

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from nets.diffusion_policy.conditional_unet1d_colab import ConditionalUnet1D


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

    # SAC arguments
    parser.add_argument("--env-id", type=str, default="PegInsertionSide-v2",
        help="the id of the environment")
    parser.add_argument("--base-policy-ckpt", type=str, default='checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt')
    parser.add_argument("--total-timesteps", type=int, default=4_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=None,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.97,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=1024,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=8000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=1e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--sac_alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=50,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--utd", type=float, default=0.25,
        help="Update-to-Data ratio (number of gradient updates / number of env steps)")
    parser.add_argument("--training-freq", type=int, default=64)
    parser.add_argument("--bootstrap-at-done", type=str, choices=['always', 'never', 'truncated'], default='truncated',
        help="in ManiSkill variable episode length and dense reward setting, set to always if positive reawrd, truncated if negative reward; in sparse reward setting, any of them should be fine")

    # Policy Decorator arguments
    parser.add_argument("--res-scale", type=float, default=0.1) # alpha, need to tune
    parser.add_argument("--prog-explore", type=int, default=100_000) # H, need to tune
    parser.add_argument("--critic-input", type=str, choices=['res', 'sum', 'concat'], default='sum')
    parser.add_argument("--actor-input", type=str, choices=['obs', 'obs_base_action'], default='obs')
    parser.add_argument("--log-std-min", type=float, default=-20)

    # Diffusion Policy arguments
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--act-horizon", type=int, default=4) # override the base policy's act_horizon

    # Eval, logging, and others
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--num-eval-envs", type=int, default=5)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--log-freq", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=1_000_000)
    parser.add_argument("--obj-ids", metavar='N', type=str, nargs='+', default=[])

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps
    args.buffer_size = min(args.total_timesteps, args.buffer_size)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    assert args.training_freq % args.num_envs == 0
    assert (args.training_freq * args.utd).is_integer()
    # fmt: on
    return args

import mani_skill2.envs
import envs.maniskill_fixed # register the environments for policy decorator
from mani_skill2.utils.wrappers import RecordEpisode
gym.logger.set_level(gym.logger.DEBUG)

class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info

def make_env(env_id, seed, control_mode=None, video_dir=None, other_kwargs={}):
    def thunk():
        env_kwargs = {'model_ids': other_kwargs['obj_ids']} if len(other_kwargs['obj_ids']) > 0 else {}
        env = gym.make(env_id, reward_mode='sparse', obs_mode='state', control_mode=control_mode,
                        render_mode='cameras' if video_dir else None, **env_kwargs)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FrameStack(env, other_kwargs['obs_horizon'])
        env = SeqActionWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class BasePolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )
        self.noise_scheduler.set_timesteps(num_inference_steps=args.ddim_steps)

        self.get_eval_action = self.get_action
    
    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)
        
        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)
    
    def get_action(self, obs_seq):
        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
            # print(obs_cond.cpu().numpy()); exit()
            # print(list(self.noise_pred_net.parameters())[-1].cpu().numpy()); exit()

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        input_dim = obs_dim if args.actor_input == 'obs' else obs_dim + np.prod(env.single_action_space.shape)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)), std=0.01)

        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            print(f"global_step={global_step}, ep_return={ep['r'][0]:.2f}, ep_len={ep['l'][0]}, success={info['success']}")
            result['return'].append(ep['r'][0])
            result['len'].append(ep["l"][0])
            result['success'].append(info['success'])
    return result

def evaluate(n, base_policy, res_actor, eval_envs, device):
    print('======= Evaluation Starts =========')
    res_actor.eval()
    result = defaultdict(list)
    obs_seq, info = eval_envs.reset() # don't seed here
    while len(result['return']) < n:
        obs_seq_tensor = torch.Tensor(obs_seq).to(device)
        with torch.no_grad():
            base_act_seq_tensor = base_policy.get_eval_action(obs_seq_tensor).detach()
            base_act_seq = base_act_seq_tensor.cpu().numpy()
            actor_input = obs_seq_tensor[:, -1] if args.actor_input == 'obs' else torch.cat([obs_seq_tensor[:, -1], base_act_seq_tensor.reshape(-1, total_act_dim)], dim=1)
            res_actions = res_actor.get_eval_action(actor_input).detach().cpu().numpy()
        res_act_seq = res_actions.reshape(-1, args.act_horizon, act_dim)
        scaled_res_seq = args.res_scale * res_act_seq
        final_act_seq = base_act_seq + scaled_res_seq
        obs_seq, rew, terminated, truncated, info = eval_envs.step(final_act_seq)
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    res_actor.train()
    return result

def is_ms1_env(env_id):
    return 'OpenCabinet' in env_id or 'MoveBucket' in env_id or 'PushChair' in env_id


if __name__ == "__main__":
    args = parse_args()
    LOG_STD_MIN = args.log_std_min

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
    from os.path import dirname as up
    exp_dir = up(up(args.base_policy_ckpt))
    with open(f'{exp_dir}/args.json', 'r') as f:
        ckpt_args = json.load(f)
        for k, v in ckpt_args.items():
            if k not in args.__dict__:
                args.__dict__[k] = v
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv or args.num_envs == 1 \
        else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
    envs = VecEnv(
        [make_env(args.env_id, args.seed + i, args.control_mode, other_kwargs=args.__dict__)
        for i in range(args.num_envs)]
    )
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
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert len(envs.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)

    max_action = float(envs.single_action_space.high[0])
    act_dim = np.prod(envs.single_action_space.shape)
    total_act_dim = act_dim * args.act_horizon
    
    # Load base policy
    base_policy = BasePolicy(envs, args).to(device)
    checkpoint = torch.load(args.base_policy_ckpt)
    for key in ['ema_agent', 'agent', 'actor', 'q']:
        if key in checkpoint:
            base_policy.load_state_dict(checkpoint[key])
            print(f'Loaded agent from checkpoint[{key}]')
            break
    base_policy.eval()
    base_policy.requires_grad_(False)
    
    # Residual agent setup
    class DummyObject: pass
    dummy_env = DummyObject() # dummy env is used to define actor and critic
    dummy_env.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
        shape=(envs.single_observation_space.shape[1],))
    dummy_env.single_action_space = gym.spaces.Box(low=envs.single_action_space.low.repeat(args.act_horizon),
        high=envs.single_action_space.high.repeat(args.act_horizon), shape=(total_act_dim,))
    res_actor = Actor(dummy_env, args).to(device)
    if args.critic_input == 'concat':
        dummy_env.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_act_dim * 2,))
    qf1 = SoftQNetwork(dummy_env).to(device)
    qf2 = SoftQNetwork(dummy_env).to(device)
    if is_ms1_env(args.env_id):
        for m in list(res_actor.modules()) + list(qf1.modules()) + list(qf2.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.zeros_(m.bias)
    qf1_target = SoftQNetwork(dummy_env).to(device)
    qf2_target = SoftQNetwork(dummy_env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(res_actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(dummy_env.single_action_space.shape).to(device)).item()
        log_sac_alpha = torch.zeros(1, requires_grad=True, device=device)
        sac_alpha = log_sac_alpha.exp().item()
        a_optimizer = optim.Adam([log_sac_alpha], lr=args.q_lr)
    else:
        sac_alpha = args.sac_alpha

    dummy_env.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        dummy_env.single_observation_space,
        dummy_env.single_action_space if args.critic_input == 'res' and args.actor_input == 'obs' else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_act_dim * 3,)),
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False, # stable-baselines3 has not fully supported Gymnasium's termination signal
    )

    # TRY NOT TO MODIFY: start the game
    obs_seq, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    global_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(args.training_freq * args.utd)
    result = defaultdict(list)
    step_in_episodes = np.zeros((args.num_envs,1,1), dtype=np.int32)
    timer = NonOverlappingTimeProfiler()

    while global_step < args.total_timesteps:

        # Collect samples from environemnts
        for local_step in range(args.training_freq // args.num_envs):
            global_step += 1 * args.num_envs

            obs_seq_tensor = torch.Tensor(obs_seq).to(device)
            base_act_seq_tensor = base_policy.get_eval_action(obs_seq_tensor)
            base_act_seq = base_act_seq_tensor.cpu().numpy() # (B, act_horizon, act_dim)
            base_actions = base_act_seq.reshape(-1, total_act_dim)
            res_ratio = min(global_step / args.prog_explore, 1)
            enable_res_masks = np.random.rand(args.num_envs) < res_ratio

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                res_actions = np.array([dummy_env.single_action_space.sample() for _ in range(envs.num_envs)]) # (B, act_horizon*act_dim)
                res_actions[np.logical_not(enable_res_masks)] = 0.0
            else:
                actor_input = obs_seq_tensor[:, -1] if args.actor_input == 'obs' else torch.cat([obs_seq_tensor[:, -1], base_act_seq_tensor.flatten(start_dim=1)], dim=1)
                res_actions, _, _ = res_actor.get_action(actor_input)
                res_actions = res_actions.detach().cpu().numpy() # (B, act_horizon*act_dim)
                res_actions[np.logical_not(enable_res_masks)] = 0.0

            res_act_seq = res_actions.reshape(-1, args.act_horizon, act_dim)
            scaled_res_seq = args.res_scale * res_act_seq # (B, act_horizon, act_dim)
            final_act_seq = base_act_seq + scaled_res_seq

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_seq, rewards, terminations, truncations, infos = envs.step(final_act_seq)
            rewards = rewards - 1.0 # negative reward + bootstrap at truncated yields best results

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            result = collect_episode_info(infos, result)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs_seq = next_obs_seq.copy()
            if args.bootstrap_at_done == 'never':
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = np.zeros_like(terminations, dtype=bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
                for idx, _need_final_obs in enumerate(need_final_obs):
                    if _need_final_obs:
                        real_next_obs_seq[idx] = infos["final_observation"][idx]

            if args.critic_input == 'res' and args.actor_input == 'obs':
                actions_to_save = res_actions
            else: # sum or concat both need base actions for s and s'
                base_next_act_seq = base_policy.get_eval_action(torch.Tensor(real_next_obs_seq).to(device)).cpu().numpy()
                base_next_actions = base_next_act_seq.reshape(-1, total_act_dim)
                actions_to_save = np.concatenate([res_actions, base_actions, base_next_actions], axis=1)
            
            rb.add(obs_seq[:, -1], real_next_obs_seq[:, -1], actions_to_save, rewards, stop_bootstrap, infos)

            step_in_episodes += args.act_horizon
            step_in_episodes[terminations | truncations] = 0

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs_seq = next_obs_seq
        
        timer.end('collect')

        # ALGO LOGIC: training.
        if rb.size() * rb.n_envs < args.learning_starts:
            continue

        learning_has_started = True
        for local_update in range(num_updates_per_training):
            global_update += 1
            data = rb.sample(args.batch_size)

            if args.critic_input != 'res' or args.actor_input == 'obs_base_action':
                res_actions = data.actions[:, :total_act_dim]
                base_actions = data.actions[:, total_act_dim:total_act_dim*2]
                base_next_actions = data.actions[:, -total_act_dim:]
            else:
                res_actions = data.actions

            #############################################
            # Train agent
            #############################################

            # update the value networks
            with torch.no_grad():
                actor_input = data.next_observations if args.actor_input == 'obs' else torch.cat([data.next_observations, base_next_actions], dim=1)
                next_state_res_actions, next_state_log_pi, _ = res_actor.get_action(actor_input)
                if args.critic_input == 'res':
                    next_state_actions = next_state_res_actions
                elif args.critic_input == 'sum':
                    scaled_res_actions = args.res_scale * next_state_res_actions
                    next_state_actions = base_next_actions + scaled_res_actions
                else: # concat
                    next_state_actions = torch.cat([next_state_res_actions, base_next_actions], dim=1)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            if args.critic_input == 'res':
                current_actions = res_actions
            elif args.critic_input == 'sum':
                scaled_res_actions = args.res_scale * res_actions
                current_actions = base_actions + scaled_res_actions
            else: # concat
                current_actions = torch.cat([res_actions, base_actions], dim=1)
            qf1_a_values = qf1(data.observations, current_actions).view(-1)
            qf2_a_values = qf2(data.observations, current_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            qf1_grad_norm = nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)
            qf2_grad_norm = nn.utils.clip_grad_norm_(qf2.parameters(), args.max_grad_norm)
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                actor_input = data.observations if args.actor_input == 'obs' else torch.cat([data.observations, base_actions], dim=1)
                res_pi, log_pi, _ = res_actor.get_action(actor_input)
                if args.critic_input == 'res':
                    pi = res_pi
                elif args.critic_input == 'sum':
                    scaled_res_actions = args.res_scale * res_pi
                    pi = base_actions + scaled_res_actions
                else: # concat
                    pi = torch.cat([res_pi, base_actions], dim=1)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(res_actor.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = res_actor.get_action(actor_input)
                    sac_alpha_loss = (-log_sac_alpha * (log_pi + target_entropy)).mean()
                    # log_sac_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    sac_alpha_loss.backward()
                    a_optimizer.step()
                    sac_alpha = log_sac_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        timer.end('train')

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/sac_alpha", sac_alpha, global_step)
            writer.add_scalar("losses/qf1_grad_norm", qf1_grad_norm.item(), global_step)
            writer.add_scalar("losses/qf2_grad_norm", qf2_grad_norm.item(), global_step)
            writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            timer.dump_to_writer(writer, global_step)
            if args.autotune:
                writer.add_scalar("losses/sac_alpha_loss", sac_alpha_loss.item(), global_step)

        # Evaluation
        if (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            result = evaluate(args.num_eval_episodes, base_policy, res_actor, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
            timer.end('eval')
        
        # Checkpoint
        if args.save_freq and ( global_step >= args.total_timesteps or \
                (global_step - args.training_freq) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'res_actor': res_actor.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
                'log_sac_alpha': log_sac_alpha if args.autotune else np.log(args.sac_alpha),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()
