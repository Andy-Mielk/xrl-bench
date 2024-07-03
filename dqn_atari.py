# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

"""
For playing Atari games. It uses convolutional layers and common atari-based pre-processing techniques.
Works with the Atari's pixel Box observation space of shape (210, 160, 3)
Works with the Discrete action space
https://docs.cleanrl.dev/rl-algorithms/dqn/
"""

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True     # 尝试使用wandb追踪实验，账户https://wandb.ai/andymielk-Nanjing%20University
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False     # 录制视频
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True    # 保存模型
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"  # 环境名称，关于版本选择详见https://gymnasium.farama.org/environments/atari/#version-history-and-naming-schemes，
    """the id of the environment"""
    total_timesteps: int = 10000000     # 总训练步数
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")   # 录制视频
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env) # 记录每个episode的统计信息

        env = NoopResetEnv(env, noop_max=30)    # 重置环境时，执行0到30个随机动作
        env = MaxAndSkipEnv(env, skip=4)    # 实现帧跳过（frame skipping）和最大化过去观察到的帧，以减少计算量并稳定训练。
        env = EpisodicLifeEnv(env)  # 将生命损失视为结束episode，但只在真正的游戏结束时重置环境，这有助于更好地估计价值函数。
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)    # Clip the reward to {+1, 0, -1} by its sign.
        env = gym.wrappers.ResizeObservation(env, (84, 84))    # Resize the observation to square image
        env = gym.wrappers.GrayScaleObservation(env)    # Grayscale the observation
        env = gym.wrappers.FrameStack(env, 4)   # Stack 4 frames，用于堆叠连续的几帧观察结果，以提供给代理一些关于动作的时间上下文。

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
            # nn.Linear(512, env.action_space.n),
            # nn.Linear(512, 4),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"loaded model from {model_path}")
    model.eval()

    obs, _ = envs.reset()   # obs:[n_envs, 4, 84, 84] type:np.ndarray
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        # 修改采用确定性策略（deterministic policy）而不是 epsilon-greedy 策略
        # if random.random() < epsilon:
        #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # else:
        #     q_values = model(torch.Tensor(obs).to(device))
        #     actions = torch.argmax(q_values, dim=1).cpu().numpy()
        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()   # actions: [n_envs,] type:np.ndarray

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs  # 矢量化环境中的子环境在episode结束时会自动调用obj：reset, 所以不需要手动reset

    return episodic_returns


# 评估函数，修改了episode结束的判定条件和episodic_return的计算方式，简化了环境创建方式
# 注意，与此同时，修改了QNetwork的__init__方法，最后一行改成了nn.Linear(512, env.action_space.n)，而不是原来的多环境的nn.Linear(512, env.single_action_space.n)
# 还修改了采用确定性策略（deterministic policy）而不是 epsilon-greedy 策略。这是因为评估的目的是测试模型的性能，而不是进一步探索环境。确定性策略通常选择当前 Q 值最高的动作，以确保模型表现出其最优策略。
# 这个代码感觉跑起来很慢，不知道为什么，而且performance不好，可能是因为没有使用vectorized envs？
def evaluate1(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    env = make_env(env_id, 0, 0, capture_video, run_name)()
    # envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(env).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"loaded model from {model_path}")
    model.eval()

    obs, _ = env.reset()    # obs: [4, 84, 84]，即4个84*84的图像, 4是由于make_env中FrameStack(env, 4)的作用, 但是类别不是ndarray而是 <class 'gymnasium.wrappers.frame_stack.LazyFrames'>
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        # obs_tensor = torch.Tensor(np.array([obs])).to(device)
        # obs_tensor = torch.as_tensor(np.array(obs)).to(device)
        # obs_tensor = torch.Tensor([obs]).to(device) # 与上一行等价，obs_tensor: [1, 4, 84, 84]
        # q_values = model(torch.as_tensor(np.array([obs])).to(device))
        # q_values = model(torch.as_tensor(obs).unsqueeze(0).to(device))
        
        # 在obs前面添加一维，作为 batch size
        obs_batch = np.expand_dims(obs, axis=0)
        obs_tensor = torch.Tensor(obs_batch).to(device)
        q_values = model(obs_tensor)
        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]  # action: numpy.int64, 0-3

        next_obs, _, _, _, info = env.step(action)

        if 'episode' in info:
            episodic_returns.append(info['episode']['r'])
            print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    env.close()
    return episodic_returns



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"   # for this, run_name = "BreakoutNoFrameskip-v4__dqn_atari__1__1634261234"
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{formatted_time}"   # for this, run_name = "BreakoutNoFrameskip-v4__dqn_atari__1__2024-07-01 20:20:34"


    # 评估模型代码
    model_path = f"runs/BreakoutNoFrameskip-v4__dqn_atari__1__2024-07-01 17:33:55/{args.exp_name}.cleanrl_model"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device {device}")
    episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=True,
        )
    # for i in range(10):
    #     evaluate0(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=1,
    #         run_name=f"{run_name}-eval",
    #         Model=QNetwork,
    #         device=device,
    #         epsilon=0.05,
    #         capture_video=False,
    #     )



    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # epsilon greedy exploration，探索或利用，具体来说，以epsilon的概率随机选择动作，以1-epsilon的概率选择Q值最大的动作
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # 训练结束后，保存模型，并评估
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
