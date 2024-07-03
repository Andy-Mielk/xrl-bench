# -*- coding: utf-8 -*-

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment

# 可视化importance
import matplotlib.pyplot as plt
import numpy as np
import os
from d3rlpy.dataset import MDPDataset
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_atari import QNetwork, make_env, Args
import tyro
import time

def normalize_scores(scores):
    """
    对显著性分数进行Min-Max归一化, 将显著性分数缩放到 [0, 1] 范围。

    Parameters:
    -----------
    scores : numpy.ndarray
        显著性分数，形状为 (n, H, W)。

    Returns:
    --------
    numpy.ndarray
        归一化后的显著性分数。
    """
    norm_scores = (scores - np.min(scores, axis=(1, 2), keepdims=True)) / (np.max(scores, axis=(1, 2), keepdims=True) - np.min(scores, axis=(1, 2), keepdims=True) + 1e-10)
    return norm_scores

def plot_importance_on_images(observations, importance_scores, save_dir='plots'):
    """
    在原始图像上绘制重要性分数，并将图像保存到文件中。
    
    Parameters:
    -----------
    observations : numpy.ndarray
        原始输入图像，形状为 (n, C, H, W)。
    importance_scores : numpy.ndarray
        计算得出的显著性分数，形状为 (n, H, W)。
    save_dir : str
        保存图像的目录。
    """
    num_images = observations.shape[0]
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 对显著性分数进行归一化
    normalized_importance_scores = normalize_scores(importance_scores)

    for i in range(num_images):
        image = observations[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        importance = normalized_importance_scores[i]
        
        plt.figure(figsize=(10, 10))
        
        # 绘制原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(image.astype(np.uint8))
        plt.title("Original Image")
        # plt.axis('off')
        
        # 绘制显著性分数热力图
        plt.subplot(1, 2, 2)
        plt.imshow(image.astype(np.uint8))
        plt.imshow(importance, cmap='jet', alpha=0.5)  # 使用透明度叠加热力图, cmap='jet', alpha=0.5
        plt.title("Saliency Map")
        # plt.axis('off')
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f'image_{i}.png'))
        plt.close()


def plot_importance_separate_from_images(observations, importance_scores, save_dir='plots_separate'):
    """
    分别绘制原始图像和显著性分数图，并将图像保存到文件中。
    
    Parameters:
    -----------
    observations : numpy.ndarray
        原始输入图像，形状为 (n, C, H, W)。
    importance_scores : numpy.ndarray
        计算得出的显著性分数，形状为 (n, H, W)。
    save_dir : str
        保存图像的目录。
    """
    num_images = observations.shape[0]
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 对显著性分数进行归一化
    normalized_importance_scores = normalize_scores(importance_scores)

    for i in range(num_images):
        image = observations[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        importance = importance_scores[i]
        
        plt.figure(figsize=(10, 5))
        
        # 绘制原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image.astype(np.uint8))
        plt.title("Original Image")
        plt.axis('off')
        
        # 绘制显著性分数图
        plt.subplot(1, 3, 2)
        plt.imshow(importance, cmap='hot', interpolation='nearest')  # 使用 'hot' 颜色映射
        plt.title("Saliency Map")
        plt.colorbar()
        plt.axis('off')

        # 绘制显著性分数图，归一化的
        plt.subplot(1, 3, 3)
        plt.imshow(normalized_importance_scores[i], cmap='hot', interpolation='nearest')  # 使用 'hot' 颜色映射
        plt.title("Normalized Saliency Map")
        plt.colorbar()
        plt.axis('off')
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f'image_{i}.png'))
        plt.close()

# 示例调用（假设 sample_observations 和 sample_importance 已经定义好）
# plot_importance_on_images(sample_observations, sample_importance, save_dir=os.path.join('plots', 'imageDeepShap'))

# Tabular数据测试
def tabular_input_test(environment, method, metric, k=3):
    environment = Environment(environment_name=environment)
    df = environment.get_dataset(generate=False)
    df_sample = df.sample(n=5000, random_state=42)
    action_sample = df_sample['action']
    state_sample = df_sample.drop(['action', 'reward'], axis=1)
    if method == "tabularShap":
        explainer = Explainer(method=method, state=state_sample, action=action_sample)
    else:
        explainer = Explainer(method=method, state=state_sample, action=action_sample, model=environment.model)
    importance = explainer.explain()
    evaluator = Evaluator(metric=metric, environment=environment)
    if metric == "RIS":
        performance = evaluator.evaluate(state_sample, action_sample, importance, explainer=explainer)
    else:
        performance = evaluator.evaluate(state_sample, action_sample, importance, k=k)
    return performance


def make_environment():
    # 创建环境 envs
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{formatted_time}"   # for this, run_name = "BreakoutNoFrameskip-v4__dqn_atari__1__2024-07-01 20:20:34"
 
    if args.num_envs == 1:
        # 不知道这样写行不行
        envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return envs

# Image数据测试
def image_input_test(env, method, metric, device, k=50):
    # environment = Environment(environment_name=environment)
    # dataset = environment.get_dataset(generate=False, data_format="h5")
    
    # 采样
    # for breakOut env, dataset's state: (S-A pair size=3776, 3, 84, 84), action: (S-A pair size=3776, 4), importance: (3776, 3, 84, 84, 4)
    # for pong env, dataset's state: (S-A pair size=4000, 3, 84, 84), action: (S-A pair size=4000, 6), importance: (4000, 3, 84, 84, 6)
    sample_indices = [1000,2000,3000,3100]
    sample_observations = dataset.observations[sample_indices]  # (n, 3, 84, 84) 0-255
    sample_actions = dataset.actions[sample_indices]     # (n, ) 0-n_actions

    """
    调用Explainer需要得到: state,action,model
    1. get_dataset()返回的是一个MDPDataset对象, 包含了state,action,reward,terminal, ndarray类型
    2. model: DQN pytorch model, x->Q:action_size
    """
    
    args = tyro.cli(Args)


    dataset = get_dataset(env=environment, agent=environment.agent, generate=True, data_format="h5")
    

    q_network = QNetwork(env).to(device)
    # 记得修改路径
    q_network.load_state_dict(torch.load(os.path.join(".", "model", "BreakOut.pth")))

    # 仅对样本数据进行解释
    if method == "imagePerturbationSaliency" or method == "imageSarfa":
        explainer = Explainer(method=method, state=sample_observations, action=sample_actions,
                          model=environment.model)
        sample_importance = explainer.explain() # (n, 84, 84) for imagePerturbationSaliency and imageSarfa
    # 下面几种基于shap的方法，需要对所有数据进行解释，然后再从中取出样本数据的解释，因为shap的计算需要background数据集
    elif method == "imageDeepShap" or method == "imageGradientShap" or method == "imageIntegratedGradient":
        explainer = Explainer(method=method, state=dataset.observations, action=dataset.actions,
                          model=environment.model)
        all_importance = explainer.explain()    # （n_SA_pairs, 3, 84, 84, n_actions）
        # 由于shap返回的是每个channel每个action的importance，所以需要选择特定动作的 SHAP 值，且在通道维度求和
        # 下面这行参考了官方aim.py的111行的写法
        # shap值: (n_samples, 3, 84, 84, n_actions) -> (n_samples, 84, 84) 选取相应的action，在通道维度求和
        all_importance_for_actions = np.array( [np.sum(all_importance[i, :, :, :, int(dataset.actions[i])], axis=0) for i in range(len(all_importance))] )        
        # 再取对应 sample_indices 的 SHAP 值
        sample_importance = all_importance_for_actions[sample_indices]
    
    # 可视化
    plot_importance_on_images(sample_observations, sample_importance, save_dir=os.path.join('plots', method))
    plot_importance_separate_from_images(sample_observations, sample_importance, save_dir=os.path.join('plots_separate', method))

    # # Print the sampled observations and their importance
    # for obs, imp in zip(sample_observations, sample_importance):
    #     print("Observation:")
    #     print(obs)
    #     print("Importance:")
    #     print(imp)
    #     print("--------------------")
    
    # evaluator = Evaluator(metric=metric, environment=environment)
    # if metric == "RIS":
    #     performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, explainer=explainer)
    # else:
    #     performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, k=k)
    # return performance
    return


def get_dataset(env, agent, generate=False, n_episodes=10, max_t=500, data_format="h5"):
        """
        从environment.py中提取过来的函数, 用于获取数据集    
        Get the dataset for the environment.

        Parameters:
        -----------
        generate : bool, optional
            Whether to generate a new dataset or use an existing one. (default=False)
        n_episode : int, optional
            The number of episodes to generate the dataset from. (default=10)
        max_t : int, optional
            The maximum number of timesteps per episode. (default=1000)
        data_format : str, optional
            The format of the dataset. Currently only supports 'h5'. (default='h5')

        Returns:
        --------
        dataset : MDPDataset
            The dataset including state, action, reward and terminal.
        """
        if generate:
            data = []
            for i in range(n_episodes):
                state = env.reset()[0]
                for t in range(max_t):
                    # todo: get action from model
                    action = agent.act(np.array(state), inferring=True)
                    next_state, reward, done, _, _ = env.step(action)
                    # todo: get the real state
                    data.append({"state": np.array(state*255, dtype=np.uint8), "action": np.array([action]), "reward": np.array([reward]), "terminal": np.array([done])})
                    state = next_state
                    if done:
                        break
            if data_format == "h5":
                observations = np.vstack([row["state"] for row in data])
                actions = np.vstack([row["action"] for row in data])
                rewards = np.vstack([row["reward"] for row in data])
                terminals = np.vstack([row["terminal"] for row in data])
                dataset = MDPDataset(observations, actions, rewards, terminals)
                dataset.dump(os.path.join(".", "data", "new", "BreakOut_dataset.h5"))   # 写入新的数据集
                return dataset
            else:
                raise NotImplementedError("This data format is not supported at the moment.")
        else:
            try:
                if data_format == "h5":
                    dataset = MDPDataset.load(os.path.join(".", "data", "BreakOut_dataset.h5")) # 读取已有的数据集
                    return dataset
                else:
                    raise NotImplementedError("This data format is not supported at the moment.")
            except:
                print("This dataset is not existing, please generate it.")








if __name__ == "__main__":
    # performance = image_input_test(env, method, metric, device=device, k=50)
    
    # Image数据测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = tyro.cli(Args)
    method = "imageGradientShap"    # 'imagePerturbationSaliency', 'imageSarfa', 'imageDeepShap', 'imageGradientShap', 'imageIntegratedGradient'
    metric = "imageAIM"

    # ## 创建环境
    env = make_environment()

    # ## 加载DQN模型
    q_network = QNetwork(env).to(device)
    # 记得修改路径
    q_network.load_state_dict(torch.load(os.path.join(".", "model", "BreakOut.pth")))

    # ## 获取数据集
    dataset = get_dataset(env=environment, agent=environment.agent, generate=True, data_format="h5")

    
    # 采样
    # for breakOut env, dataset's state: (S-A pair size=3776, 3, 84, 84), action: (S-A pair size=3776, 4), importance: (3776, 3, 84, 84, 4)
    # for pong env, dataset's state: (S-A pair size=4000, 3, 84, 84), action: (S-A pair size=4000, 6), importance: (4000, 3, 84, 84, 6)
    sample_indices = [1000,2000,3000,3100]
    sample_observations = dataset.observations[sample_indices]  # (n, 3, 84, 84) 0-255
    sample_actions = dataset.actions[sample_indices]     # (n, ) 0-n_actions

    """
    调用Explainer需要得到: state,action,model
    1. get_dataset()返回的是一个MDPDataset对象, 包含了state,action,reward,terminal, ndarray类型
    2. model: DQN pytorch model, x->Q:action_size
    """
    


    



    # 仅对样本数据进行解释
    if method == "imagePerturbationSaliency" or method == "imageSarfa":
        explainer = Explainer(method=method, state=sample_observations, action=sample_actions,
                          model=environment.model)
        sample_importance = explainer.explain() # (n, 84, 84) for imagePerturbationSaliency and imageSarfa
    # 下面几种基于shap的方法，需要对所有数据进行解释，然后再从中取出样本数据的解释，因为shap的计算需要background数据集
    elif method == "imageDeepShap" or method == "imageGradientShap" or method == "imageIntegratedGradient":
        explainer = Explainer(method=method, state=dataset.observations, action=dataset.actions,
                          model=environment.model)
        all_importance = explainer.explain()    # （n_SA_pairs, 3, 84, 84, n_actions）
        # 由于shap返回的是每个channel每个action的importance，所以需要选择特定动作的 SHAP 值，且在通道维度求和
        # 下面这行参考了官方aim.py的111行的写法
        # shap值: (n_samples, 3, 84, 84, n_actions) -> (n_samples, 84, 84) 选取相应的action，在通道维度求和
        all_importance_for_actions = np.array( [np.sum(all_importance[i, :, :, :, int(dataset.actions[i])], axis=0) for i in range(len(all_importance))] )        
        # 再取对应 sample_indices 的 SHAP 值
        sample_importance = all_importance_for_actions[sample_indices]
    
    # 可视化
    plot_importance_on_images(sample_observations, sample_importance, save_dir=os.path.join('plots', method))
    plot_importance_separate_from_images(sample_observations, sample_importance, save_dir=os.path.join('plots_separate', method))

    # # Print the sampled observations and their importance
    # for obs, imp in zip(sample_observations, sample_importance):
    #     print("Observation:")
    #     print(obs)
    #     print("Importance:")
    #     print(imp)
    #     print("--------------------")
    
    # evaluator = Evaluator(metric=metric, environment=environment)
    # if metric == "RIS":
    #     performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, explainer=explainer)
    # else:
    #     performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, k=k)
    # return performance













