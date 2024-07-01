import gymnasium as gym
import torch
import os
import numpy as np
from collections import deque
from d3rlpy.dataset import MDPDataset
from xrlbench.custom_environment.breakout.agent import Agent
import torchvision.transforms as T

"""
测试环境提供的预训练模型的性能。
发现很差，需要重新训练。
执行以下代码，重新训练模型。
在这里手动停止：
Episode 143903 Average Score: 6.67
"""

def preprocess_state(state):
    return T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.ToTensor()])(state).unsqueeze(0)

class BreakOut:
    def __init__(self, env_id="Breakout-v0"):
        self.env = gym.make(env_id)
        self.agent = Agent(action_size=self.env.action_space.n)
        self.model = self.agent.policy_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_states = None
        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(os.path.join(".", "model", "BreakOut_new.pth"), map_location=self.device))
            print("Loaded model successfully.")
        except:
            print("This model is not existing, please train it.")

    def train_model(self, n_episodes=50000, max_t=10000, ending_score=16):
        """
        Train the reinforcement learning agent.

        Parameters:
        -----------
        n_episodes : int, optional
            The maximum number of episodes to train for. (default=500000)
        max_t : int, optional
            The maximum number of timesteps per episode. (default=10000)
        ending_score : float, optional
            The average score at which to consider the environment solved. (default=16)
        """
        scores_window = deque(maxlen=100)
        for i in range(1, n_episodes + 1):
            train = len(self.agent.replay_buffer) > 5000
            score = 0
            state = preprocess_state(self.env.reset()[0])
            for t in range(max_t):
                action = self.agent.act(np.array(state), train)
                next_state, reward, done, _, _ = self.env.step(action)
                # print(next_state, reward, done)
                next_state = preprocess_state(next_state)
                self.agent.replay_buffer.add(state, action, reward, next_state, done)
                self.agent.t_step += 1
                if self.agent.t_step % self.agent.policy_update == 0:
                    self.agent.optimize_model(train)

                if self.agent.t_step % self.agent.target_update == 0:
                    self.agent.target_network.load_state_dict(self.agent.policy_network.state_dict())
                state = next_state
                score += reward
                if done:
                    print("score:", score, "t:", t)
                    break
            scores_window.append(score)
            print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}', end='')
            if i % 100 == 0:
                print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}')
            if np.mean(scores_window) >= ending_score:
                print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i, np.mean(scores_window)))
                break
        torch.save(self.agent.policy_network.state_dict(), os.path.join(".", "model", "BreakOut_new.pth"))
        return self.agent.policy_network

    def evaluate_model(self, n_episodes=5):
        """
        Evaluate the pretrained model.

        Parameters:
        -----------
        n_episodes : int
            The number of episodes to evaluate the model.
        """
        total_score = 0
        for i in range(n_episodes):
            step = 0
            state = preprocess_state(self.env.reset()[0])
            done = False
            score = 0
            while not done:
                action = self.agent.act(np.array(state))
                step += 1
                # next_state, reward, done, _, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # print(next_state, reward, done)
                next_state = preprocess_state(next_state)
                state = next_state
                score += reward
            total_score += score
            print(f'Episode {i+1}: Score {score} in {step} steps.')
        average_score = total_score / n_episodes
        print(f'Average Score over {n_episodes} episodes: {average_score}')

# 实例化并运行评估
breakout = BreakOut()
breakout.evaluate_model(n_episodes=20)
breakout.train_model(n_episodes=1000000,ending_score=20)  # 可以调整n_episodes来指定训练的回合数