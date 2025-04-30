import numpy as np
import matplotlib.pyplot as plt
import random

# Define the environment
class DroneEnv:
    def __init__(self, grid_size, obstacles, start, goal):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        next_state = (x, y)

        if (x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or next_state in self.obstacles):
            reward = -1
            next_state = self.state  # reset to previous state if hit obstacle or out of bounds
        elif next_state == self.goal:
            reward = 10
        else:
            reward = -0.1  # small negative reward for each step to encourage efficiency

        self.state = next_state
        return next_state, reward, next_state == self.goal

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for obs in self.obstacles:
            grid[obs] = -1
        grid[self.goal] = 1
        grid[self.state] = 0.5
        print(grid)

# Initialize Q-Learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.discount_factor * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.learning_rate * td_error

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            self.exploration_rate *= self.exploration_decay

# Set up the environment
grid_size = 5
obstacles = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)]
start = (0, 0)
goal = (4, 4)
env = DroneEnv(grid_size, obstacles, start, goal)

# Train the Q-Learning agent
agent = QLearningAgent(env)
agent.train(episodes=500)

# Evaluate the agent
state = env.reset()
env.render()
done = False
while not done:
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    env.render()
    state = next_state

print("Training complete.")
