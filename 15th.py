import numpy as np
import random

# Define the environment
class MarketEnv:
    def __init__(self, n_competitors, n_price_levels):
        self.n_competitors = n_competitors
        self.n_price_levels = n_price_levels
        self.state = self.reset()

    def reset(self):
        # State is a tuple of current price level and competitors' average price level
        self.state = (random.randint(0, self.n_price_levels - 1), 
                      random.randint(0, self.n_price_levels - 1))
        return self.state

    def step(self, action):
        # Action is the new price level chosen by the agent
        price_level = action
        competitors_price_level = random.randint(0, self.n_price_levels - 1)
        
        # Simulate reward based on action and competitors' price
        if price_level < competitors_price_level:
            reward = 10  # High sales due to lower price
        elif price_level == competitors_price_level:
            reward = 5  # Moderate sales due to same price
        else:
            reward = 1  # Low sales due to higher price
        
        next_state = (price_level, competitors_price_level)
        return next_state, reward, False

# Initialize Q-Learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.n_price_levels, env.n_price_levels, env.n_price_levels))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.env.n_price_levels - 1)
        else:
            current_price_level, competitors_price_level = state
            return np.argmax(self.q_table[current_price_level, competitors_price_level])

    def learn(self, state, action, reward, next_state):
        current_price_level, competitors_price_level = state
        next_price_level, next_competitors_price_level = next_state
        best_next_action = np.argmax(self.q_table[next_price_level, next_competitors_price_level])
        td_target = reward + self.discount_factor * self.q_table[next_price_level, next_competitors_price_level, best_next_action]
        td_error = td_target - self.q_table[current_price_level, competitors_price_level, action]
        self.q_table[current_price_level, competitors_price_level, action] += self.learning_rate * td_error

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

# Main function to load data, train the model, infer probabilities, and update the model
def main():
    n_competitors = 3
    n_price_levels = 10
    env = MarketEnv(n_competitors, n_price_levels)
    
    # Train the Q-Learning agent
    agent = QLearningAgent(env)
    agent.train(episodes=5000)
    
    # Evaluate the agent
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    
    print("Total reward after training:", total_reward)

if __name__ == "__main__":
    main()
