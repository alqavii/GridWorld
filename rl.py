import numpy as np
import random
import matplotlib.pyplot as plt

board = [24, 23, 22, 21, 20,
         19, 18, 17, 16, 15,
         14, 13, 12, 11, 10,
         9,  8,  7,  6,  5,
         4,  3,  2,  1,  0]

ACTIONS = ["up","down","left","right"]
holes = [18, 17, 13, 12]





class bot:
    def __init__(self):
        self.position = 0
        self.holes = set(holes)
        self.goal = 24
        self.qtable = {s: [0]*4 for s in board}

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.position = None
        self.cumulative_reward = 0
        self.steps = 0

        self.topedge = [24, 23, 22, 21, 20]
        self.bottomedge = [4, 3, 2, 1, 0]
        self.leftedge = [24, 19, 14, 9, 4]
        self.rightedge = [20, 15, 10, 5, 0]

    def invalidActions(self, state):
        invalid = []
        if state in self.topedge:
            invalid.append(0)
        if state in self.bottomedge:
            invalid.append(1)
        if state in self.leftedge:
            invalid.append(2)
        if state in self.rightedge:
            invalid.append(3)
        return invalid

    def chooseAction(self, state): 
        invalid = self.invalidActions(state)
        if state == self.goal:
            return None
        if np.random.rand() < self.epsilon:
            return np.random.choice([a for a in ACTIONS if ACTIONS.index(a) not in invalid]), "Exploration"
        if len([i for i in range(len(self.qtable[state])) if i not in invalid and self.qtable[state][i] == max(self.qtable[state])]) > 1:
            return np.random.choice([a for a in ACTIONS if self.qtable[state][ACTIONS.index(a)] == max(self.qtable[state]) and ACTIONS.index(a) not in invalid]), "Dup"
        best = max((i for i in range(len(self.qtable[state]))), key=lambda x: self.qtable[state][x] if x not in invalid else -np.inf)
        return ACTIONS[best], "Exploitation"

    def step(self, action):
        done = False
        new = self.move(action)
        self.steps += 1
        reward = -1 
        if new in self.holes:
            reward = -10
        elif new == self.goal:
            reward = 100
            done = True
        return new, reward, done
    
    def move(self, action):
        if action == "up":
            self.position += 5
        elif action == "down":
            self.position -= 5
        elif action == "left":
            self.position += 1
        elif action == "right":
            self.position -= 1
        return self.position

    def updateQTable(self, state, action, reward, next_state):
        action_index = ACTIONS.index(action)
        bestNextActionIndex = max((i for i in range(len(self.qtable[next_state]))), key=lambda x: self.qtable[next_state][x] if x not in self.invalidActions(next_state) else -np.inf)
        td_target = reward + self.gamma * self.qtable[next_state][bestNextActionIndex]
        td_error = td_target - self.qtable[state][action_index]
        self.qtable[state][action_index] += self.alpha * td_error
    
    def train(self, episodes=1000, max_steps=10000, decay_epsilon=False):
        stepstaken = []
        self.epsilon = 0.1  # Exploration rate
        for episode in range(episodes):
            self.steps = 0
            self.position = 0
            self.cumulative_reward = 0
            if decay_epsilon:
                self.epsilon *= 0.99  # Decay epsilon
            done = False
            while not done and self.steps < max_steps:
                state = self.position
                action, exploration_type = self.chooseAction(self.position)
                #print(f"Q-Table[{self.position}][{action}] = {self.qtable[self.position]}, Exploration Type: {exploration_type}")
                if action is None:
                    break
                next_state, reward, done = self.step(action)
                self.cumulative_reward += reward
                self.updateQTable(state, action, reward, next_state)
                self.position = next_state
                

            stepstaken.append(self.steps)
            print(f"Episode {episode + 1}: Steps = {self.steps}, Cumulative Reward = {self.cumulative_reward}, Done = {done}")
        plt.plot(stepstaken)
        plt.xlabel('Episode')
        plt.ylabel('Steps Taken')
        plt.show()

bot = bot()
bot.goal = 24
bot.train(episodes=1000, max_steps=1000, decay_epsilon=True)
