import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

board = [24, 23, 22, 21, 20,
         19, 18, 17, 16, 15,
         14, 13, 12, 11, 10,
         9,  8,  7,  6,  5,
         4,  3,  2,  1,  0]

ACTIONS = ["up", "down", "left", "right"]
holes = [18, 17, 13, 12]

class tabQ:
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
        self.stepsTaken = []

        self.topedge = [24, 23, 22, 21, 20]
        self.bottomedge = [4, 3, 2, 1, 0]
        self.leftedge = [24, 19, 14, 9, 4]
        self.rightedge = [20, 15, 10, 5, 0]

        self.trained = False

    def invalidActions(self, state):
        invalid = []
        if state in self.topedge: invalid.append(0)
        if state in self.bottomedge: invalid.append(1)
        if state in self.leftedge: invalid.append(2)
        if state in self.rightedge: invalid.append(3)
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
        if action == "up": self.position += 5
        elif action == "down": self.position -= 5
        elif action == "left": self.position += 1
        elif action == "right": self.position -= 1
        return self.position

    def updateQTable(self, state, action, reward, next_state):
        action_index = ACTIONS.index(action)
        bestNextActionIndex = max((i for i in range(len(self.qtable[next_state]))), key=lambda x: self.qtable[next_state][x] if x not in self.invalidActions(next_state) else -np.inf)
        td_target = reward + self.gamma * self.qtable[next_state][bestNextActionIndex]
        td_error = td_target - self.qtable[state][action_index]
        self.qtable[state][action_index] += self.alpha * td_error
    
    def train(self, episodes=1000, max_steps=10000, decay_epsilon=False):
        self.epsilon = 0.1
        self.stepsTaken = []
        for episode in range(episodes):
            self.steps = 0
            self.position = 0
            self.cumulative_reward = 0
            if decay_epsilon:
                self.epsilon *= 0.99
            done = False
            while not done and self.steps < max_steps:
                state = self.position
                action, exploration_type = self.chooseAction(self.position)
                if action is None:
                    break
                next_state, reward, done = self.step(action)
                self.cumulative_reward += reward
                self.updateQTable(state, action, reward, next_state)
                self.position = next_state
            self.stepsTaken.append(self.steps)
        self.trained = True

    def avg_q_grid(self):
        grid = np.zeros((5, 5))
        for s in range(25):
            x = 4 - (s % 5)
            y = 4 - (s // 5)
            grid[y][x] = np.mean(self.qtable[s])
        return grid


    def state_value_grid(self):
        grid = np.zeros((5, 5))

        def move_preview(pos, action):
            if action == "up" and pos not in self.topedge: return pos + 5
            if action == "down" and pos not in self.bottomedge: return pos - 5
            if action == "left" and pos not in self.leftedge: return pos + 1
            if action == "right" and pos not in self.rightedge: return pos - 1
            return None

        for s_from in range(25):
            for a_idx, action in enumerate(ACTIONS):
                s_to = move_preview(s_from, action)
                if s_to is None or s_to in self.holes: continue
                x = 4 - (s_to % 5)
                y = 4 - (s_to // 5)
                grid[y][x] += self.qtable[s_from][a_idx]

        return grid


# --- Streamlit App ---
st.title("🤖 Q-learning Gridworld Agent")

alpha = st.slider("Learning rate (α)", 0.01, 1.0, 0.1, 0.01)
gamma = st.slider("Discount factor (γ)", 0.1, 1.0, 0.9, 0.01)
epsilon = st.slider("Exploration rate (ε)", 0.0, 1.0, 0.1, 0.01)
episodes = st.slider("Training Episodes", 10, 1000, 250, 10)
decay = st.checkbox("Decay Epsilon", value=True)

if st.button("🚀 Train Agent"):
    self = tabQ()
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.train(episodes=episodes, max_steps=1000, decay_epsilon=decay)

    st.subheader("📈 Steps per Episode")
    fig1, ax1 = plt.subplots()
    ax1.plot(self.stepsTaken)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Steps Taken")
    ax1.set_title("Learning Curve")
    st.pyplot(fig1)

    # --- Average Q-value Heatmap ---
    st.subheader("🗺️ Average Q-Value per State")
    qgrid = self.avg_q_grid()
    fig2, ax2 = plt.subplots()
    sns.heatmap(qgrid, annot=True, cmap="YlGnBu", square=True, cbar=True, linewidths=0.3, linecolor='black', ax=ax2)
    ax2.set_title("Top Left: Target, Bottom Right: Start")
    st.pyplot(fig2)

    # --- State Value Heatmap ---
    st.subheader("🌀 State Value per Cell (Incoming Q-values)")
    state_val_grid = self.state_value_grid()
    fig3, ax3 = plt.subplots()
    sns.heatmap(state_val_grid, annot=True, cmap="coolwarm", square=True, cbar=True, linewidths=0.3, linecolor='black', ax=ax3)
    ax3.set_title("Higher = Better")
    st.pyplot(fig3)
