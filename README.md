# 🧠 Gridworld Q-Learning Visualizer

A simple reinforcement learning project that demonstrates **tabular Q-learning** in a 5x5 gridworld. The agent learns to reach a goal while avoiding holes, with all learning visualized using **Streamlit**.

📍 **Try it live:**  
👉 [https://gridworld.streamlit.app/](https://gridworld.streamlit.app/)

---

## Features

- Adjustable hyperparameters: `alpha`, `gamma`, `epsilon`, and training episodes
- Plots of:
  - Steps taken per episode
  - Q-values per state (how good it is to act from a state)
  - State values (how good it is to arrive at a state, based on incoming Q-values)
- Clean 5x5 grid with color-coded heatmaps
- Streamlit UI for interactive training and exploration

---

## Outline

The agent starts in the bottom-left and learns through trial-and-error:
- 🟢 Goal at the top-left gives `+100` reward
- 🔴 Holes penalize with `-10`
- All other steps incur a `-1` cost

Learning is driven by the standard **Q-learning update rule**.

---

## Libs Used

- Python
- Streamlit
- Matplotlib & Seaborn
- Numpy

---

