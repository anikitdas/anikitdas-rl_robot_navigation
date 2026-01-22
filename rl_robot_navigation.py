"""
Reinforcement Learning Robot Navigation
Python 3.10 compatible
"""

import numpy as np

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2), (3, 1)]

class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = START
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0: x -= 1   # Up
        if action == 1: x += 1   # Down
        if action == 2: y -= 1   # Left
        if action == 3: y += 1   # Right

        if x < 0 or y < 0 or x >= GRID_SIZE or y >= GRID_SIZE:
            return self.state, -1, False

        if (x, y) in OBSTACLES:
            return self.state, -5, True

        self.state = (x, y)

        if self.state == GOAL:
            return self.state, 10, True

        return self.state, -0.1, False


class QLearningAgent:
    def __init__(self):
        self.q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q[state[0], state[1]])

    def learn(self, s, a, r, ns):
        self.q[s[0], s[1], a] += self.alpha * (
            r + self.gamma * np.max(self.q[ns[0], ns[1]]) - self.q[s[0], s[1], a]
        )


env = Environment()
agent = QLearningAgent()

for ep in range(1, 201):
    s = env.reset()
    total = 0

    while True:
        a = agent.choose_action(s)
        ns, r, done = env.step(a)
        agent.learn(s, a, r, ns)
        s = ns
        total += r
        if done:
            break

    print(f"Episode {ep:03d} | Total Reward: {total:.2f}")

print("\nTraining completed successfully!")

print("\nGrid Layout:")
grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
for o in OBSTACLES:
    grid[o[0]][o[1]] = "X"
grid[GOAL[0]][GOAL[1]] = "G"

for row in grid:
    print(" ".join(row))
