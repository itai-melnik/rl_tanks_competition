# MicroTank Arena Cup

Welcome to the MicroTank Arena RL Competition!

## The Goal
Train a Reinforcement Learning agent to control a tank in a small grid arena. Your tank fights against scripted opponent bots.
Your objective is to maximize your Win Rate (and secondary stats like HP difference and speed) by tweaking the **Reward Function**, **Hyperparameters**, and **Network Architecture**.

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Train your agent
```bash
python train.py
```
This will train the agent for the number of episodes specified in `config.py` and save the best model to `checkpoints/agent.pt`.

In this exercise, you will probably want to **increase `NUM_EPISODES` in `config.py`** beyond the default once you have a reward function and architecture that seem reasonable. Longer training usually helps, but be careful not to **overfit to a small set of maps or bots**: keep your `BOT_POOL` and `MAP_POOL` diverse, watch performance on fresh seeds (and new maps/bots you did not tune on), and avoid cranking `NUM_EPISODES` so high that your agent mainly memorizes the scripted opponents instead of learning a robust policy.

### 2. Evaluate your agent
```bash
python evaluate.py
```
This runs your trained agent on a fixed set of maps/seeds and reports your Win Rate and Stats.

### 3. Optional: Visualize with pygame

You can watch the tank battles in a small pygame window. This is useful to understand *how* your agent is playing, but it will slow training/evaluation, so use it for short debug runs.

1. In `config.py`, set:
   ```python
   USE_PYGAME_RENDER = True
   ```
2. Run either:
   ```bash
   python train.py
   ```
   or
   ```bash
   python evaluate.py
   ```

#### What you see in the pygame window

- The **arena** (top):
  - Gray squares are **walls**, darker squares are **floor**.
  - **Blue tank (P)** is your agent.
  - **Red tank (E)** is the scripted opponent.
  - Small colored circles are **bullets** (yellow = yours, orange = enemy).
  - A short line sticking out of each tank shows its **facing direction**.
  - When a tank's shield is active, you will see a colored ring (halo) around it.

- The **HUD bar** (bottom):
  - `P (Blue) HP` / `E (Red) HP`: current hit points for each tank, plus small bars.
  - `Cooldown`: current gun cooldown for each tank (0 = can shoot).
  - `Ammo` / `Reload`: remaining bullets before reload, and reload timer steps.
  - `Shield`: whether shield is currently on and/or the remaining shield cooldown.
  - `Mode`: current run mode label (for example `train` or `eval-trained`).
  - `Episode` / `Step`: which episode and time step you are currently watching.
  - `Epsilon`: current exploration rate during training (only shown in `train` mode).
  - `Ep Reward`: cumulative reward so far in this episode.

To go back to fast, headless training/eval, set:
```python
USE_PYGAME_RENDER = False
```

## Game Rules (Important Details)

- Tanks start with a fixed amount of HP and a finite ammo pool.
- **Ammo and reload**:
  - Each tank has a limited number of shots (`MAX_AMMO`) before it must reload.
  - When ammo reaches 0, the tank enters a reload phase for several **game steps**; during reload, shooting does nothing.
- **Anti-camping penalty**:
  - If a tank stays on the **same tile** for too many consecutive **game steps** (currently 6+), it starts to **lose HP each step** it keeps camping there.
  - This means that never moving (pure turret play) will eventually kill you; agents are encouraged to reposition. 

## Competition Rules

### What you CAN change (for training on your own machine)

- **`rewards.py`**: Modify `compute_reward` to shape the learning signal. This is the most important part.
- **`config.py`**: Tune hyperparameters like learning rate, gamma, batch size, replay buffer size, epsilon schedule, number of episodes, and whether to enable pygame.
- **`model.py`**: Change `HIDDEN_SIZES` to adjust the neural network size (constraints: max 3 layers, max 256 neurons per layer).
- **`bots.py`**: Add or modify bots and the `BOT_POOL` used during training. This changes which scripted opponents your agent trains against. If your trained tank starts consistently beating the existing bots, you should create smarter or harder bots so that your agent stays challenged and learns a policy that works well against stronger opponents too.
- **`maps.py`**: Add or modify maps in `MAP_POOL` to create different arenas for training.


All **official evaluation** (public and hidden) will be run against the **canonical codebase and settings in the instructor’s repo**, so any changes you make here only affect how you train, not how we evaluate.

### What you MUST NOT change

To avoid breaking evaluation (technical incompatibility), do **not** change:

- **`env.py`**: The environment logic (physics, observation layout, action definitions, game rules) must remain exactly as provided.
- **`spaces.py`**: The definitions of `Discrete` and `Box`.
- **Interface-shaping constants** in `constants.py`, such as:
  - `GRID_SIZE`
  - Direction constants (`DIR_UP`, `DIR_RIGHT`, `DIR_DOWN`, `DIR_LEFT`) and the `DX`, `DY` vectors  
  Changing these would break the observation space or game geometry.
- **`train.py`, `evaluate.py`, `evaluate_hidden.py`**: The training loop, evaluation logic, and hidden evaluation runner. Treat these as fixed infrastructure.
- **`pygame_renderer.py`**: The visualization code (you can enable/disable it via `config.USE_PYGAME_RENDER`, but do not edit the file).
- **File layout / names expected for submission**: Follow the submission instructions below exactly so that your model can be evaluated automatically.

### Version control and GitHub repository

- You must keep your work for this assignment in a **private Github repository** 
- Every change you make to the code should be committed with a clear commit message.
- Push all commits to your private remote repository regularly.
- Invite the instructor to your private repository

### Submission
You will submit only your **trained weights**, **model config**, and **personal details** (no code).

Each student must submit a **single zip file** named:

- `lastname_firstname.zip`  
  - Example: `doe_jane.zip`

Inside that zip, there should be **exactly three files**, all in the top level (no folders):

1. `lastname_firstname_agent.pt`  
2. `lastname_firstname_model_config.json`  
3. `lastname_firstname_student.json`

Example contents for Jane Doe:

- `doe_jane_agent.pt`
- `doe_jane_model_config.json`
- `doe_jane_student.json`

The `*_agent.pt` and `*_model_config.json` files should be the ones produced by `train.py`.  
The `*_student.json` file must contain your details in this format:

{
  "first_name": "Jane",
  "last_name": "Doe",
  "email": "jane.doe@post.runi.ac.il"
}


We will run your weights against a **Hidden Evaluation Set** (new maps, hidden seeds, mixed opponents).

## Scoring Formula
Your score is calculated as:
```python
Score = (WinRate * 100) + (AvgHPDiff * 5) - (AvgLength * 0.05)
```
- **WinRate**: Percentage of matches won (0.0 to 1.0).
- **AvgHPDiff**: (My Final HP - Enemy Final HP). Higher is better.
- **AvgLength**: Average steps taken to win. Lower is better (faster wins).


Good luck, Commander!


