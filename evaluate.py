import torch
import numpy as np
import json
import os
from env import MicroTankArenaEnv
from model import DQN
import config


def evaluate_agent(model, num_eval_episodes=50, seed_offset=1000):
    env = MicroTankArenaEnv(config)
    renderer = None
    if getattr(config, "USE_PYGAME_RENDER", False):
        from pygame_renderer import PygameRenderer
        renderer = PygameRenderer(env.grid_size)
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    hp_diffs = []

    for i in range(num_eval_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        truncated = False
        steps = 0
        ep_reward = 0.0

        while not (done or truncated):
            with torch.no_grad():
                q_values = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()

            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            ep_reward += reward

            if renderer is not None:
                hud_info = {
                    "mode": "eval-trained",
                    "episode": i + 1,
                    "step": steps,
                    "ep_reward": ep_reward,
                }
                alive = renderer.render(env, hud_info=hud_info)
                if not alive:
                    renderer = None

        total_steps += steps

        final_me_hp = env.state["me"]["hp"]
        final_enemy_hp = env.state["enemy"]["hp"]
        hp_diffs.append(final_me_hp - final_enemy_hp)

        if final_me_hp > 0 and final_enemy_hp <= 0:
            wins += 1
        elif final_me_hp <= 0:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_steps": total_steps,
        "hp_diffs": hp_diffs,
        "num_episodes": num_eval_episodes,
    }


def print_stats(title, stats):
    n = stats["num_episodes"]
    wins = stats["wins"]
    losses = stats["losses"]
    draws = stats["draws"]
    total_steps = stats["total_steps"]
    hp_diffs = stats["hp_diffs"]

    print("-" * 30)
    print(f"{title} over {n} episodes:")
    print(f"Win Rate: {wins / n * 100:.1f}%")
    print(f"Loss Rate: {losses / n * 100:.1f}%")
    print(f"Draw Rate: {draws / n * 100:.1f}%")
    print(f"Avg Steps: {total_steps / n:.1f}")
    print(f"Avg HP Diff: {np.mean(hp_diffs):.2f}")
    print("-" * 30)


def evaluate():
    if not os.path.exists(config.MODEL_CONFIG_PATH):
        print("Model config not found. Please train first.")
        return

    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    hidden_sizes = model_config["hidden_sizes"]
    obs_dim = model_config["obs_dim"]
    action_dim = model_config["action_dim"]

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Model weights not found. Please train first.")
        return

    model = DQN(obs_dim, action_dim, hidden_sizes)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.eval()

    num_eval_episodes = 50
    print(f"Evaluating trained agent over {num_eval_episodes} episodes...")
    trained_stats = evaluate_agent(model, num_eval_episodes=num_eval_episodes, seed_offset=1000)
    print_stats("Trained agent results", trained_stats)


if __name__ == "__main__":
    evaluate()


