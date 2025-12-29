"""
Evaluate agent specifically against AggressiveBot on all maps.
This isolates the hardest opponent to measure true combat ability.
"""
import torch
import numpy as np
import json
import os
from env import MicroTankArenaEnv
from model import DQN
from bots import AggressiveBot
import config


class AggressiveBotOnlyEnv(MicroTankArenaEnv):
    """Environment that ONLY uses AggressiveBot as opponent."""
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Override bot selection - always use AggressiveBot
        self.bot = AggressiveBot()
        return obs, info


def evaluate_against_aggressive(model, num_eval_episodes=50, seed_offset=2000):
    """Evaluate agent specifically against AggressiveBot."""
    env = AggressiveBotOnlyEnv(config)
    
    renderer = None
    if getattr(config, "USE_PYGAME_RENDER", False):
        from pygame_renderer import PygameRenderer
        renderer = PygameRenderer(env.grid_size)
    
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    hp_diffs = []
    map_results = {}  # Track results per map

    for i in range(num_eval_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        map_name = env.map_name
        
        if map_name not in map_results:
            map_results[map_name] = {"wins": 0, "losses": 0, "draws": 0, "count": 0}
        map_results[map_name]["count"] += 1
        
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
                    "mode": "eval-aggressive",
                    "episode": i + 1,
                    "step": steps,
                    "map": map_name,
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
            map_results[map_name]["wins"] += 1
        elif final_me_hp <= 0:
            losses += 1
            map_results[map_name]["losses"] += 1
        else:
            draws += 1
            map_results[map_name]["draws"] += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_steps": total_steps,
        "hp_diffs": hp_diffs,
        "num_episodes": num_eval_episodes,
        "map_results": map_results,
    }


def main():
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

    num_eval_episodes = 100  # More episodes for statistical significance
    print(f"\n{'='*50}")
    print(f"Evaluating against AGGRESSIVEBOT ONLY")
    print(f"Episodes: {num_eval_episodes}")
    print(f"{'='*50}\n")
    
    stats = evaluate_against_aggressive(model, num_eval_episodes=num_eval_episodes)
    
    n = stats["num_episodes"]
    wins = stats["wins"]
    losses = stats["losses"]
    draws = stats["draws"]
    
    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS vs AggressiveBot")
    print(f"{'='*50}")
    print(f"Win Rate:  {wins / n * 100:5.1f}%  ({wins}/{n})")
    print(f"Loss Rate: {losses / n * 100:5.1f}%  ({losses}/{n})")
    print(f"Draw Rate: {draws / n * 100:5.1f}%  ({draws}/{n})")
    print(f"Avg Steps: {stats['total_steps'] / n:.1f}")
    print(f"Avg HP Diff: {np.mean(stats['hp_diffs']):.1f}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS BY MAP")
    print(f"{'='*50}")
    for map_name, results in sorted(stats["map_results"].items()):
        count = results["count"]
        if count > 0:
            win_rate = results["wins"] / count * 100
            print(f"{map_name:12s}: {win_rate:5.1f}% wins ({results['wins']}/{count})")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()

