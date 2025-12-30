"""
Evaluate agent against UNSEEN bots AND UNSEEN maps.
Tests true generalization to never-before-seen scenarios.
"""
import torch
import numpy as np
import json
import os
from env import MicroTankArenaEnv
from model import DQN
from bots import SniperBot, DodgerBot, AggressiveBot
from maps import MAP_POOL_UNSEEN
import config


class SpecificBotEnv(MicroTankArenaEnv):
    """Environment that uses a specific bot as opponent."""
    
    def __init__(self, config, bot_class):
        super().__init__(config)
        self.bot_class = bot_class
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Override bot selection
        self.bot = self.bot_class()
        return obs, info


class UnseenMapEnv(MicroTankArenaEnv):
    """Environment that uses UNSEEN maps only."""
    
    def __init__(self, config, bot_class=None):
        super().__init__(config)
        self.bot_class = bot_class
        self.unseen_map_names = list(MAP_POOL_UNSEEN.keys())
    
    def reset(self, seed=None, options=None):
        # Call parent reset to handle seeding and state init
        obs, info = super().reset(seed=seed, options=options)
        
        # Override map with unseen map
        map_idx = self.rng.integers(0, len(self.unseen_map_names))
        self.map_name = self.unseen_map_names[map_idx]
        self.current_map = MAP_POOL_UNSEEN[self.map_name]
        self.state['grid'] = self.current_map
        
        # Re-spawn tanks on new map
        self.state = self._spawn_tanks()
        self.state['bullets'] = []
        self.state['grid'] = self.current_map
        
        # Override bot if specified
        if self.bot_class:
            self.bot = self.bot_class()
        
        return self._get_obs(), info


def evaluate_against_bot(model, bot_class, bot_name, num_episodes=100, seed_offset=3000, use_unseen_maps=False):
    """Evaluate agent against a specific bot type."""
    if use_unseen_maps:
        env = UnseenMapEnv(config, bot_class)
    else:
        env = SpecificBotEnv(config, bot_class)
    
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    hp_diffs = []
    map_results = {}

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        map_name = env.map_name
        
        if map_name not in map_results:
            map_results[map_name] = {"wins": 0, "losses": 0, "draws": 0, "count": 0}
        map_results[map_name]["count"] += 1
        
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            with torch.no_grad():
                q_values = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()

            obs, reward, done, truncated, info = env.step(action)
            steps += 1

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
        "bot_name": bot_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_steps": total_steps,
        "hp_diffs": hp_diffs,
        "num_episodes": num_episodes,
        "map_results": map_results,
    }


def print_results(stats):
    n = stats["num_episodes"]
    wins = stats["wins"]
    losses = stats["losses"]
    draws = stats["draws"]
    
    print(f"\n{'='*50}")
    print(f"RESULTS vs {stats['bot_name']}")
    print(f"{'='*50}")
    print(f"Win Rate:  {wins / n * 100:5.1f}%  ({wins}/{n})")
    print(f"Loss Rate: {losses / n * 100:5.1f}%  ({losses}/{n})")
    print(f"Draw Rate: {draws / n * 100:5.1f}%  ({draws}/{n})")
    print(f"Avg Steps: {stats['total_steps'] / n:.1f}")
    print(f"Avg HP Diff: {np.mean(stats['hp_diffs']):.1f}")
    
    print(f"\nBy Map:")
    for map_name, results in sorted(stats["map_results"].items()):
        count = results["count"]
        if count > 0:
            win_rate = results["wins"] / count * 100
            print(f"  {map_name:12s}: {win_rate:5.1f}% wins ({results['wins']}/{count})")


def main():
    if not os.path.exists(config.MODEL_CONFIG_PATH):
        print("Model config not found. Please train first.")
        return

    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Model weights not found. Please train first.")
        return

    model = DQN(
        model_config["obs_dim"],
        model_config["action_dim"],
        model_config["hidden_sizes"]
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.eval()

    num_episodes = 100
    
    print(f"\n{'#'*60}")
    print(f"#  EVALUATING AGAINST UNSEEN BOTS AND MAPS")
    print(f"#  Testing TRUE generalization to never-seen scenarios")
    print(f"#  Episodes per test: {num_episodes}")
    print(f"{'#'*60}")
    
    all_results = []
    
    # ===== PART 1: UNSEEN BOTS on KNOWN MAPS =====
    print(f"\n{'='*60}")
    print("PART 1: UNSEEN BOTS on KNOWN MAPS")
    print(f"{'='*60}")
    
    unseen_bots = [
        (SniperBot, "SniperBot"),
        (DodgerBot, "DodgerBot"),
    ]
    
    for bot_class, bot_name in unseen_bots:
        stats = evaluate_against_bot(model, bot_class, f"{bot_name} (unseen bot)", num_episodes, use_unseen_maps=False)
        all_results.append(stats)
        print_results(stats)
    
    # ===== PART 2: KNOWN BOTS on UNSEEN MAPS =====
    print(f"\n{'='*60}")
    print("PART 2: KNOWN BOTS on UNSEEN MAPS")
    print(f"(Maps: corridors, l_walls, scattered, ring)")
    print(f"{'='*60}")
    
    known_bots = [
        (AggressiveBot, "AggressiveBot"),
    ]
    
    for bot_class, bot_name in known_bots:
        stats = evaluate_against_bot(model, bot_class, f"{bot_name} (unseen maps)", num_episodes, seed_offset=4000, use_unseen_maps=True)
        all_results.append(stats)
        print_results(stats)
    
    # ===== PART 3: UNSEEN BOTS on UNSEEN MAPS (Hardest!) =====
    print(f"\n{'='*60}")
    print("PART 3: UNSEEN BOTS on UNSEEN MAPS (HARDEST TEST)")
    print(f"{'='*60}")
    
    for bot_class, bot_name in unseen_bots:
        stats = evaluate_against_bot(model, bot_class, f"{bot_name} (FULL UNSEEN)", num_episodes, seed_offset=5000, use_unseen_maps=True)
        all_results.append(stats)
        print_results(stats)
    
    # ===== PART 4: BASELINE =====
    print(f"\n{'='*60}")
    print("BASELINE: AggressiveBot on KNOWN MAPS")
    print(f"{'='*60}")
    
    stats = evaluate_against_bot(model, AggressiveBot, "AggressiveBot (baseline)", num_episodes, seed_offset=6000, use_unseen_maps=False)
    all_results.append(stats)
    print_results(stats)
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test Scenario':<40} {'Win%':>8} {'Loss%':>8} {'Draw%':>8}")
    print("-" * 70)
    for stats in all_results:
        n = stats["num_episodes"]
        print(f"{stats['bot_name']:<40} {stats['wins']/n*100:>7.1f}% {stats['losses']/n*100:>7.1f}% {stats['draws']/n*100:>7.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

