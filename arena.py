"""
Arena: Make two trained agents compete against each other.
Usage: 
    python arena.py                                    # Default matchup
    python arena.py agent1.pt agent2.pt                # Custom agents
    
Set config.USE_PYGAME_RENDER = True to watch matches.
"""
import torch
import numpy as np
import json
import os
import argparse
from env import MicroTankArenaEnv
from model import DQN
from constants import *
import config


class AgentBot:
    """Bot wrapper for a trained agent - plays from enemy's perspective."""
    
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.name = os.path.basename(model_path)
        
        config_path = "checkpoints/model_config.json"
        if os.path.exists(model_path) and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            self.model = DQN(
                model_config["obs_dim"],
                model_config["action_dim"],
                model_config["hidden_sizes"]
            )
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"Loaded agent: {self.name}")
        else:
            print(f"ERROR: Could not load {model_path}")
    
    def act(self, state, rng):
        """Act from this agent's perspective (as enemy in env state)."""
        if self.model is None:
            return ACTION_DO_NOTHING
        
        obs = self._state_to_obs(state)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(obs_tensor)
            return q_values.argmax().item()
    
    def _state_to_obs(self, state):
        """Convert state to observation from this bot's perspective (as enemy)."""
        me = state['enemy']      # This agent is the "enemy" in env terms
        enemy = state['me']      # The opponent is "me" in env terms
        grid = state['grid']
        bullets = state['bullets']
        
        gs = GRID_SIZE - 1.0
        obs = []
        
        # Me (this agent)
        obs.extend([me['x']/gs, me['y']/gs])
        d_onehot = [0]*4
        d_onehot[me['dir']] = 1
        obs.extend(d_onehot)
        
        me_hp = max(0, min(MAX_HP, me['hp']))
        obs.append(me_hp/float(MAX_HP))
        obs.append(me.get('cooldown', 0)/float(GUN_COOLDOWN_STEPS))
        obs.append(me.get('ammo', MAX_AMMO)/float(MAX_AMMO))
        obs.append(me.get('reload', 0)/float(RELOAD_STEPS))
        me_shield_active = 1.0 if me.get('shield_steps', 0) > 0 else 0.0
        me_shield_cd = me.get('shield_cooldown', 0)
        obs.append(me_shield_active)
        obs.append(me_shield_cd/float(SHIELD_COOLDOWN_STEPS))
        
        # Enemy (opponent)
        obs.extend([enemy['x']/gs, enemy['y']/gs])
        e_d_onehot = [0]*4
        e_d_onehot[enemy['dir']] = 1
        obs.extend(e_d_onehot)
        enemy_hp = max(0, min(MAX_HP, enemy['hp']))
        obs.append(enemy_hp/float(MAX_HP))
        enemy_shield_active = 1.0 if enemy.get('shield_steps', 0) > 0 else 0.0
        obs.append(enemy_shield_active)
        
        # Relative
        obs.append((enemy['x'] - me['x']) / gs)
        obs.append((enemy['y'] - me['y']) / gs)
        dist = abs(enemy['x'] - me['x']) + abs(enemy['y'] - me['y'])
        obs.append(dist / (2*gs))
        
        # Raycasts
        obs.extend(self._raycast(me['x'], me['y'], me['dir'], enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']-1)%4, enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']+1)%4, enemy, grid))
        
        # Bullet danger (fear bullets from 'me' which is the opponent)
        danger_front, danger_left, danger_right = self._compute_bullet_danger(
            me, bullets, grid, owner_to_fear='me'
        )
        obs.extend([danger_front, danger_left, danger_right])
        
        return np.array(obs, dtype=np.float32)
    
    def _raycast(self, x, y, direction, enemy, grid):
        cx, cy = x, y
        dist = 0
        found_enemy = False
        
        while True:
            cx += DX[direction]
            cy += DY[direction]
            dist += 1
            
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                break
            if grid[cy, cx] == 1:
                break
            if cx == enemy['x'] and cy == enemy['y']:
                found_enemy = True
                break
        
        return [dist / GRID_SIZE, 1.0 if found_enemy else 0.0]
    
    def _compute_bullet_danger(self, me, bullets, grid, owner_to_fear):
        mx, my = me['x'], me['y']
        danger_front = 0.0
        danger_left = 0.0
        danger_right = 0.0
        max_dist = max(1.0, GRID_SIZE - 1.0)
        
        for b in bullets:
            if b.get('owner') != owner_to_fear:
                continue
            
            bx, by = b['x'], b['y']
            dx = bx - mx
            dy = by - my
            
            if dx != 0 and dy != 0:
                continue
            
            blocked = False
            dist = None
            dir_to_bullet = None
            
            if b['dir'] == DIR_RIGHT:
                if dy != 0 or dx >= 0:
                    continue
                dist = -dx
                for check_x in range(bx + 1, mx):
                    if grid[my, check_x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_LEFT
            elif b['dir'] == DIR_LEFT:
                if dy != 0 or dx <= 0:
                    continue
                dist = dx
                for check_x in range(mx + 1, bx):
                    if grid[my, check_x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_RIGHT
            elif b['dir'] == DIR_DOWN:
                if dx != 0 or dy >= 0:
                    continue
                dist = -dy
                for check_y in range(by + 1, my):
                    if grid[check_y, mx] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_UP
            elif b['dir'] == DIR_UP:
                if dx != 0 or dy <= 0:
                    continue
                dist = dy
                for check_y in range(my + 1, by):
                    if grid[check_y, mx] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_DOWN
            else:
                continue
            
            if blocked or dist is None or dist <= 0:
                continue
            
            rel = (dir_to_bullet - me['dir']) % 4
            
            if rel == 0:
                slot = 'front'
            elif rel == 1:
                slot = 'right'
            elif rel == 3:
                slot = 'left'
            else:
                continue
            
            danger = max(0.0, 1.0 - float(dist) / max_dist)
            
            if slot == 'front' and danger > danger_front:
                danger_front = danger
            elif slot == 'left' and danger > danger_left:
                danger_left = danger
            elif slot == 'right' and danger > danger_right:
                danger_right = danger
        
        return danger_front, danger_left, danger_right


class ArenaEnv(MicroTankArenaEnv):
    """Environment for agent vs agent battles."""
    
    def __init__(self, cfg, agent2_bot):
        super().__init__(cfg)
        self.agent2_bot = agent2_bot
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Override bot with agent2
        self.bot = self.agent2_bot
        return obs, info


def run_arena(agent1_path, agent2_path, num_episodes=100):
    """
    Run arena battle between two agents.
    Uses config.USE_PYGAME_RENDER to control rendering.
    """
    
    should_render = getattr(config, "USE_PYGAME_RENDER", False)
    
    # Load agents
    print(f"\n{'='*60}")
    print("ARENA: Agent vs Agent Battle")
    print(f"{'='*60}")
    
    # Agent 1 plays as "me" in the environment
    config_path = "checkpoints/model_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    
    agent1 = DQN(
        model_config["obs_dim"],
        model_config["action_dim"],
        model_config["hidden_sizes"]
    )
    agent1.load_state_dict(torch.load(agent1_path, map_location='cpu'))
    agent1.eval()
    agent1_name = os.path.basename(agent1_path)
    print(f"Agent 1 (Blue): {agent1_name}")
    
    # Agent 2 plays as "enemy" via AgentBot wrapper
    agent2_bot = AgentBot(agent2_path)
    agent2_name = agent2_bot.name
    print(f"Agent 2 (Red):  {agent2_name}")
    
    # Create environment
    env = ArenaEnv(config, agent2_bot)
    
    # Setup pygame renderer (same as train.py and evaluate.py)
    renderer = None
    if should_render:
        from pygame_renderer import PygameRenderer
        renderer = PygameRenderer(env.grid_size)
        print("Rendering: ON (config.USE_PYGAME_RENDER = True)")
    else:
        print("Rendering: OFF (set config.USE_PYGAME_RENDER = True to watch)")
    
    # Track results
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    map_results = {}
    
    print(f"\nBattling for {num_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=7000 + ep)
        map_name = env.map_name
        
        if map_name not in map_results:
            map_results[map_name] = {"agent1": 0, "agent2": 0, "draw": 0, "count": 0}
        map_results[map_name]["count"] += 1
        
        done = False
        truncated = False
        
        while not (done or truncated):
            with torch.no_grad():
                q_values = agent1(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()
            
            obs, _, done, truncated, _ = env.step(action)
            
            # Render frame if pygame is enabled
            if renderer is not None:
                hud_info = {
                    "episode": ep + 1,
                    "map": map_name,
                    "agent1": agent1_name,
                    "agent2": agent2_name,
                    "score": f"{agent1_wins}-{agent2_wins}",
                }
                alive = renderer.render(env, hud_info=hud_info)
                if not alive:
                    renderer = None  # Window closed
        
        # Determine winner
        agent1_hp = env.state["me"]["hp"]
        agent2_hp = env.state["enemy"]["hp"]
        
        if agent1_hp > 0 and agent2_hp <= 0:
            agent1_wins += 1
            map_results[map_name]["agent1"] += 1
        elif agent2_hp > 0 and agent1_hp <= 0:
            agent2_wins += 1
            map_results[map_name]["agent2"] += 1
        else:
            draws += 1
            map_results[map_name]["draw"] += 1
        
        # Progress indicator
        if (ep + 1) % 25 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: {agent1_name} {agent1_wins} - {agent2_wins} {agent2_name} ({draws} draws)")
    
    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n{agent1_name} vs {agent2_name}")
    print(f"\n  {agent1_name}:  {agent1_wins} wins ({agent1_wins/num_episodes*100:.1f}%)")
    print(f"  {agent2_name}:  {agent2_wins} wins ({agent2_wins/num_episodes*100:.1f}%)")
    print(f"  Draws:       {draws} ({draws/num_episodes*100:.1f}%)")
    
    print("\nResults by Map:")
    for map_name, results in sorted(map_results.items()):
        print(f"  {map_name:12s}: {agent1_name} {results['agent1']:2d} - {results['agent2']:2d} {agent2_name} ({results['draw']} draws)")
    
    # Winner announcement
    print(f"\n{'='*60}")
    if agent1_wins > agent2_wins:
        print(f"🏆 WINNER: {agent1_name}")
    elif agent2_wins > agent1_wins:
        print(f"🏆 WINNER: {agent2_name}")
    else:
        print("🤝 TIE!")
    print(f"{'='*60}\n")
    
    return agent1_wins, agent2_wins, draws


def main():
    parser = argparse.ArgumentParser(
        description="Arena: Make two trained agents compete against each other",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arena.py                                        # Default matchup
  python arena.py checkpoints/agent.pt checkpoints/championv5.pt
  python arena.py checkpoints/championv4.pt checkpoints/championv6.pt -n 50

To watch matches, set USE_PYGAME_RENDER = True in config.py
        """
    )
    
    parser.add_argument("agent1", nargs="?", default="checkpoints/championv4.pt",
                        help="Path to first agent (default: checkpoints/championv4.pt)")
    parser.add_argument("agent2", nargs="?", default="checkpoints/championv6.pt",
                        help="Path to second agent (default: checkpoints/championv6.pt)")
    parser.add_argument("-n", "--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.agent1):
        print(f"ERROR: {args.agent1} not found")
        return
    if not os.path.exists(args.agent2):
        print(f"ERROR: {args.agent2} not found")
        return
    
    run_arena(args.agent1, args.agent2, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
