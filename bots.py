import numpy as np
import torch
import os
import json
from model import DQN
from constants import *


class BaseBot:
    def act(self, state, rng):
        """
        state: A dictionary containing at least:
            'me':    {'x', 'y', 'dir', 'cooldown', 'hp', 'ammo', 'reload',
                      'shield_steps', 'shield_cooldown', ...}
            'enemy': {'x', 'y', 'dir', 'cooldown', 'hp', 'ammo', 'reload',
                      'shield_steps', 'shield_cooldown', ...}
            'bullets': list of {'x', 'y', 'dir', 'owner'}
            'grid': 2D numpy array (1 = wall, 0 = empty)
        rng: numpy random generator
        """
        return ACTION_DO_NOTHING

class RandomBot(BaseBot):
    def act(self, state, rng):
        # Avoid obviously bad actions (running into a wall or shooting when the
        # gun cannot fire), otherwise choose randomly.
        me = state["me"]
        grid = state["grid"]

        actions = [ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_SHOOT]

        # Filter out forward moves that would hit a wall or go out of bounds.
        nx = me["x"] + DX[me["dir"]]
        ny = me["y"] + DY[me["dir"]]
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[ny, nx] == 1:
            actions = [a for a in actions if a != ACTION_MOVE_FORWARD]

        # Filter out shooting when the tank cannot actually shoot.
        can_shoot = (
            me.get("cooldown", 0) == 0
            and me.get("ammo", 0) > 0
            and me.get("reload", 0) == 0
        )
        if not can_shoot:
            actions = [a for a in actions if a != ACTION_SHOOT]

        if not actions:
            return ACTION_TURN_LEFT

        return rng.choice(actions)

class AggressiveBot(BaseBot):
    def act(self, state, rng):
        me = state['me']
        enemy = state['enemy']
        
        # If enemy has a clear shot on us and our shield is ready, use it
        if self._can_use_shield(me) and self._can_hit_enemy(enemy, me, state['grid']):
            return ACTION_SHIELD

        # If in line of sight and the gun is ready, shoot
        if self._can_hit_enemy(me, enemy, state['grid']) and self._can_shoot(me):
            return ACTION_SHOOT
            
        # Move towards enemy
        # Simple heuristic: turn to face enemy, then move forward
        target_dir = self._get_dir_to(me['x'], me['y'], enemy['x'], enemy['y'])
        
        if me['dir'] == target_dir:
            return ACTION_MOVE_FORWARD
        else:
            # Turn towards target
            diff = (target_dir - me['dir']) % 4
            if diff == 1:
                return ACTION_TURN_RIGHT
            else:
                return ACTION_TURN_LEFT
    
    def _get_dir_to(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return DIR_RIGHT if dx > 0 else DIR_LEFT
        else:
            return DIR_DOWN if dy > 0 else DIR_UP # y increases down

    def _can_hit_enemy(self, me, enemy, grid):
        # Check if enemy is in straight line and no walls
        if me['dir'] == DIR_UP:
            if me['x'] == enemy['x'] and me['y'] > enemy['y']:
                # Check walls
                for y in range(enemy['y'] + 1, me['y']):
                    if grid[y, me['x']] == 1: return False
                return True
        elif me['dir'] == DIR_DOWN:
            if me['x'] == enemy['x'] and me['y'] < enemy['y']:
                for y in range(me['y'] + 1, enemy['y']):
                    if grid[y, me['x']] == 1:
                        return False
                return True
        elif me['dir'] == DIR_LEFT:
            if me['y'] == enemy['y'] and me['x'] > enemy['x']:
                for x in range(enemy['x'] + 1, me['x']):
                    if grid[me['y'], x] == 1: return False
                return True
        elif me['dir'] == DIR_RIGHT:
            if me['y'] == enemy['y'] and me['x'] < enemy['x']:
                for x in range(me['x'] + 1, enemy['x']):
                    if grid[me['y'], x] == 1:
                        return False
                return True
        return False

    def _can_shoot(self, tank):
        return (
            tank.get("cooldown", 0) == 0
            and tank.get("ammo", 0) > 0
            and tank.get("reload", 0) == 0
        )

    def _can_use_shield(self, tank):
        return (
            tank.get("shield_steps", 0) == 0
            and tank.get("shield_cooldown", 0) == 0
        )

class CamperBot(BaseBot):
    def act(self, state, rng):
        # Stays in place, turns to enemy, shoots
        me = state['me']
        enemy = state['enemy']
        
        # If enemy can hit us and shield is ready, use it
        aggr = AggressiveBot()
        if aggr._can_use_shield(me) and aggr._can_hit_enemy(enemy, me, state['grid']):
            return ACTION_SHIELD

        # If can shoot, shoot
        if aggr._can_hit_enemy(me, enemy, state['grid']) and aggr._can_shoot(me):
            return ACTION_SHOOT
            
        # Face enemy
        target_dir = aggr._get_dir_to(me['x'], me['y'], enemy['x'], enemy['y'])
        if me['dir'] != target_dir:
            diff = (target_dir - me['dir']) % 4
            if diff == 1:
                return ACTION_TURN_RIGHT
            else:
                return ACTION_TURN_LEFT
        
        return ACTION_DO_NOTHING


class EvadeBot(BaseBot):
    def act(self, state, rng):
        """
        Simple evasion bot: tries to run away from the enemy by turning to face
        away from them and moving forward when possible.
        """
        me = state["me"]
        enemy = state["enemy"]
        grid = state["grid"]

        # Direction from me to enemy
        target_dir = self._get_dir_to(me["x"], me["y"], enemy["x"], enemy["y"])
        # Direction pointing away from enemy
        away_dir = (target_dir + 2) % 4

        # If we are already facing away and the tile ahead is free, move forward
        if me["dir"] == away_dir:
            nx = me["x"] + DX[away_dir]
            ny = me["y"] + DY[away_dir]
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                return ACTION_MOVE_FORWARD

        # Otherwise, turn towards the away direction
        diff = (away_dir - me["dir"]) % 4
        if diff == 1:
            return ACTION_TURN_RIGHT
        elif diff == 3:
            return ACTION_TURN_LEFT
        elif diff == 2:
            # Opposite direction: choose a turn consistently
            return ACTION_TURN_LEFT

        # Fallback: do nothing if something unexpected happens
        return ACTION_DO_NOTHING

    def _get_dir_to(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return DIR_RIGHT if dx > 0 else DIR_LEFT
        else:
            return DIR_DOWN if dy > 0 else DIR_UP  # y increases down


# ============================================================================
# CHALLENGE BOTS (for evaluation only - not in training pool)
# ============================================================================

class SniperBot(BaseBot):
    """
    Stays at range, only shoots with perfect alignment.
    Backs up if enemy gets too close.
    """
    
    def act(self, state, rng):
        me = state['me']
        enemy = state['enemy']
        grid = state['grid']
        
        # If enemy can hit us and shield is ready, use it
        if self._can_use_shield(me) and self._enemy_can_hit_me(me, enemy, grid):
            return ACTION_SHIELD
        
        # Check distance
        dist = abs(enemy['x'] - me['x']) + abs(enemy['y'] - me['y'])
        
        # If we have a clear shot and can shoot, take it
        if self._can_hit_enemy(me, enemy, grid) and self._can_shoot(me):
            return ACTION_SHOOT
        
        # If too close (< 4 tiles), back up
        if dist < 4:
            back_dir = (me['dir'] + 2) % 4
            nx = me['x'] + DX[back_dir]
            ny = me['y'] + DY[back_dir]
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                return ACTION_MOVE_BACKWARD
        
        # Turn to face enemy for sniping
        target_dir = self._get_dir_to(me['x'], me['y'], enemy['x'], enemy['y'])
        if me['dir'] != target_dir:
            diff = (target_dir - me['dir']) % 4
            if diff == 1:
                return ACTION_TURN_RIGHT
            else:
                return ACTION_TURN_LEFT
        
        # Stay put and wait for shot
        return ACTION_DO_NOTHING
    
    def _can_hit_enemy(self, me, enemy, grid):
        if me['dir'] == DIR_UP:
            if me['x'] == enemy['x'] and me['y'] > enemy['y']:
                for y in range(enemy['y'] + 1, me['y']):
                    if grid[y, me['x']] == 1:
                        return False
                return True
        elif me['dir'] == DIR_DOWN:
            if me['x'] == enemy['x'] and me['y'] < enemy['y']:
                for y in range(me['y'] + 1, enemy['y']):
                    if grid[y, me['x']] == 1:
                        return False
                return True
        elif me['dir'] == DIR_LEFT:
            if me['y'] == enemy['y'] and me['x'] > enemy['x']:
                for x in range(enemy['x'] + 1, me['x']):
                    if grid[me['y'], x] == 1:
                        return False
                return True
        elif me['dir'] == DIR_RIGHT:
            if me['y'] == enemy['y'] and me['x'] < enemy['x']:
                for x in range(me['x'] + 1, enemy['x']):
                    if grid[me['y'], x] == 1:
                        return False
                return True
        return False
    
    def _enemy_can_hit_me(self, me, enemy, grid):
        # Check if enemy is aligned and facing us
        if enemy['dir'] == DIR_UP and enemy['x'] == me['x'] and enemy['y'] > me['y']:
            for y in range(me['y'] + 1, enemy['y']):
                if grid[y, me['x']] == 1:
                    return False
            return True
        elif enemy['dir'] == DIR_DOWN and enemy['x'] == me['x'] and enemy['y'] < me['y']:
            for y in range(enemy['y'] + 1, me['y']):
                if grid[y, me['x']] == 1:
                    return False
            return True
        elif enemy['dir'] == DIR_LEFT and enemy['y'] == me['y'] and enemy['x'] > me['x']:
            for x in range(me['x'] + 1, enemy['x']):
                if grid[me['y'], x] == 1:
                    return False
            return True
        elif enemy['dir'] == DIR_RIGHT and enemy['y'] == me['y'] and enemy['x'] < me['x']:
            for x in range(enemy['x'] + 1, me['x']):
                if grid[me['y'], x] == 1:
                    return False
            return True
        return False
    
    def _can_shoot(self, tank):
        return (
            tank.get("cooldown", 0) == 0
            and tank.get("ammo", 0) > 0
            and tank.get("reload", 0) == 0
        )
    
    def _can_use_shield(self, tank):
        return (
            tank.get("shield_steps", 0) == 0
            and tank.get("shield_cooldown", 0) == 0
        )
    
    def _get_dir_to(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return DIR_RIGHT if dx > 0 else DIR_LEFT
        else:
            return DIR_DOWN if dy > 0 else DIR_UP


class DodgerBot(BaseBot):
    """
    Actively dodges incoming bullets, counter-attacks when safe.
    Uses bullet danger detection to sidestep.
    """
    
    def act(self, state, rng):
        me = state['me']
        enemy = state['enemy']
        grid = state['grid']
        bullets = state['bullets']
        
        # Check for incoming danger
        danger_dir = self._check_bullet_danger(me, bullets, grid)
        
        if danger_dir is not None:
            # Dodge: move perpendicular to the bullet
            dodge_action = self._get_dodge_action(me, danger_dir, grid)
            if dodge_action is not None:
                return dodge_action
        
        # If safe and can shoot, take the shot
        if self._can_hit_enemy(me, enemy, grid) and self._can_shoot(me):
            return ACTION_SHOOT
        
        # Chase enemy
        target_dir = self._get_dir_to(me['x'], me['y'], enemy['x'], enemy['y'])
        
        if me['dir'] == target_dir:
            nx = me['x'] + DX[me['dir']]
            ny = me['y'] + DY[me['dir']]
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                return ACTION_MOVE_FORWARD
        else:
            diff = (target_dir - me['dir']) % 4
            if diff == 1:
                return ACTION_TURN_RIGHT
            else:
                return ACTION_TURN_LEFT
        
        return ACTION_DO_NOTHING
    
    def _check_bullet_danger(self, me, bullets, grid):
        """Check if any bullet is heading toward us. Return direction of danger."""
        mx, my = me['x'], me['y']
        
        for b in bullets:
            if b.get('owner') == 'enemy':  # Only fear player's bullets
                continue
            
            bx, by = b['x'], b['y']
            
            # Check if bullet is on collision course
            if b['dir'] == DIR_RIGHT and by == my and bx < mx:
                # Bullet moving right toward us
                blocked = False
                for x in range(bx + 1, mx):
                    if grid[my, x] == 1:
                        blocked = True
                        break
                if not blocked and (mx - bx) <= 4:  # Close enough to worry
                    return DIR_LEFT  # Danger from left
                    
            elif b['dir'] == DIR_LEFT and by == my and bx > mx:
                blocked = False
                for x in range(mx + 1, bx):
                    if grid[my, x] == 1:
                        blocked = True
                        break
                if not blocked and (bx - mx) <= 4:
                    return DIR_RIGHT
                    
            elif b['dir'] == DIR_DOWN and bx == mx and by < my:
                blocked = False
                for y in range(by + 1, my):
                    if grid[y, mx] == 1:
                        blocked = True
                        break
                if not blocked and (my - by) <= 4:
                    return DIR_UP
                    
            elif b['dir'] == DIR_UP and bx == mx and by > my:
                blocked = False
                for y in range(my + 1, by):
                    if grid[y, mx] == 1:
                        blocked = True
                        break
                if not blocked and (by - my) <= 4:
                    return DIR_DOWN
        
        return None
    
    def _get_dodge_action(self, me, danger_dir, grid):
        """Get action to dodge a bullet coming from danger_dir."""
        # Move perpendicular to the danger
        if danger_dir in [DIR_LEFT, DIR_RIGHT]:
            # Danger from side, move up or down
            for try_dir in [DIR_UP, DIR_DOWN]:
                nx = me['x'] + DX[try_dir]
                ny = me['y'] + DY[try_dir]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                    if me['dir'] == try_dir:
                        return ACTION_MOVE_FORWARD
                    # Turn toward escape direction
                    diff = (try_dir - me['dir']) % 4
                    if diff == 1:
                        return ACTION_TURN_RIGHT
                    elif diff == 3:
                        return ACTION_TURN_LEFT
        else:
            # Danger from up/down, move left or right
            for try_dir in [DIR_LEFT, DIR_RIGHT]:
                nx = me['x'] + DX[try_dir]
                ny = me['y'] + DY[try_dir]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                    if me['dir'] == try_dir:
                        return ACTION_MOVE_FORWARD
                    diff = (try_dir - me['dir']) % 4
                    if diff == 1:
                        return ACTION_TURN_RIGHT
                    elif diff == 3:
                        return ACTION_TURN_LEFT
        
        return None
    
    def _can_hit_enemy(self, me, enemy, grid):
        if me['dir'] == DIR_UP:
            if me['x'] == enemy['x'] and me['y'] > enemy['y']:
                for y in range(enemy['y'] + 1, me['y']):
                    if grid[y, me['x']] == 1:
                        return False
                return True
        elif me['dir'] == DIR_DOWN:
            if me['x'] == enemy['x'] and me['y'] < enemy['y']:
                for y in range(me['y'] + 1, enemy['y']):
                    if grid[y, me['x']] == 1:
                        return False
                return True
        elif me['dir'] == DIR_LEFT:
            if me['y'] == enemy['y'] and me['x'] > enemy['x']:
                for x in range(enemy['x'] + 1, me['x']):
                    if grid[me['y'], x] == 1:
                        return False
                return True
        elif me['dir'] == DIR_RIGHT:
            if me['y'] == enemy['y'] and me['x'] < enemy['x']:
                for x in range(me['x'] + 1, enemy['x']):
                    if grid[me['y'], x] == 1:
                        return False
                return True
        return False
    
    def _can_shoot(self, tank):
        return (
            tank.get("cooldown", 0) == 0
            and tank.get("ammo", 0) > 0
            and tank.get("reload", 0) == 0
        )
    
    def _get_dir_to(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):
            return DIR_RIGHT if dx > 0 else DIR_LEFT
        else:
            return DIR_DOWN if dy > 0 else DIR_UP


class SelfPlayBot(BaseBot):
    """
    Bot that uses a trained agent's policy.
    Converts state to observation from its own perspective (swapping me/enemy).
    Periodically reloads the model to pick up the latest trained version.
    """
    
    def __init__(self, model_path="checkpoints/agent.pt", device="cpu", reload_every=500):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.config_path = "checkpoints/model_config.json"
        self.reload_every = reload_every  # Reload model every N actions
        self.action_count = 0
        self.last_mtime = 0  # Track file modification time
        
        self._load_model()
    
    def _load_model(self):
        """Load or reload the model from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            try:
                # Check if file has been updated
                current_mtime = os.path.getmtime(self.model_path)
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                self.model = DQN(
                    model_config["obs_dim"],
                    model_config["action_dim"],
                    model_config["hidden_sizes"]
                )
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.last_mtime = current_mtime
            except (OSError, KeyError, RuntimeError) as e:
                if self.model is None:
                    print(f"SelfPlayBot: Could not load model: {e}")
    
    def act(self, state, rng):
        self.action_count += 1
        
        # Periodically check if model file has been updated
        if self.action_count % self.reload_every == 0:
            try:
                current_mtime = os.path.getmtime(self.model_path)
                if current_mtime > self.last_mtime:
                    self._load_model()  # Reload if file changed
            except OSError:
                pass
        if self.model is None:
            # Fallback to aggressive behavior if model not loaded
            return AggressiveBot().act(state, rng)
        
        # Convert state to observation from THIS bot's perspective
        # (swap me/enemy since this bot IS the enemy from env's view)
        obs = self._state_to_obs(state)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(obs_tensor)
            return q_values.argmax().item()
    
    def _state_to_obs(self, state):
        """
        Convert state to observation from this bot's perspective.
        This bot is 'enemy' in the state, so we swap me/enemy.
        """
        
        # From this bot's POV: I am 'enemy', opponent is 'me'
        me = state['enemy']      # This bot
        enemy = state['me']      # The opponent (player)
        grid = state['grid']
        bullets = state['bullets']
        
        gs = GRID_SIZE - 1.0
        obs = []
        
        # Me (this bot): x, y, dir(onehot), hp, cooldown, ammo, reload, shield_active, shield_cooldown
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
        
        # Enemy (the player): x, y, dir(onehot), hp, shield_active
        obs.extend([enemy['x']/gs, enemy['y']/gs])
        e_d_onehot = [0]*4
        e_d_onehot[enemy['dir']] = 1
        obs.extend(e_d_onehot)
        enemy_hp = max(0, min(MAX_HP, enemy['hp']))
        obs.append(enemy_hp/float(MAX_HP))
        enemy_shield_active = 1.0 if enemy.get('shield_steps', 0) > 0 else 0.0
        obs.append(enemy_shield_active)
        
        # Relative position (from this bot's view)
        obs.append((enemy['x'] - me['x']) / gs)
        obs.append((enemy['y'] - me['y']) / gs)
        dist = abs(enemy['x'] - me['x']) + abs(enemy['y'] - me['y'])
        obs.append(dist / (2*gs))
        
        # Raycasts from this bot's position
        obs.extend(self._raycast(me['x'], me['y'], me['dir'], enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']-1)%4, enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']+1)%4, enemy, grid))
        
        # Bullet danger (from this bot's perspective - enemy bullets are from 'me' owner)
        danger_front, danger_left, danger_right = self._compute_bullet_danger(
            me, bullets, grid, owner_to_fear='me'  # Fear bullets from 'me' (the player)
        )
        obs.extend([danger_front, danger_left, danger_right])
        
        return np.array(obs, dtype=np.float32)
    
    def _raycast(self, x, y, direction, enemy, grid):
        """Raycast from position in direction, return [dist_to_wall, is_enemy_visible]."""
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
        """Compute bullet danger from this bot's perspective."""
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
                for x in range(bx + 1, mx):
                    if grid[my, x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_LEFT
            elif b['dir'] == DIR_LEFT:
                if dy != 0 or dx <= 0:
                    continue
                dist = dx
                for x in range(mx + 1, bx):
                    if grid[my, x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_RIGHT
            elif b['dir'] == DIR_DOWN:
                if dx != 0 or dy >= 0:
                    continue
                dist = -dy
                for y in range(by + 1, my):
                    if grid[y, mx] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_UP
            elif b['dir'] == DIR_UP:
                if dx != 0 or dy <= 0:
                    continue
                dist = dy
                for y in range(my + 1, by):
                    if grid[y, mx] == 1:
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


class LeagueBot(BaseBot):
    """
    Bot that randomly selects from a pool of saved agent versions.
    Creates diverse opponents from past training checkpoints.
    """
    
    def __init__(self, league_dir="checkpoints/league/", device="cpu"):
        self.device = device
        self.league_dir = league_dir
        self.config_path = "checkpoints/model_config.json"
        self.agents = []
        self.current_agent_idx = None
        self.episodes_with_current = 0
        self.switch_every = 5  # Switch opponent every N episodes
        
        self._load_league()
    
    def _load_league(self):
        """Load all agent versions from the league directory."""
        if not os.path.exists(self.league_dir) or not os.path.exists(self.config_path):
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        
        for filename in sorted(os.listdir(self.league_dir)):
            if filename.endswith('.pt'):
                try:
                    model = DQN(
                        model_config["obs_dim"],
                        model_config["action_dim"],
                        model_config["hidden_sizes"]
                    )
                    model.load_state_dict(torch.load(
                        os.path.join(self.league_dir, filename),
                        map_location=self.device
                    ))
                    model.eval()
                    self.agents.append((filename, model))
                except (OSError, KeyError, RuntimeError):
                    pass
        
        if self.agents:
            print(f"LeagueBot: Loaded {len(self.agents)} agents from {self.league_dir}")
    
    def act(self, state, rng):
        if not self.agents:
            # Fallback to AggressiveBot if no league agents
            return AggressiveBot().act(state, rng)
        
        # Pick a random agent periodically
        if self.current_agent_idx is None:
            self.current_agent_idx = rng.integers(0, len(self.agents))
        
        _, model = self.agents[self.current_agent_idx]
        obs = self._state_to_obs(state)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = model(obs_tensor)
            return q_values.argmax().item()
    
    def reset_opponent(self, rng):
        """Call this at episode start to potentially switch opponents."""
        self.episodes_with_current += 1
        if self.episodes_with_current >= self.switch_every and self.agents:
            self.current_agent_idx = rng.integers(0, len(self.agents))
            self.episodes_with_current = 0
    
    def _state_to_obs(self, state):
        """Convert state to observation from this bot's perspective (same as SelfPlayBot)."""
        me = state['enemy']
        enemy = state['me']
        grid = state['grid']
        bullets = state['bullets']
        
        gs = GRID_SIZE - 1.0
        obs = []
        
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
        
        obs.extend([enemy['x']/gs, enemy['y']/gs])
        e_d_onehot = [0]*4
        e_d_onehot[enemy['dir']] = 1
        obs.extend(e_d_onehot)
        enemy_hp = max(0, min(MAX_HP, enemy['hp']))
        obs.append(enemy_hp/float(MAX_HP))
        enemy_shield_active = 1.0 if enemy.get('shield_steps', 0) > 0 else 0.0
        obs.append(enemy_shield_active)
        
        obs.append((enemy['x'] - me['x']) / gs)
        obs.append((enemy['y'] - me['y']) / gs)
        dist = abs(enemy['x'] - me['x']) + abs(enemy['y'] - me['y'])
        obs.append(dist / (2*gs))
        
        obs.extend(self._raycast(me['x'], me['y'], me['dir'], enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']-1)%4, enemy, grid))
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']+1)%4, enemy, grid))
        
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


# Phase 5b: BALANCED SELF-PLAY
# Keep heuristic bots (prevents forgetting) + add self-play as bonus
# Distribution: ~71% heuristic bots, ~29% self-play variants
BOT_POOL = {
    # Heuristic bots (5 bots = 71%)
    "base": BaseBot(),
    "random": RandomBot(),
    "aggressive": AggressiveBot(),   # Key opponent - don't forget this!
    "camper": CamperBot(),
    "evade": EvadeBot(),
    # Self-play variants (2 bots = 29%)
    "selfplay": SelfPlayBot(),       # Current agent (reloads when file changes)
    "league": LeagueBot(),           # Past versions for diversity
}


