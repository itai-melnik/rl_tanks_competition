import numpy as np
import random
from copy import deepcopy

from constants import *
from maps import get_random_map
from bots import BOT_POOL
from rewards import compute_reward
from spaces import Discrete, Box

class MicroTankArenaEnv:
    # We expose a simple ANSI text render mode. Graphical rendering is handled
    # externally via pygame_renderer.PygameRenderer when enabled in config.
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None):
        super().__init__()

        # Optional external config (used for reproducibility and max_steps)
        self._config = config
        self._base_seed = getattr(config, "SEED", None) if config is not None else None
        self._episode_idx = 0

        # Define action and observation space
        self.action_space = Discrete(7)
        

        self.obs_dim = 32
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        
        self.grid_size = GRID_SIZE
        self.max_steps = 100
        if config and hasattr(config, 'MAX_STEPS_PER_EPISODE'):
            self.max_steps = config.MAX_STEPS_PER_EPISODE

        # RNG used for maps, spawns, and bot selection.
        # If a base seed is provided via config.SEED, we derive per-episode
        # seeds from it for reproducible training/evaluation runs.
        if self._base_seed is not None:
            self.rng = np.random.default_rng(self._base_seed)
        else:
            self.rng = np.random.default_rng()
        self.current_map = None
        self.map_name = None
        
        self.state = {}
        self.step_count = 0
        self.bot = None

    def reset(self, seed=None, options=None):
        # When a specific seed is passed (e.g. from evaluation),
        # respect it directly. Otherwise, if we were constructed
        # with a base seed, derive a deterministic per-episode seed.
        if seed is None and self._base_seed is not None:
            seed = self._base_seed + self._episode_idx
            self._episode_idx += 1

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Select map
        self.current_map, self.map_name = get_random_map(self.rng)
        
        # Select bot
        bot_name = self.rng.choice(list(BOT_POOL.keys()))
        self.bot = BOT_POOL[bot_name]
        
        # Spawn tanks
        self.state = self._spawn_tanks()
        self.state['bullets'] = []
        self.state['grid'] = self.current_map
        
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        prev_state = deepcopy(self.state)
        
        # 1. Get Enemy Action
        enemy_action = self.bot.act(self.state, self.rng)
        
        # 2. Process Actions (Shield, Shooting, Movement)
        # We process shield first, then shooting (spawn bullets), then movement.
        
        events = {
            'winner': None,
            'hit_enemy': False,
            'hit_by_enemy': False,
            'step_count': self.step_count
        }
        
        # Handle shield activation (if chosen)
        self._handle_shield('me', action)
        self._handle_shield('enemy', enemy_action)
        
        # Spawn bullets (respect cooldown, ammo, and reload)
        self._handle_shooting('me', action)
        self._handle_shooting('enemy', enemy_action)

        # Decrease cooldowns and handle reload for both tanks
        self._update_cooldowns_and_reload('me')
        self._update_cooldowns_and_reload('enemy')
        self._update_shield_timers('me')
        self._update_shield_timers('enemy')
        
        # OPTION C: Simultaneous resolution
        # Step 1: Compute tank intents (without mutating)
        me_intent = self._compute_tank_intent('me', action)
        enemy_intent = self._compute_tank_intent('enemy', enemy_action)
        
        # Step 2: Resolve tank-tank conflicts
        me_final, enemy_final = self._resolve_tank_conflicts(me_intent, enemy_intent)
        
        # Step 3: Compute bullet intents
        bullet_intents = self._compute_bullet_intents()
        
        # Step 4: Resolve bullet-tank collisions using segments
        surviving_bullet_indices = self._resolve_bullet_tank_collisions(
            bullet_intents, me_final, enemy_final, events
        )
        
        # Step 5: Apply resolved state (commit all changes at once)
        # Apply tank positions and directions
        self.state['me']['x'] = me_final[0]
        self.state['me']['y'] = me_final[1]
        self.state['me']['dir'] = me_final[2]
        self.state['enemy']['x'] = enemy_final[0]
        self.state['enemy']['y'] = enemy_final[1]
        self.state['enemy']['dir'] = enemy_final[2]
        
        # Update surviving bullets (create mapping from bullet_idx to next position)
        bullet_pos_map = {bi: (bx, by) for bi, bx, by in bullet_intents}
        new_bullets = []
        for bullet_idx in surviving_bullet_indices:
            b = self.state['bullets'][bullet_idx]
            if bullet_idx in bullet_pos_map:
                b['x'], b['y'] = bullet_pos_map[bullet_idx]
            new_bullets.append(b)
        self.state['bullets'] = new_bullets

        # Anti-camping: penalize staying too long on the same tile
        self._apply_camping_penalty('me')
        self._apply_camping_penalty('enemy')
        
        # 4. Check Termination
        terminated = False
        me_dead = self.state['me']['hp'] <= 0
        enemy_dead = self.state['enemy']['hp'] <= 0
        if me_dead and enemy_dead:
            # Simultaneous death is treated as a draw
            events['winner'] = None
            terminated = True
        elif me_dead:
            events['winner'] = 'enemy'
            terminated = True
        elif enemy_dead:
            events['winner'] = 'me'
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # 5. Compute Reward
        reward = compute_reward(prev_state, self.state, events)
        
        return self._get_obs(), reward, terminated, truncated, events

    def _spawn_tanks(self):
        # Simple spawning: random empty spots, ensuring min distance
        empty_spots = np.argwhere(self.current_map == 0)
        
        while True:
            # Pick two random spots
            idx1 = self.rng.choice(len(empty_spots))
            idx2 = self.rng.choice(len(empty_spots))
            if idx1 == idx2: continue
            
            p1 = empty_spots[idx1] # [row, col] -> [y, x]
            p2 = empty_spots[idx2]
            
            # Check Manhattan distance > MIN_SPAWN_DISTANCE
            dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            if dist > MIN_SPAWN_DISTANCE:
                return {
                    'me': {
                        'y': p1[0],
                        'x': p1[1],
                        'dir': self.rng.integers(0, 4),
                        'hp': MAX_HP,
                        'cooldown': 0,
                        'ammo': MAX_AMMO,
                        'reload': 0,
                        'stay_steps': 0,
                        'last_x': p1[1],
                        'last_y': p1[0],
                        'shield_steps': 0,
                        'shield_cooldown': 0,
                    },
                    'enemy': {
                        'y': p2[0],
                        'x': p2[1],
                        'dir': self.rng.integers(0, 4),
                        'hp': MAX_HP,
                        'cooldown': 0,
                        'ammo': MAX_AMMO,
                        'reload': 0,
                        'stay_steps': 0,
                        'last_x': p2[1],
                        'last_y': p2[0],
                        'shield_steps': 0,
                        'shield_cooldown': 0,
                    },
                }

    def _handle_shooting(self, tank_key, action):
        tank = self.state[tank_key]
        if action == ACTION_SHOOT and tank['cooldown'] == 0 and tank['ammo'] > 0 and tank['reload'] == 0:
            # Spawn bullet
            self.state['bullets'].append({
                'x': tank['x'],
                'y': tank['y'],
                'dir': tank['dir'],
                'owner': 'me' if tank_key == 'me' else 'enemy'
            })
            tank['cooldown'] = GUN_COOLDOWN_STEPS
            tank['ammo'] -= 1
            if tank['ammo'] <= 0:
                tank['ammo'] = 0
                tank['reload'] = RELOAD_STEPS

    def _handle_shield(self, tank_key, action):
        tank = self.state[tank_key]
        if action == ACTION_SHIELD and tank.get('shield_steps', 0) == 0 and tank.get('shield_cooldown', 0) == 0:
            tank['shield_steps'] = SHIELD_DURATION_STEPS
            tank['shield_cooldown'] = SHIELD_COOLDOWN_STEPS

    def _update_cooldowns_and_reload(self, tank_key):
        tank = self.state[tank_key]
        if tank['cooldown'] > 0:
            tank['cooldown'] -= 1
        if tank['reload'] > 0:
            tank['reload'] -= 1
            if tank['reload'] <= 0 and tank['ammo'] == 0:
                # Fully reloaded
                tank['ammo'] = MAX_AMMO

    def _update_shield_timers(self, tank_key):
        tank = self.state[tank_key]
        if tank.get('shield_steps', 0) > 0:
            tank['shield_steps'] -= 1
            if tank['shield_steps'] < 0:
                tank['shield_steps'] = 0
        if tank.get('shield_cooldown', 0) > 0:
            tank['shield_cooldown'] -= 1
            if tank['shield_cooldown'] < 0:
                tank['shield_cooldown'] = 0

    def _apply_camping_penalty(self, tank_key):
        tank = self.state[tank_key]
        # Track how long we've been on the same tile
        if tank['x'] == tank['last_x'] and tank['y'] == tank['last_y']:
            tank['stay_steps'] += 1
        else:
            tank['stay_steps'] = 0
            tank['last_x'] = tank['x']
            tank['last_y'] = tank['y']

        # After threshold, start chipping HP to discourage static play
        if tank['stay_steps'] > STAY_STEPS_THRESHOLD and tank['hp'] > 0:
            tank['hp'] -= 1

    def _compute_tank_intent(self, tank_key, action):
        """
        Compute intended position and direction for a tank without mutating state.
        Returns (intended_x, intended_y, intended_dir).
        If move is invalid, returns current position.
        """
        tank = self.state[tank_key]
        current_x, current_y = tank['x'], tank['y']
        current_dir = tank['dir']
        
        # Compute intended direction (turns always apply)
        intended_dir = current_dir
        if action == ACTION_TURN_LEFT:
            intended_dir = (current_dir - 1) % 4
        elif action == ACTION_TURN_RIGHT:
            intended_dir = (current_dir + 1) % 4
        
        # Compute intended position
        intended_x, intended_y = current_x, current_y
        
        if action == ACTION_MOVE_FORWARD:
            nx, ny = current_x + DX[intended_dir], current_y + DY[intended_dir]
            if self._is_valid_move_for_intent(tank_key, nx, ny):
                intended_x, intended_y = nx, ny
        elif action == ACTION_MOVE_BACKWARD:
            back_dir = (intended_dir + 2) % 4
            nx, ny = current_x + DX[back_dir], current_y + DY[back_dir]
            if self._is_valid_move_for_intent(tank_key, nx, ny):
                intended_x, intended_y = nx, ny
        
        return intended_x, intended_y, intended_dir

    def _is_valid_move_for_intent(self, tank_key, x, y):
        """
        Check if a position is valid for movement intent computation.
        This checks bounds, walls, but NOT other tank (that's resolved later).
        """
        # Check bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        # Check walls
        if self.current_map[y, x] == 1:
            return False
        return True

    def _resolve_tank_conflicts(self, me_intent, enemy_intent):
        """
        Resolve conflicts between two tanks trying to move.
        me_intent: (x, y, dir) for player tank
        enemy_intent: (x, y, dir) for enemy tank
        
        Returns (me_final, enemy_final) where each is (x, y, dir).
        
        Rules:
        - If both intend the same destination: block both (stay in place)
        - If they intend to swap: block both (stay in place)
        - Otherwise: allow both moves
        """
        me_prev = (self.state['me']['x'], self.state['me']['y'])
        enemy_prev = (self.state['enemy']['x'], self.state['enemy']['y'])
        
        me_intent_pos = (me_intent[0], me_intent[1])
        enemy_intent_pos = (enemy_intent[0], enemy_intent[1])
        
        # Check if both intend the same destination
        if me_intent_pos == enemy_intent_pos:
            # Block both
            return me_prev + (me_intent[2],), enemy_prev + (enemy_intent[2],)
        
        # Check if they intend to swap
        if me_intent_pos == enemy_prev and enemy_intent_pos == me_prev:
            # Block both
            return me_prev + (me_intent[2],), enemy_prev + (enemy_intent[2],)
        
        # No conflict, allow both moves
        return me_intent, enemy_intent

    def _compute_bullet_intents(self):
        """
        Compute intended next positions for all bullets.
        Returns list of (bullet_index, next_x, next_y) for bullets that survive.
        Bullets that go out of bounds or hit walls are filtered out.
        """
        intents = []
        for i, b in enumerate(self.state['bullets']):
            next_x = b['x'] + DX[b['dir']]
            next_y = b['y'] + DY[b['dir']]
            
            # Check bounds and walls
            if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
                continue
            if self.current_map[next_y, next_x] == 1:
                continue
            
            intents.append((i, next_x, next_y))
        return intents

    def _resolve_bullet_tank_collisions(self, bullet_intents, me_final, enemy_final, events):
        """
        Resolve collisions between bullets and tanks using segment-aware detection.
        
        bullet_intents: list of (bullet_index, next_x, next_y)
        me_final: (x, y, dir) for player tank final position
        enemy_final: (x, y, dir) for enemy tank final position
        
        Returns list of bullets that survive (not hit).
        Updates events and tank HP as needed.
        """
        me_prev = (self.state['me']['x'], self.state['me']['y'])
        enemy_prev = (self.state['enemy']['x'], self.state['enemy']['y'])
        
        me_final_pos = (me_final[0], me_final[1])
        enemy_final_pos = (enemy_final[0], enemy_final[1])
        
        surviving_bullets = []
        hit_bullet_indices = set()
        
        for bullet_idx, bullet_next_x, bullet_next_y in bullet_intents:
            b = self.state['bullets'][bullet_idx]
            bullet_prev = (b['x'], b['y'])
            bullet_next = (bullet_next_x, bullet_next_y)
            
            hit = False
            target = None
            target_key = None
            
            # Check collision with enemy (if bullet is owned by me)
            if b['owner'] == 'me':
                # Case 1: Bullet lands on enemy's final position
                if bullet_next == enemy_final_pos:
                    hit = True
                    target = self.state['enemy']
                    target_key = 'enemy'
                # Case 2: Enemy lands on bullet's old position
                elif enemy_final_pos == bullet_prev:
                    hit = True
                    target = self.state['enemy']
                    target_key = 'enemy'
                # Case 3: Swap crossing
                elif enemy_prev == bullet_next and enemy_final_pos == bullet_prev:
                    hit = True
                    target = self.state['enemy']
                    target_key = 'enemy'
            
            # Check collision with me (if bullet is owned by enemy)
            elif b['owner'] == 'enemy':
                # Case 1: Bullet lands on my final position
                if bullet_next == me_final_pos:
                    hit = True
                    target = self.state['me']
                    target_key = 'me'
                # Case 2: I land on bullet's old position
                elif me_final_pos == bullet_prev:
                    hit = True
                    target = self.state['me']
                    target_key = 'me'
                # Case 3: Swap crossing
                elif me_prev == bullet_next and me_final_pos == bullet_prev:
                    hit = True
                    target = self.state['me']
                    target_key = 'me'
            
            if hit:
                hit_bullet_indices.add(bullet_idx)
                # Shield blocks bullet damage but the bullet still disappears
                if target.get('shield_steps', 0) <= 0:
                    target['hp'] -= DAMAGE_PER_HIT
                    if target_key == 'enemy':
                        events['hit_enemy'] = True
                    elif target_key == 'me':
                        events['hit_by_enemy'] = True
            else:
                surviving_bullets.append(bullet_idx)
        
        return surviving_bullets

    def _move_tank(self, tank_key, action):
        """Legacy method - kept for compatibility but should not be used in new code."""
        tank = self.state[tank_key]
        if action == ACTION_TURN_LEFT:
            tank['dir'] = (tank['dir'] - 1) % 4
        elif action == ACTION_TURN_RIGHT:
            tank['dir'] = (tank['dir'] + 1) % 4
        elif action == ACTION_MOVE_FORWARD:
            nx, ny = tank['x'] + DX[tank['dir']], tank['y'] + DY[tank['dir']]
            if self._is_valid_move(tank_key, nx, ny):
                tank['x'], tank['y'] = nx, ny
        elif action == ACTION_MOVE_BACKWARD:
            # Move opposite to facing
            back_dir = (tank['dir'] + 2) % 4
            nx, ny = tank['x'] + DX[back_dir], tank['y'] + DY[back_dir]
            if self._is_valid_move(tank_key, nx, ny):
                tank['x'], tank['y'] = nx, ny

    def _is_valid_move(self, tank_key, x, y):
        # Check bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        # Check walls
        if self.current_map[y, x] == 1:
            return False

        # Check collision with the other tank (simple: can't share a cell).
        # We only need to test against the *other* tank because the caller
        # always passes in a prospective position different from the mover's
        # current location.
        other_key = 'enemy' if tank_key == 'me' else 'me'
        other = self.state[other_key]
        if x == other['x'] and y == other['y']:
            return False

        return True

    def _get_obs(self):
        # Normalize everything to [0, 1]
        me = self.state['me']
        enemy = self.state['enemy']
        gs = self.grid_size - 1.0
        
        obs = []
        
        # Me: x, y, dir(onehot), hp, cooldown, ammo, reload, shield_active, shield_cooldown
        obs.extend([me['x']/gs, me['y']/gs])
        # Dir one-hot
        d_onehot = [0]*4
        d_onehot[me['dir']] = 1
        obs.extend(d_onehot)
        
        # Clamp HP so observations stay within [0, 1]
        me_hp = max(0, min(MAX_HP, me['hp']))
        obs.append(me_hp/float(MAX_HP))
        obs.append(me['cooldown']/float(GUN_COOLDOWN_STEPS))
        obs.append(me['ammo']/float(MAX_AMMO))
        obs.append(me['reload']/float(RELOAD_STEPS))
        me_shield_active = 1.0 if me.get('shield_steps', 0) > 0 else 0.0
        me_shield_cd = me.get('shield_cooldown', 0)
        obs.append(me_shield_active)
        obs.append(me_shield_cd/float(SHIELD_COOLDOWN_STEPS))
        
        # Enemy: x, y, dir(onehot), hp, shield_active
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
        
        # Raycasts / Line of Sight (Simplified)
        # Check wall/enemy in front, left, right
        # Front
        obs.extend(self._raycast(me['x'], me['y'], me['dir']))
        # Left
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']-1)%4))
        # Right
        obs.extend(self._raycast(me['x'], me['y'], (me['dir']+1)%4))
        
        # Bullet danger: summarize nearby enemy bullets that can hit us
        danger_front, danger_left, danger_right = self._compute_bullet_danger()
        obs.extend([danger_front, danger_left, danger_right])
        
        return np.array(obs, dtype=np.float32)

    def _compute_bullet_danger(self):
        """
        Return three scalars in [0, 1] indicating danger from enemy bullets
        approximately in front, to the left, and to the right of our tank.

        A bullet contributes to danger if:
          - it is owned by the enemy,
          - it is aligned horizontally or vertically with us,
          - it is moving toward us (given its direction),
          - and there is no wall between the bullet and us.

        Closer bullets yield higher danger; for each direction we take the max
        danger over all matching bullets.
        """
        me = self.state['me']
        mx, my = me['x'], me['y']

        danger_front = 0.0
        danger_left = 0.0
        danger_right = 0.0

        max_dist = max(1.0, self.grid_size - 1.0)

        for b in self.state['bullets']:
            if b.get('owner') != 'enemy':
                continue

            bx, by = b['x'], b['y']
            dx = bx - mx
            dy = by - my

            # Only consider bullets that are axis-aligned with us
            if dx != 0 and dy != 0:
                continue

            blocked = False
            dist = None

            # Check if bullet is moving toward us and can reach us
            if b['dir'] == DIR_RIGHT:
                # Bullet moving to the right; must be left of us on same row
                if dy != 0 or dx >= 0:
                    continue
                dist = -dx
                for x in range(bx + 1, mx):
                    if self.current_map[my, x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_LEFT
            elif b['dir'] == DIR_LEFT:
                # Bullet moving to the left; must be right of us on same row
                if dy != 0 or dx <= 0:
                    continue
                dist = dx
                for x in range(mx + 1, bx):
                    if self.current_map[my, x] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_RIGHT
            elif b['dir'] == DIR_DOWN:
                # Bullet moving down; must be above us on same column
                if dx != 0 or dy >= 0:
                    continue
                # dy = by - my is negative here; distance is my - by
                dist = -dy
                for y in range(by + 1, my):
                    if self.current_map[y, mx] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_UP
            elif b['dir'] == DIR_UP:
                # Bullet moving up; must be below us on same column
                if dx != 0 or dy <= 0:
                    continue
                # dy = by - my is positive here; distance is by - my
                dist = dy
                for y in range(my + 1, by):
                    if self.current_map[y, mx] == 1:
                        blocked = True
                        break
                dir_to_bullet = DIR_DOWN
            else:
                continue

            if blocked or dist is None or dist <= 0:
                continue

            # Direction from us to the bullet, relative to where we face
            rel = (dir_to_bullet - me['dir']) % 4

            # Map relative direction to front/left/right slots; ignore bullets behind us
            slot = None
            if rel == 0:
                slot = 'front'
            elif rel == 1:
                slot = 'right'
            elif rel == 3:
                slot = 'left'
            else:
                continue

            # Closer bullets are more dangerous
            danger = max(0.0, 1.0 - float(dist) / max_dist)

            if slot == 'front':
                if danger > danger_front:
                    danger_front = danger
            elif slot == 'left':
                if danger > danger_left:
                    danger_left = danger
            elif slot == 'right':
                if danger > danger_right:
                    danger_right = danger

        return danger_front, danger_left, danger_right

    def _raycast(self, x, y, direction):
        # Returns [dist_to_wall, is_enemy_visible]
        # dist normalized
        
        cx, cy = x, y
        dist = 0
        found_wall = False
        found_enemy = False
        
        while True:
            cx += DX[direction]
            cy += DY[direction]
            dist += 1
            
            if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                found_wall = True
                break
            
            if self.current_map[cy, cx] == 1:
                found_wall = True
                break
                
            if cx == self.state['enemy']['x'] and cy == self.state['enemy']['y']:
                found_enemy = True
                break
        
        return [dist / self.grid_size, 1.0 if found_enemy else 0.0]

    def render(self):
        # Simple ASCII render
        grid_str = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.current_map[r, c] == 1:
                    grid_str[r][c] = '#'
                    
        # Draw tanks
        mx, my = self.state['me']['x'], self.state['me']['y']
        ex, ey = self.state['enemy']['x'], self.state['enemy']['y']
        
        # Arrows for direction
        arrows = ['^', '>', 'v', '<']
        grid_str[my][mx] = 'P' # arrows[self.state['me']['dir']]
        grid_str[ey][ex] = 'E' # arrows[self.state['enemy']['dir']]
        
        # Draw bullets
        for b in self.state['bullets']:
            if 0 <= b['x'] < self.grid_size and 0 <= b['y'] < self.grid_size:
                grid_str[b['y']][b['x']] = '*'
                
        print("-" * (self.grid_size + 2))
        for row in grid_str:
            print("|" + "".join(row) + "|")
        print("-" * (self.grid_size + 2))
        print(f"Me HP: {self.state['me']['hp']} | Enemy HP: {self.state['enemy']['hp']}")

