from constants import DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, GRID_SIZE


def has_clear_shot(state, me, enemy):
    """Check if we have a clear cardinal-direction shot (no walls between us)."""
    dx = enemy['x'] - me['x']
    dy = enemy['y'] - me['y']
    
    if dx != 0 and dy != 0:
        return False  # Diagonal - can't hit with cardinal bullets
    if dx == 0 and dy == 0:
        return False  # Same cell (shouldn't happen)
    
    grid = state['grid']
    if dx == 0:  # Same column, check vertical
        step = 1 if dy > 0 else -1
        for y in range(me['y'] + step, enemy['y'], step):
            if grid[y, me['x']] == 1:
                return False
    else:  # Same row, check horizontal
        step = 1 if dx > 0 else -1
        for x in range(me['x'] + step, enemy['x'], step):
            if grid[me['y'], x] == 1:
                return False
    return True


def is_aiming_at_enemy(me, enemy):
    """Check if our facing direction points directly toward enemy (axis-aligned)."""
    dx = enemy['x'] - me['x']
    dy = enemy['y'] - me['y']
    
    if me['dir'] == DIR_UP and dy < 0 and dx == 0:
        return True
    if me['dir'] == DIR_DOWN and dy > 0 and dx == 0:
        return True
    if me['dir'] == DIR_LEFT and dx < 0 and dy == 0:
        return True
    if me['dir'] == DIR_RIGHT and dx > 0 and dy == 0:
        return True
    return False


def compute_reward(prev_state, curr_state, events):
    """
    Improved "Berserker" Reward Function v2:
    - Fixed: Reduced aim reward to prevent "aim without shooting" local optimum
    - Added: Reward for shooting when aligned (encourages pulling the trigger)
    - Increased existential penalty to make draws more painful
    """
    reward = 0.0
    
    # --- 1. TERMINAL REWARDS ---
    # Win big, lose big.
    if events['winner'] == 'me':
        return 100.0
    elif events['winner'] == 'enemy':
        return -100.0
    
    me = curr_state['me']
    enemy = curr_state['enemy']
    prev_me = prev_state['me']
    
    # --- 2. COMBAT REWARDS ---
    # Asymmetric: trading damage is profitable (+20 hit, -5 hurt = +15 net)
    if events['hit_enemy']:
        reward += 20.0
    if events['hit_by_enemy']:
        reward -= 5.0
    
    # --- 3. AMMO DISCIPLINE ---
    prev_ammo = prev_me['ammo']
    curr_ammo = me['ammo']
    shot_fired = curr_ammo < prev_ammo
    
    if shot_fired:
        if events['hit_enemy']:
            pass  # Already rewarded above
        elif enemy.get('shield_steps', 0) > 0:
            reward -= 3.0  # Shot into shield - wasteful
        else:
            # Check if we had a clear shot when we fired
            # If we shot while aligned, small penalty; if random spam, bigger penalty
            if has_clear_shot(prev_state, prev_me, prev_state['enemy']) and \
               is_aiming_at_enemy(prev_me, prev_state['enemy']):
                reward -= 0.5  # Missed but was aligned - bad luck or enemy dodged
            else:
                reward -= 2.0  # Shot without alignment - wasteful
    
    # --- 4. SHOOTING WHEN ALIGNED (NEW - fixes "aim without shoot") ---
    # Reward for taking the shot when you have a clear opportunity
    has_shot_opportunity = has_clear_shot(curr_state, me, enemy) and \
                           is_aiming_at_enemy(me, enemy) and \
                           me.get('cooldown', 0) == 0 and \
                           me.get('ammo', 0) > 0 and \
                           me.get('reload', 0) == 0
    
    # If we have a shot opportunity but didn't shoot, small penalty
    # This directly combats the "aim and stare" behavior
    if has_shot_opportunity and not shot_fired:
        reward -= 0.3  # "Pull the trigger!"
    
    # --- 5. ALIGNMENT HINT (reduced from 1.0 to 0.15) ---
    # Just a small hint, not a goal in itself
    if has_clear_shot(curr_state, me, enemy) and is_aiming_at_enemy(me, enemy):
        reward += 0.15  # Gentle hint: "you're aimed correctly"
    
    # --- 6. PROXIMITY & MOVEMENT ---
    # Reward for getting closer to enemy
    prev_dist = abs(prev_state['enemy']['x'] - prev_me['x']) + \
                abs(prev_state['enemy']['y'] - prev_me['y'])
    curr_dist = abs(enemy['x'] - me['x']) + abs(enemy['y'] - me['y'])
    
    if curr_dist < prev_dist:
        reward += 0.2  # Moved closer - good!
    elif curr_dist > prev_dist:
        reward -= 0.1  # Moved away - bad (unless tactical)
    
    # Small bonus for being close (proximity pressure)
    max_dist = 2 * (GRID_SIZE - 1)
    reward += 0.02 * (1.0 - curr_dist / max_dist)
    
    # --- 7. HP ADVANTAGE ---
    hp_diff = me['hp'] - enemy['hp']
    reward += 0.005 * hp_diff
    
    # --- 8. EXISTENTIAL PENALTY (increased from -0.1 to -0.15) ---
    # Each step costs more, making draws painful (~-15 over 100 steps)
    reward -= 0.15
    
    return reward
