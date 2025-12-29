import numpy as np

def compute_reward(prev_state, curr_state, events):
    """
    Computes the reward for the current step.
    
    Args:
        prev_state: Dict with 'me', 'enemy' state BEFORE the step.
        curr_state: Dict with 'me', 'enemy' state AFTER the step.
        events: Dict containing event flags, typically:
            - 'winner': 'me', 'enemy', or None
            - 'hit_enemy': bool (my bullet hit enemy)
            - 'hit_by_enemy': bool (enemy bullet hit me)
        - 'step_count': int (0-based index of the step in the episode; not
          used in the default shaping but provided for custom schemes)
            
    Returns:
        float: The scalar reward
    """
    reward = 0.0
    
    # Win/Loss
    if events['winner'] == 'me':
        reward += 50.0
    elif events['winner'] == 'enemy':
        reward -= 50.0
        
    # Damage dealt/received
    if events['hit_enemy']:
        reward += 10.0
    if events['hit_by_enemy']:
        reward -= 10.0
        
    # Time penalty to encourage faster wins
    reward -= 0.01
    
    # Distance shaping (optional baseline)
    # Reward for getting closer to enemy
    prev_dist = abs(prev_state['me']['x'] - prev_state['enemy']['x']) + \
                abs(prev_state['me']['y'] - prev_state['enemy']['y'])
    curr_dist = abs(curr_state['me']['x'] - curr_state['enemy']['x']) + \
                abs(curr_state['me']['y'] - curr_state['enemy']['y'])
                
    if curr_dist < prev_dist:
        reward += 0.02
    elif curr_dist > prev_dist:
        reward -= 0.02
        
    return reward


