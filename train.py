import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import random
import logging

from env import MicroTankArenaEnv
from model import DQN, HIDDEN_SIZES
from utils import ReplayBuffer
import config

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 1. Setup
    set_seed(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Save model config for evaluation
    model_config = {
        "hidden_sizes": HIDDEN_SIZES,
        "obs_dim": 0, # To be filled
        "action_dim": 0 # To be filled
    }
    
    # 2. Environment
    env = MicroTankArenaEnv(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model_config["obs_dim"] = obs_dim
    model_config["action_dim"] = int(action_dim)
    
    # Save config immediately
    with open(config.MODEL_CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=4)
        
    logger.info(f"Environment initialized. State dim: {obs_dim}, action dim: {action_dim}")
    logger.info(f"Device: {config.DEVICE}")

    # 3. Agents
    policy_net = DQN(obs_dim, action_dim, HIDDEN_SIZES).to(config.DEVICE)
    
    # Load checkpoint if enabled and exists
    if getattr(config, "LOAD_CHECKPOINT", False) and os.path.exists(config.MODEL_SAVE_PATH):
        logger.info(f"Loading checkpoint from {config.MODEL_SAVE_PATH}")
        policy_net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    
    target_net = DQN(obs_dim, action_dim, HIDDEN_SIZES).to(config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, config.DEVICE)

    # Optional pygame renderer
    renderer = None
    if getattr(config, "USE_PYGAME_RENDER", False):
        from pygame_renderer import PygameRenderer
        renderer = PygameRenderer(env.grid_size)
    
    # 4. Training Loop
    epsilon = config.EPS_START
    global_step = 0
    
    # Trackers
    ep_rewards = []
    ep_wins = []
    
    logger.info("Starting training...")
    
    for episode in range(config.NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        step_in_ep = 0
        
        while not (done or truncated):
            # Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(obs, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
            
            # Step
            next_obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_in_ep += 1

            # Optional visualization (only when enabled)
            if renderer is not None:
                hud_info = {
                    "mode": "train",
                    "episode": episode + 1,
                    "step": step_in_ep,
                    "epsilon": epsilon,
                    "ep_reward": ep_reward,
                }
                alive = renderer.render(env, hud_info=hud_info)
                if not alive:
                    renderer = None
            
            # Buffer
            # Treat both true terminations and time-limit truncations as terminal
            # for target computation, so we do not bootstrap beyond episode end.
            done_flag = 1.0 if (done or truncated) else 0.0
            buffer.add(obs, action, reward, next_obs, done_flag)
            
            obs = next_obs
            global_step += 1
            
            # Train
            if len(buffer) > config.MIN_BUFFER_SIZE and global_step % config.TRAIN_FREQ == 0:
                batch = buffer.sample(config.BATCH_SIZE)
                
                # Compute Q Targets
                with torch.no_grad():
                    target_q = target_net(batch['next_obs']).max(1)[0]
                    target = batch['rews'] + config.GAMMA * target_q * (1 - batch['dones'])
                    
                # Compute Current Q
                current_q = policy_net(batch['obs']).gather(1, batch['acts'].unsqueeze(1)).squeeze(1)
                
                loss = F.mse_loss(current_q, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Update Target Net
            if global_step % config.TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        # Epsilon Decay (Linear)
        # Decay over EPS_DECAY_STEPS episodes
        if config.EPS_DECAY_STEPS > 0:
             # Calculate decay per episode
             decay_amount = (config.EPS_START - config.EPS_END) / config.EPS_DECAY_STEPS
             epsilon = max(config.EPS_END, epsilon - decay_amount)
        else:
             epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)
             
        ep_rewards.append(ep_reward)

        # Track whether this episode was a win (for simple progress signal)
        final_me_hp = env.state["me"]["hp"]
        final_enemy_hp = env.state["enemy"]["hp"]
        ep_wins.append(1 if final_me_hp > 0 and final_enemy_hp <= 0 else 0)

        # Logging every N episodes so students can see learning progress
        if (episode + 1) % 50 == 0:
            window = 50
            recent_rewards = ep_rewards[-window:]
            recent_wins = ep_wins[-window:]
            avg_rew = np.mean(recent_rewards) if recent_rewards else 0.0
            win_rate = (np.mean(recent_wins) * 100.0) if recent_wins else 0.0

            logger.info(
                f"Episode {episode+1}/{config.NUM_EPISODES} | "
                f"Avg reward (last {len(recent_rewards)}): {avg_rew:.2f} | "
                f"Win rate (last {len(recent_wins)}): {win_rate:.1f}% | "
                f"Epsilon: {epsilon:.2f} | Steps: {global_step}"
            )
        
        # Save checkpoint to league folder for self-play diversity
        # (every 300 episodes, save a snapshot for the league)
        league_dir = os.path.join(config.CHECKPOINT_DIR, "league")
        if (episode + 1) % 300 == 0:
            os.makedirs(league_dir, exist_ok=True)
            league_path = os.path.join(league_dir, f"agent_ep{episode+1}.pt")
            torch.save(policy_net.state_dict(), league_path)
            # Also update the main checkpoint for dynamic reload
            torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)
            
    # 5. Save
    logger.info(f"Training complete. Saving to {config.MODEL_SAVE_PATH}")
    torch.save(policy_net.state_dict(), config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

