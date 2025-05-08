#!/usr/bin/env python
import argparse
import os
import csv
import time
import glob
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from dqn import DQNAgent
from sawyer_ik_env import SawyerIKEnv # Make sure SawyerIKEnv.py is in the same directory or PYTHONPATH
from collections import deque

# ========================================
# Configurable constants
# ========================================
LOG_DIR      = "logs"
MODEL_DIR    = "models"
ALG_NAME     = "DQN" # Algorithm name for directory

# ensure folders exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, ALG_NAME), exist_ok=True)


# fixed goals for quick evaluation
EVAL_GOALS = [
    np.array([0.3, 0.2, 0.5], dtype=np.float32),
    np.array([0.25, 0.1, 0.4], dtype=np.float32),
    np.array([0.35, 0.3, 0.55], dtype=np.float32),
]

# CSV log path
LOG_CSV = os.path.join(LOG_DIR, f"{ALG_NAME}_training_log.csv")

def train(env_id: str, alg_name_for_dir: str):
    env = SawyerIKEnv()
    
    # Hyperparameters from paper (Table 1) and DQN defaults
    hyperparams = {
        'BUFFER_SIZE': int(1e5),
        'BATCH_SIZE': 64,
        'GAMMA': 0.9,
        'TAU': 1e-3,
        'LEARNING_RATE': 0.01,
        'UPDATE_EVERY': 1, # Paper implies learning per iteration (step)
        'GRADIENT_CLIP': 1.0,
        'LEARN_STARTS': 64 # Start learning after BATCH_SIZE experiences are collected
    }

    agent_state_size = 7 # Sawyer joint angles
    agent = DQNAgent(state_size=agent_state_size,
                     action_size=env.action_space.n,
                     seed=0,
                     hyperparams=hyperparams)

    MAX_EPISODES = 20000 # Increased episodes
    MAX_STEPS_PER_EPISODE = 300 # Increased steps per episode
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.9999 # Slower decay due to more episodes
    EVAL_FREQ = 200
    SAVE_FREQ = 1000

    eps = EPS_START
    scores_window = deque(maxlen=100)
    total_steps_R = 0

    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_steps", "mean_reward_100_eps", "mean_eval_final_dist", "epsilon"])
    
    print(f"Training {alg_name_for_dir}...")
    for i_episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        state = obs[:agent_state_size] # Agent state is joint angles
        episode_reward = 0.0
        
        for t_step_episode in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state, eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs[:agent_state_size]
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps_R += 1
            
            if done:
                break
        
        scores_window.append(episode_reward)
        eps = max(EPS_END, EPS_DECAY * eps)

        if i_episode % 10 == 0:
             print(f"\rEpisode {i_episode}\tTotal Steps: {total_steps_R}\tAvg Reward (Last 100): {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}", end="")
        if i_episode % 100 == 0:
            print() 

        if i_episode % EVAL_FREQ == 0:
            mean_eval_dist = evaluate_agent(env, agent, EVAL_GOALS, max_eval_steps=MAX_STEPS_PER_EPISODE)
            print(f"\n--- Evaluation at Episode {i_episode} ---")
            print(f"Mean final distance on eval goals: {mean_eval_dist:.4f} m")
            
            with open(LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow([i_episode, total_steps_R, np.mean(scores_window), mean_eval_dist, eps])

        if i_episode % SAVE_FREQ == 0:
            model_path = os.path.join(MODEL_DIR, alg_name_for_dir, f"{alg_name_for_dir}_episode_{i_episode}.pth")
            agent.save(model_path)
    
    env.close()
    print("\nTraining complete.")

def evaluate_agent(env_instance: SawyerIKEnv, agent_to_eval: DQNAgent, goals_to_eval, max_eval_steps=200):
    dists = []
    agent_state_size = agent_to_eval.state_size

    # Store original env state if needed, though SawyerIKEnv reset should be stateless regarding qpos
    # original_qpos = env_instance.data.qpos[:agent_state_size].copy()

    for goal_idx, goal_pos in enumerate(goals_to_eval):
        obs, _ = env_instance.reset(seed=42+goal_idx) # Use different seeds for eval resets if they are random
        env_instance.goal = goal_pos.copy()
        
        mujoco.mj_forward(env_instance.model, env_instance.data)
        ee_pos = env_instance.data.xpos[-1] # Assuming last body is end-effector
        env_instance.prev_dist = np.linalg.norm(env_instance.goal - ee_pos)
        
        current_qpos = env_instance.data.qpos[:agent_state_size].copy()
        state = current_qpos

        for _ in range(max_eval_steps):
            action = agent_to_eval.act(state, eps=0.0) # Greedy policy
            next_obs, _, terminated, truncated, _ = env_instance.step(action)
            next_state = next_obs[:agent_state_size]
            done = terminated or truncated
            
            state = next_state
            if done:
                break
        
        final_ee_pos = env_instance.data.xpos[-1]
        final_dist = np.linalg.norm(env_instance.goal - final_ee_pos)
        dists.append(final_dist)

    # env_instance.data.qpos[:agent_state_size] = original_qpos
    # mujoco.mj_forward(env_instance.model, env_instance.data)
    return float(np.mean(dists))


def test(env_id: str, alg_name_for_dir: str,
         threshold: float = 0.01,
         max_steps: int = 300):

    model_folder = os.path.join(MODEL_DIR, alg_name_for_dir)
    ckpts = glob.glob(os.path.join(model_folder, f"{alg_name_for_dir}_episode_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No {alg_name_for_dir} models in {model_folder}")
    
    ckpts.sort(key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    model_path = ckpts[-1]
    print(f"⏬ Loading policy from {model_path}")

    original_xml = "mujoco_menagerie/rethink_robotics_sawyer/scene.xml"
    scene_dir    = os.path.dirname(original_xml)
    
    env_tmp = SawyerIKEnv(xml_path=original_xml)
    obs_tmp, _ = env_tmp.reset(seed=123) # Use a fixed seed for test goal generation
    test_goal  = env_tmp.goal.copy()
    env_tmp.close()
    print(f"🎯 Test Goal at {test_goal}")

    tree = ET.parse(original_xml)
    root = tree.getroot()
    wb   = root.find("worldbody")
    
    # Remove existing goal_marker if any, to avoid duplicates if script is re-run
    existing_marker = wb.find("./site[@name='goal_marker']")
    if existing_marker is not None:
        wb.remove(existing_marker)

    site = ET.Element("site", {
        "name": "goal_marker",
        "pos":  f"{test_goal[0]} {test_goal[1]} {test_goal[2]}",
        "size": "0.03", "type": "sphere", "rgba": "0 1 0 0.7", # Slightly transparent
    })
    wb.insert(0, site) # Insert at the beginning of worldbody for visibility
    tmpxml_path = os.path.join(scene_dir, "scene_with_goal_test.xml")
    tree.write(tmpxml_path)

    env = SawyerIKEnv(xml_path=tmpxml_path)
    obs, _ = env.reset(seed=123) # Use same seed to ensure consistent start if reset is random
    env.goal = test_goal # Explicitly set the goal
    mujoco.mj_forward(env.model, env.data)
    env.prev_dist = np.linalg.norm(env.goal - env.data.xpos[-1])
    
    agent_state_size = 7
    state = obs[:agent_state_size]

    # Dummy hyperparams for loading, actual values are part of the saved model structure implicitly
    hyperparams_load_dummy = {'LEARNING_RATE': 0.01} # Only need one to satisfy constructor if it expects dict
    agent = DQNAgent.load(model_path,
                          state_size=agent_state_size,
                          action_size=env.action_space.n,
                          seed=0, 
                          hyperparams=hyperparams_load_dummy)

    dist_to_goal = np.linalg.norm(env.goal - env.data.xpos[-1])
    print(f"Initial dist to goal: {dist_to_goal:.4f} m")

    try:
        env.render() # Call once to initialize viewer if not already
        time.sleep(1.0) # Pause to see initial state

        for step_count_test in range(max_steps):
            action = agent.act(state, eps=0.0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = next_obs[:agent_state_size]
            done = terminated or truncated
            
            env.render()
            time.sleep(0.05) # Uncomment for slower visualization

            dist_to_goal = np.linalg.norm(env.goal - env.data.xpos[-1])
            print(f"\rStep: {step_count_test+1}/{max_steps}, Dist: {dist_to_goal:.4f}, Reward: {reward:.2f}   ", end="")

            if done or dist_to_goal <= threshold :
                break
        print() 

        if dist_to_goal <= threshold:
            print(f"✅ Reached goal in {step_count_test+1} steps (final dist={dist_to_goal:.4f})")
        else:
            print(f"❌ Failed after {step_count_test+1} steps (final dist={dist_to_goal:.4f})")
        time.sleep(15.0) # Pause to see final state

    finally:
        if hasattr(env, '_viewer') and env._viewer is not None and env._viewer.is_running():
            env._viewer.close()
        env.close() 
        if os.path.exists(tmpxml_path):
            try:
                os.remove(tmpxml_path)
            except OSError as e:
                print(f"Error removing temporary XML file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a custom DQN agent for Sawyer IK.")
    subs = parser.add_subparsers(dest="mode", required=True)

    p_tr = subs.add_parser("train", help="Train the CustomDQN agent.")
    
    p_te = subs.add_parser("test", help="Test the CustomDQN agent.")
    p_te.add_argument("--threshold", type=float, default=0.01, help="Success threshold for distance to goal (meters).")
    p_te.add_argument("--max-steps", type=int, default=300, help="Max steps per test episode.")

    args = parser.parse_args()
    
    if args.mode == "train":
        train(env_id=None, alg_name_for_dir=ALG_NAME)
    else: # test mode
        test(env_id=None, alg_name_for_dir=ALG_NAME,
             threshold=args.threshold,
             max_steps=args.max_steps)