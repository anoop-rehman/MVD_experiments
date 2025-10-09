#!/usr/bin/env python3
"""
Simple visualization script for Multi-view Disentanglement environments.
This script allows you to:
1. Run random actions in the environment
2. Load and test trained models (if available)
3. Record videos of the agent's behavior
"""

import argparse
import os
import time
import torch
import numpy as np
import cv2
import imageio
from train import make_env
from arguments import parse_args
import algorithms
import utils

def create_visualization_args():
    """Create arguments for visualization"""
    parser = argparse.ArgumentParser(description='Visualize MVD environments')
    
    # Environment settings
    parser.add_argument('--domain_name', default='Panda', type=str, 
                       help='Environment domain (Panda or MetaWorld)')
    parser.add_argument('--task_name', default='PandaReachDense-v3', type=str,
                       help='Task name')
    parser.add_argument('--cameras', nargs='+', default=['first_person', 'third_person_front'], 
                       type=str, help='Camera views to use')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    # Visualization settings
    parser.add_argument('--num_episodes', default=3, type=int, 
                       help='Number of episodes to run')
    parser.add_argument('--max_steps', default=200, type=int,
                       help='Maximum steps per episode')
    parser.add_argument('--save_video', action='store_true',
                       help='Save video of the episodes')
    parser.add_argument('--video_dir', default='visualization_videos', type=str,
                       help='Directory to save videos')
    parser.add_argument('--show_render', action='store_true',
                       help='Show live rendering (requires display)')
    
    # Model loading (optional)
    parser.add_argument('--model_path', default=None, type=str,
                       help='Path to trained model checkpoint')
    
    # Algorithm settings (needed for model loading)
    parser.add_argument('--algorithm', default='sac', type=str)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--use_proprioceptive_state', default="True", type=str)
    
    return parser.parse_args()

def load_model(model_path, env, device):
    """Load a trained model from checkpoint"""
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return None
    
    # Create agent with same config as training
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    
    # Create a minimal config for agent creation
    class Config:
        def __init__(self):
            self.algorithm = 'sac'
            self.device = device
            self.hidden_dim = 1024
            self.hidden_depth = 2
            self.discount = 0.99
            self.batch_size = 128
            self.actor_lr = 1e-3
            self.actor_beta = 0.9
            self.actor_log_std_min = -10
            self.actor_log_std_max = 2
            self.actor_update_freq = 2
            self.init_temperature = 0.1
            self.alpha_lr = 1e-4
            self.critic_lr = 1e-3
            self.critic_tau = 0.01
            self.critic_target_update_freq = 2
            self.encoder_tau = 0.05
            self.num_conv_layers = 4
            self.feature_dim = 50
            self.num_filters = 32
            self.image_reconstruction_loss = False
            self.decoder_weight_lambda = 1e-7
            self.decoder_update_freq = 1
            self.mvd_lr = 1e-3
            self.mvd_beta = 0.9
            self.mvd_update_freq = 2
            self.use_proprioceptive_state = True
    
    cfg = Config()
    agent = algorithms.make_agent(
        env.observation_space.shape, 
        env.action_space.shape, 
        action_range, 
        cfg,
        proprioceptive_state_shape=env.proprioceptive_state_shape if cfg.use_proprioceptive_state else None
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.critic_target.load_state_dict(checkpoint['critic_target'])
    
    print(f"✅ Loaded model from {model_path}")
    return agent

def run_episode(env, agent, max_steps, use_trained_model=True, camera_view='third_person_front'):
    """Run a single episode"""
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    frames = []
    
    print(f"🎬 Starting episode (max {max_steps} steps)")
    
    for step in range(max_steps):
        if use_trained_model and agent is not None:
            with utils.eval_mode(agent):
                action = agent.act(obs, proprioceptive_state=info.get("proprioceptive_state"), sample=False)
        else:
            # Random actions
            action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        
        # Render frame
        try:
            frame = env.render(mode='rgb_array', height=480, width=480, camera=camera_view)
            frames.append(frame)
        except Exception as e:
            print(f"⚠️  Rendering failed: {e}")
            # Try default render
            try:
                frame = env.render()
                frames.append(frame)
            except:
                pass
        
        # Check if episode is done
        if terminated or truncated:
            break
    
    success = info.get('success', info.get('is_success', False))
    print(f"📊 Episode completed: {episode_steps} steps, reward: {episode_reward:.2f}, success: {success}")
    
    return frames, episode_reward, success

def main():
    args = create_visualization_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Set seed
    utils.set_seed_everywhere(args.seed)
    
    # Create environment
    print(f"🌍 Creating {args.domain_name} environment: {args.task_name}")
    print(f"📷 Using cameras: {args.cameras}")
    
    env = make_env(args, cameras=args.cameras)
    print(f"✅ Environment created successfully!")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Load model if specified
    agent = None
    use_trained_model = False
    if args.model_path:
        agent = load_model(args.model_path, env, device)
        if agent is not None:
            use_trained_model = True
            print("🤖 Using trained model")
        else:
            print("🎲 Using random actions (model loading failed)")
    else:
        print("🎲 Using random actions (no model specified)")
    
    # Create video directory
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"📹 Videos will be saved to: {args.video_dir}")
    
    # Run episodes
    total_rewards = []
    total_successes = 0
    
    for episode in range(args.num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{args.num_episodes}")
        
        # Run episode
        frames, reward, success = run_episode(
            env, agent, args.max_steps, use_trained_model, 
            camera_view=args.cameras[0] if args.cameras else 'third_person_front'
        )
        
        total_rewards.append(reward)
        if success:
            total_successes += 1
        
        # Save video if requested
        if args.save_video and frames:
            video_path = os.path.join(args.video_dir, f'episode_{episode + 1}.mp4')
            try:
                imageio.mimsave(video_path, frames, fps=10)
                print(f"💾 Video saved: {video_path}")
            except Exception as e:
                print(f"❌ Failed to save video: {e}")
    
    # Print summary
    print(f"\n📈 Summary:")
    print(f"   Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   Success rate: {total_successes}/{args.num_episodes} ({100*total_successes/args.num_episodes:.1f}%)")
    
    env.close()
    print("✅ Visualization complete!")

if __name__ == '__main__':
    main()
