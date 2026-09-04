import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from RL.student_vision_env import VisionStudentEnv

# 1. Define the Student CNN Architecture
class VisionStudentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Depth Image Processor (64x64)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU()
        )
        # Combine Proprioception (35) with Processed Image (128)
        self.mlp = nn.Sequential(
            nn.Linear(128 + 35, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: target [v_x, v_y, omega_z]
        )

    def forward(self, proprio, depth):
        img_features = self.cnn(depth)
        combined = torch.cat([proprio, img_features], dim=1)
        return self.mlp(combined)

def collect_dagger_data(teacher_model, env, num_episodes=50):
    proprio_data, depth_data, expert_actions = [], [], []
    
    for _ in range(num_episodes):
        # We must call the base env directly here so the Teacher gets its required scandots
        obs, _ = env.base_env.reset()
        student_obs, _ = env.reset() # Align the student's renderer
        
        terminated, truncated = False, False
        while not (terminated or truncated):
            # Teacher predicts action based on privileged scandots
            expert_action, _ = teacher_model.predict(obs, deterministic=True)
            
            # Save the Student's input alongside the Teacher's perfect answer
            proprio_data.append(student_obs["proprioception"])
            depth_data.append(student_obs["depth_image"])
            expert_actions.append(expert_action)
            
            # Step the simulation
            obs, reward, terminated, truncated, _ = env.base_env.step(expert_action)
            student_obs = env._get_student_obs(obs)
            
    return (torch.tensor(np.array(proprio_data), dtype=torch.float32), 
            torch.tensor(np.array(depth_data), dtype=torch.float32), 
            torch.tensor(np.array(expert_actions), dtype=torch.float32))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Distilling Policy on {device}...")

    # Load Phase 1 Teacher
    teacher_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_ppo_final.zip')
    env = VisionStudentEnv()
    teacher = PPO.load(teacher_path)

    # Initialize Phase 2 Student
    student = VisionStudentPolicy().to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # DAgger Loop
    for iteration in range(10):
        print(f"DAgger Iteration {iteration+1}/10: Collecting Expert Data...")
        proprio, depth, expert_actions = collect_dagger_data(teacher, env)
        
        dataset = TensorDataset(proprio.to(device), depth.to(device), expert_actions.to(device))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        print("Training Student Network...")
        for epoch in range(5):
            total_loss = 0
            for p, d, targets in dataloader:
                optimizer.zero_grad()
                predictions = student(p, d)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    torch.save(student.state_dict(), os.path.join(os.path.dirname(__file__), 'logs', 'student_vision_policy.pth'))
    print("Distillation Complete. Vision Student ready for Phase 3 Execution.")

if __name__ == '__main__':
    main()
