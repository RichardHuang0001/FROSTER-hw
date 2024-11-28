import torch

checkpoint_path = '/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/checkpoints/checkpoint_epoch_00001.pyth'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")