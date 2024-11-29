import torch

for i in range(1, 13):
    checkpoint_path = f'/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/checkpoints/checkpoint_epoch_{i:05d}.pyth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded successfully. Epoch: {i}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}   Epoch: {i}")