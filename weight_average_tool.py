import torch
import os

# ROOT=/mnt/SSD8T/home/huangwei/projects/FROSTER
# CKPT=$ROOT/checkpoints

raw_clip = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
# source_dir = '/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample_seed2/checkpoints'
# output_dir = '/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample_seed2/wa_checkpoints'
source_dir = '/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_ssv2_froster/checkpoints'
output_dir = '/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_ssv2_froster/checkpoints/wa_checkpoints'

wa_start = 2
wa_end = 12

def average_checkpoint(checkpoint_list):
    ckpt_list = []
    
    # raw clip
    raw_clip_weight = {}
    clip_ori_state = torch.jit.load(raw_clip, map_location='cpu').state_dict() 
    _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
    for key in clip_ori_state:
        raw_clip_weight['model.' + key] = clip_ori_state[key]

    ckpt_list.append((0, raw_clip_weight))
    for name, ckpt_id in checkpoint_list:
        ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')['model_state']))
    
    linear_porj_keys = []
    for k, v in ckpt_list[-1][1].items():
        if 'projector' in k:
            linear_porj_keys.append(k)
        elif 'adapter' in k:
            linear_porj_keys.append(k)
        elif 'post_prompt' in k:
            linear_porj_keys.append(k)
    print(linear_porj_keys)

    # threshold filter
    new_ckpt_list = []
    ckpt_id_list = []
    for i in ckpt_list:
        if int(i[0]) >= wa_start and int(i[0]) <= wa_end:
            new_ckpt_list.append(i)
            ckpt_id_list.append(int(i[0]))
    
    print("Files with the following paths will participate in the parameter averaging")
    print(ckpt_id_list)

    state_dict = {}
    for key in raw_clip_weight:
        state_dict[key] = []
        for ckpt in new_ckpt_list:
            state_dict[key].append(ckpt[1][key])
    
    for key in linear_porj_keys:
        state_dict[key] = []
        for ckpt in new_ckpt_list:
            state_dict[key].append(ckpt[1][key])
    
    for key in state_dict:
        try:
            state_dict[key] = torch.mean(torch.stack(state_dict[key], 0), 0)
        except:
            print(key)

    return state_dict


os.makedirs(output_dir, exist_ok=True)
checkpoint_list = os.listdir(source_dir)

# 修改文件过滤和解析逻辑
filtered_checkpoint_list = []
for filename in checkpoint_list:
    try:
        # 只处理.pyth文件
        if not filename.endswith('.pyth'):
            continue
            
        # 解析类似 checkpoint_epoch_00003.pyth 格式
        epoch_str = filename.split('_')[-1].split('.')[0]  # 获取 00003
        epoch_num = int(epoch_str)  # 转换为数字
        
        filtered_checkpoint_list.append((os.path.join(source_dir, filename), epoch_num))
    except ValueError:
        print(f"Skipping file {filename} as it doesn't match the expected format")

if not filtered_checkpoint_list:
    raise ValueError(f"No valid checkpoint files found in {source_dir}")

# 按epoch数字排序
checkpoint_list = sorted(filtered_checkpoint_list, key=lambda d: d[1])

print("Found checkpoints:", [f"{os.path.basename(c[0])} (epoch {c[1]})" for c in checkpoint_list])

swa_state_dict = average_checkpoint(checkpoint_list)
torch.save({'model_state': swa_state_dict}, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))

