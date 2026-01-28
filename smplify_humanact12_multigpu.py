'''
Multi-GPU version of smplify_humanact12.py
Parallelizes SMPLify processing across multiple GPUs

credit to joints2smpl
https://github.com/wangsen1312/joints2smpl

Use SMPLify to process humanact12 dataset and obtain SMPL parameters

Input folder: './pose_data/humanact12'
Output folder: './humanact12/
'''

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from joints2smpl.src import config
import smplx
import h5py
from joints2smpl.src.smplify import SMPLify3D

# Configuration
NUM_GPUS = 8
num_joints = 22
joint_category = "AMASS"
num_smplify_iters = 150
fix_foot = False

data_dir = './pose_data/humanact12/humanact12'
save_dir = './humanact12/humanact12/'


def process_file(file_name, data_dir, save_dir, device):
    """Process a single file with SMPLify on the specified device."""
    input_joints = np.load(os.path.join(data_dir, file_name))

    input_joints = input_joints[:, :, [0, 2, 1]]  # amass stands on x, y

    # XY at origin
    input_joints[..., [0, 1]] -= input_joints[0, 0, [0, 1]]

    # Put on Floor
    floor_height = input_joints[:, :, 2].min()
    input_joints[:, :, 2] -= floor_height

    batch_size = input_joints.shape[0]

    smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                             model_type="smpl", gender="neutral", ext="pkl",
                             batch_size=batch_size).to(device)

    # Load the mean pose as original
    smpl_mean_file = config.SMPL_MEAN_FILE
    file = h5py.File(smpl_mean_file, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)
    file.close()

    # Initialize SMPLify
    smplify = SMPLify3D(smplxmodel=smplmodel,
                        batch_size=batch_size,
                        joints_category=joint_category,
                        num_iters=num_smplify_iters,
                        device=device)

    keypoints_3d = torch.Tensor(input_joints).to(device).float()

    pred_betas = init_mean_shape
    pred_pose = init_mean_pose
    pred_cam_t = cam_trans_zero

    confidence_input = torch.ones(num_joints)
    if fix_foot:
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5

    new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(device),
        )

    poses = new_opt_pose.detach().cpu().numpy()
    betas = new_opt_betas.mean(axis=0).detach().cpu().numpy()
    trans = keypoints_3d[:, 0].detach().cpu().numpy()

    input_joints = input_joints[:, :, [0, 2, 1]]  # jts stands on x, z
    input_joints[..., 0] *= -1
    param = {
        'bdata_poses': poses,
        'bdata_trans': trans,
        'betas': betas,
        'gender': 'male',
        'jtr': input_joints,
    }

    np.save(os.path.join(save_dir, file_name), param)


def worker(gpu_id, file_list, data_dir, save_dir):
    """Worker function that processes a subset of files on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")

    # Create progress bar for this worker
    pbar = tqdm(file_list, desc=f"GPU {gpu_id}", position=gpu_id)

    for file_name in pbar:
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            continue

        try:
            process_file(file_name, data_dir, save_dir, device)
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing {file_name}: {e}")
            continue


def main():
    os.makedirs(save_dir, exist_ok=True)

    # Get list of files to process
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # Filter out already processed files
    files_to_process = [f for f in all_files if not os.path.exists(os.path.join(save_dir, f))]

    print(f"Total files: {len(all_files)}")
    print(f"Already processed: {len(all_files) - len(files_to_process)}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Using {NUM_GPUS} GPUs")

    if len(files_to_process) == 0:
        print("All files already processed!")
        return

    # Split files across GPUs
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, f in enumerate(files_to_process):
        chunks[i % NUM_GPUS].append(f)

    for i, chunk in enumerate(chunks):
        print(f"GPU {i}: {len(chunk)} files")

    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Launch workers
    processes = []
    for gpu_id in range(NUM_GPUS):
        if len(chunks[gpu_id]) > 0:
            p = mp.Process(target=worker, args=(gpu_id, chunks[gpu_id], data_dir, save_dir))
            p.start()
            processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All processing complete!")


if __name__ == "__main__":
    main()
