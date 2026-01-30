'''
Compute Mean, Std

Input folder: './HumanML3D/smpl'
Output folder: './HumanML3D/

Purpose: Compute mean and standard deviation for normalization                                                                                                  
┌────────────────────────┬─────────────────────────┐                                                                                                            
│         Input          │         Output          │                                                                                                            
├────────────────────────┼─────────────────────────┤                                                                                                            
│ ./HumanML3D/smpl/*.npy │ ./HumanML3D/Mean_6d.npy │                                                                                                            
├────────────────────────┼─────────────────────────┤                                                                                                            
│                        │ ./HumanML3D/Std_6d.npy  │                                                                                                            
└────────────────────────┴─────────────────────────┘
                                                                                                            
Details: Aggregates all 6D pose data from the processed files and computes per-dimension mean and std for later normalization. 
'''
import os
import numpy as np
   
if __name__ == "__main__":
    data_dir = './HumanML3D/smpl/'
    save_dir = './HumanML3D/'

    file_list = os.listdir(data_dir)
    poses6d_list = []

    for file in file_list:
        data = np.load(os.path.join(data_dir, file),allow_pickle=True).item()
        poses6d_list.append(data['pose_6d'])

    poses6d = np.concatenate(poses6d_list, axis=0)

    poses6d_mean = poses6d.mean(axis=0)
    poses6d_std = poses6d.std(axis=0)

    np.save(os.path.join(save_dir, 'Mean_6d.npy'), poses6d_mean)
    np.save(os.path.join(save_dir, 'Std_6d.npy'), poses6d_std)
