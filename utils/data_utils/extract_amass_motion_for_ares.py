import os 
import numpy as np 
import json 
import pickle as pkl 

def matrot2axisangle(matrots):
    '''
    :param matrots: N*num_joints*9
    :return: N*num_joints*3
    '''
    import cv2
    batch_size = matrots.shape[0]
    matrots = matrots.reshape([batch_size,-1,9])
    out_axisangle = []
    for mIdx in range(matrots.shape[0]):
        cur_axisangle = []
        for jIdx in range(matrots.shape[1]):
            a = cv2.Rodrigues(matrots[mIdx, jIdx:jIdx + 1, :].reshape(3, 3))[0].reshape((1, 3))
            cur_axisangle.append(a)

        out_axisangle.append(np.array(cur_axisangle).reshape([1,-1,3]))
    return np.vstack(out_axisangle)

def read_motion_from_habitat(amass_ori_folder, habitat_folder, extract_index_pkl_path):
    pkl_data = pkl.load(open(extract_index_pkl_path, 'rb'))

    for k in pkl_data:
        curr_seq_data = pkl_data[k]

        scene_name = curr_seq_data['scene_name']
        seq_name = curr_seq_data['seq_name']

        start_frame_idx = curr_seq_data['start_frame_idx']
        num_frames = curr_seq_data['num_frames'] 

        ori_data_path = os.path.join(amass_ori_folder, curr_seq_data['path'])
        ori_npz_data = np.load(ori_data_path, allow_pickle=True) 

        seq_folder = os.path.join(habitat_folder, scene_name, seq_name) 
        dest_npz_path = os.path.join(seq_folder, "ori_motion_seq.npz")

        np.savez(dest_npz_path, fps=ori_npz_data['fps'],
            gender=ori_npz_data['gender'],
            floor_height=ori_npz_data['floor_height'],
            contacts=ori_npz_data['contacts'][start_frame_idx:start_frame_idx+num_frames],
            trans=ori_npz_data['trans'][start_frame_idx:start_frame_idx+num_frames],
            root_orient=ori_npz_data['root_orient'][start_frame_idx:start_frame_idx+num_frames],
            pose_body=ori_npz_data['pose_body'][start_frame_idx:start_frame_idx+num_frames],
            betas=ori_npz_data['betas'],
            joints=ori_npz_data['joints'][start_frame_idx:start_frame_idx+num_frames],
            trans_vel=ori_npz_data['trans_vel'][start_frame_idx:start_frame_idx+num_frames],
            root_orient_vel=ori_npz_data['root_orient_vel'][start_frame_idx:start_frame_idx+num_frames],
            pose_body_vel=ori_npz_data['pose_body_vel'][start_frame_idx:start_frame_idx+num_frames],
            world2aligned_rot=ori_npz_data['world2aligned_rot'][start_frame_idx:start_frame_idx+num_frames])

if __name__ == "__main__":
    amass_data_folder = "data/processed_amass_same_shape_seq"
    habitat_folder = "data/ares/ares_ego_videos"
    extract_index_pkl_path = "data/ares/extract_amass_for_ares.pkl"
    read_motion_from_habitat(amass_data_folder, habitat_folder, extract_index_pkl_path)
