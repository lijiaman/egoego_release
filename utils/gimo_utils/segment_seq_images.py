import os 
import numpy  as np 
import json 
import csv
import shutil 

def segment_data_w_csv():
    ori_root_folder = "data/gimo/gaze_dataset"
    dest_root_folder = "data/gimo/segmented_ori_data"

    csv_path = "data/gimo/gaze_dataset/dataset.csv" 
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        row_cnt = 0
        for row in reader:
            if row_cnt > 0:
                seq_name = row[0]
                scene_name = row[3]
                start_frame = int(row[1])
                end_frame = int(row[2])

                # Copy selected image files to new folder 
                img_folder = os.path.join(ori_root_folder, scene_name, seq_name, "PV") 
                tmp_img_files = os.listdir(img_folder)
                ori_img_files = []
                for tmp_name in tmp_img_files:
                    if ".png" in tmp_name:
                        ori_img_files.append(tmp_name)
                ori_img_files.sort() 
                selected_img_files = ori_img_files[start_frame:end_frame] 

                for tmp_idx in range(20):
                    dest_img_folder = os.path.join(dest_root_folder, scene_name, seq_name+"_b_"+str(tmp_idx), "egocentric_imgs")
                    if not os.path.exists(dest_img_folder):
                        break 

                for img_idx in range(len(selected_img_files)):
                    img_name = selected_img_files[img_idx]
                    ori_img_path = os.path.join(img_folder, img_name)
                    dest_img_path = os.path.join(dest_img_folder, ("%05d"%img_idx)+".png")
                    if not os.path.exists(dest_img_folder):
                        os.makedirs(dest_img_folder) 
                    shutil.copy(ori_img_path, dest_img_path)

                # Copy selected smplx parameters to new folder 
                ori_smplx_folder = os.path.join(ori_root_folder, scene_name, seq_name, "smplx_local")
                tmp_ori_smplx_files = os.listdir(ori_smplx_folder)
                ori_smplx_files = []
                for smplx_name in tmp_ori_smplx_files:
                    if ".pkl" in smplx_name:
                        ori_smplx_files.append(smplx_name) 
                ori_smplx_files.sort() 
                selected_smplx_files = ori_smplx_files[start_frame:end_frame] 

                for tmp_idx in range(20):
                    dest_smplx_folder = os.path.join(dest_root_folder, scene_name, seq_name+"_b_"+str(tmp_idx),  "smplx_local")
                    if not os.path.exists(dest_smplx_folder):
                        break 
                
                for smplx_idx in range(len(selected_smplx_files)):
                    smplx_name = selected_smplx_files[smplx_idx]
                    ori_smplx_path = os.path.join(ori_smplx_folder, smplx_name)
                    dest_smplx_path = os.path.join(dest_smplx_folder, ("%05d"%smplx_idx)+".pkl")
                    if not os.path.exists(dest_smplx_folder):
                        os.makedirs(dest_smplx_folder)
                    shutil.copy(ori_smplx_path, dest_smplx_path)

            row_cnt += 1 
           

if __name__ == "__main__":
    segment_data_w_csv()    
    # ['poses', 'root_trans', 'root_orient', 'beta']
