import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())


if __name__ == "__main__":
    fpv_base = "/insert_directory_here/fpv_raw"
    fpv_of_base = "/insert_directory_here/fpv_of"
    all_wild_captures = glob.glob(osp.join(fpv_base, "09-02*"))
    for wild_cap in all_wild_captures:
        wild_cap = wild_cap.lower()
        mocap_id = wild_cap.split("/")[-1].replace(".mp4", "")

        # if osp.isdir(osp.join(fpv_of_base, mocap_id)):
            # continue
        # if mocap_id != "2021-05-04-00-01-04":
            # continue

        try:
            if sys.version_info.major == 3:
                fpv_cmd = "python relive/data_process/process_fpv_raw.py --mocap-id " + mocap_id
                print(fpv_cmd)
                os.system(fpv_cmd)
            else:
                of_cmd = "python relive/data_process/script_pwc.py --mocap-id " + mocap_id
                os.system(of_cmd)

        except KeyboardInterrupt:
            sys.exit(0)
        