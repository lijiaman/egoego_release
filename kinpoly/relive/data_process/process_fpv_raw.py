import argparse
import os
import glob
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--mocap-id', type=str, default='1011')
parser.add_argument('--crop-size', type=int, default=1000)
parser.add_argument('--scale-size', type=int, default=224)
parser.add_argument('--v-width', type=int, default=1920)
parser.add_argument('--v-height', type=int, default=1080)
parser.add_argument('--wild', action='store_true', default=False)
parser.add_argument('--data_dir', action='store_true', default="/insert_directory_here/")

args = parser.parse_args()

v_width = args.v_width
v_height = args.v_height
c_size = args.crop_size
s_size = args.scale_size
fpv_raw_folder = f'{args.data_dir}/fpv_raw'
fpv_proc_folder = f'{args.data_dir}/fpv_process_vids'
fpv_frames_folder = f'{args.data_dir}/fpv_frames'
slam_frames_folder = f'{args.data_dir}/fpv_slams'
os.makedirs(fpv_frames_folder, exist_ok=True)
os.makedirs(slam_frames_folder, exist_ok=True)

# fpv_files = glob.glob(os.path.join(fpv_raw_folder, '%s_*.MP4' % args.mocap_id)) \
            # + glob.glob(os.path.join(fpv_raw_folder, '%s_*.mp4' % args.mocap_id))
fpv_files = glob.glob(os.path.join(fpv_raw_folder, '%s*.MP4' % args.mocap_id)) \
            + glob.glob(os.path.join(fpv_raw_folder, '%s*.mp4' % args.mocap_id))
fpv_files.sort()

for file in fpv_files:
    name = os.path.splitext(os.path.basename(file))[0]
    frame_dir = os.path.join(fpv_frames_folder, name)
    frame_slam_dir = os.path.join(slam_frames_folder, name)
    os.makedirs(frame_dir, exist_ok=True)
    if args.wild:
        os.makedirs(frame_slam_dir, exist_ok=True)

    crop_file = '%s/%s_crop.mp4' % (fpv_proc_folder, name)
    scale_file = '%s/%s_scale.mp4' % (fpv_proc_folder, name)

    cmd = ['ffmpeg', '-y', '-i', file, '-vf',
            'crop=%d:%d:%d:%d' % (c_size, c_size, v_width // 2 - c_size // 2, v_height // 2 - c_size // 2),
            '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', crop_file]
    subprocess.call(cmd)
    
    cmd = ['ffmpeg', '-y', '-i', crop_file, '-vf', 'scale=%d:-1' % s_size,
           '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', scale_file]
    subprocess.call(cmd)

    cmd = ['ffmpeg', '-i', scale_file, '-r', '30', '-start_number', '0', os.path.join(frame_dir, '%05d.png')]
    subprocess.call(cmd)
    if args.wild:
        cmd = ['ffmpeg', '-i', file, '-r', '60', '-start_number', '0', os.path.join(frame_slam_dir, '%05d.png')]
        subprocess.call(cmd)

    # cmd = ['rm', '-rf', crop_file, scale_file]
    # subprocess.call(cmd)
    