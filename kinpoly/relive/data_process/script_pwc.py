import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
sys.path.append("./3rdparty/PWC_Net/PyTorch")

import cv2
import argparse
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
# import pwc_models
from PWC_Net.PyTorch.models.PWCNet import pwc_dc_net
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mocap-id', type=str, default=None)
parser.add_argument('--take-id', type=str, default=None)
parser.add_argument('--data_dir', action='store_true', default="/insert_directory_here/")
args = parser.parse_args()

print('model loaded...')
pwc_model_fn = './3rdparty/PWC-Net/PyTorch/pwc_net.pth.tar'
net = pwc_dc_net(pwc_model_fn)
print('model loaded...')
net = net.cuda()
net.eval()


def get_flow(im1, im2):
	im_all = [im[:, :, :3] for im in [im1, im2]]
	# rescale the image size to be multiples of 64e
	divisor = 64.
	H = im_all[0].shape[0]
	W = im_all[0].shape[1]
	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)
	for i in range(len(im_all)):
		im_all[i] = cv2.resize(im_all[i], (W_, H_))

	for _i, _inputs in enumerate(im_all):
		# im_all[_i] = im_all[_i][:, :, ::-1]   # for OpenCV, it is already BGR mode, so no need for this reversion
		im_all[_i] = 1.0 * im_all[_i]/255.0
		im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
		im_all[_i] = torch.from_numpy(im_all[_i])
		im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
		im_all[_i] = im_all[_i].float()

	im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)
	flo = net(im_all)
	flo = flo[0] * 20.0
	flo = flo.cpu().data.numpy()
	# scale the flow back to the input size 
	flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
	u_ = cv2.resize(flo[:,:,0],(W,H))
	v_ = cv2.resize(flo[:,:,1],(W,H))
	u_ *= W/ float(W_)
	v_ *= H/ float(H_)
	flo = np.dstack((u_,v_))
	return flo

def visualize_flow(flo, vis_fn):
	mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
	hsv = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=np.uint8)
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = 255
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	cv2.imwrite(vis_fn, rgb)

fpv_frames_folder = os.path.expanduser("{}/fpv_frames".format(args.data_dir))
fpv_of_folder = os.path.expanduser("{}/fpv_of".format(args.data_dir))
if not os.path.exists(fpv_of_folder):
	os.makedirs(fpv_of_folder)

if args.mocap_id is not None:
	# take_folders = [os.path.basename(x) for x in glob.glob('%s/%s_*' % (fpv_frames_folder, args.mocap_id))]
    take_folders = [os.path.basename(x) for x in glob.glob('%s/%s*' % (fpv_frames_folder, args.mocap_id))]
else:
	take_folders = [args.take_id]
take_folders.sort()
for folder in take_folders:
	print(folder)

for folder in take_folders:
	flo_folder = os.path.join(fpv_of_folder, folder)
	if not os.path.exists(flo_folder):
		os.makedirs(flo_folder)
	frames = glob.glob(os.path.join(fpv_frames_folder, folder, '*.png'))
	frames.sort()
	print(frames)
	im1 = cv2.imread(frames[0])
	for i in range(0, len(frames) - 1):
		im2 = cv2.imread(frames[i + 1])
		flo = get_flow(im1, im2)
		print(i, np.min(flo), np.max(flo))
		np.save('%s/%05d.npy' % (flo_folder, i), flo)
		visualize_flow(flo, '%s/%05d.png' % (flo_folder, i))
		im1 = im2