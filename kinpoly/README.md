# Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation

[[paper]](https://arxiv.org/abs/2106.05969) [[website]](https://zhengyiluo.github.io/projects/kin_poly/) [[Video]](https://www.youtube.com/watch?v=yEiK9K1N-zw)


This repo is still under construction: 
- [x] UHC eval code runnable.
- [ ] UHC training code runnable.
- [ ] Kin-poly training (Supervised) code runnable.
- [ ] Kin-poly training (RL + Supervised) code runnable.
- [ ] Kin-poly eval code runnable.

## Citation
If you find our work useful in your research, please cite our paper [kin_poly](https://zhengyiluo.github.io/projects/kin_poly/):
```
@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Introduction

In this project, we demonstrate the ability to estimate 3D human pose and human-object interactions from egocentric videos. This code base contains all the necessary files to train and reproduce the results reported in our paper, and contain configuration files and hyperparameters used in our experiments. Some training data (namely, [AMASS](https://amass.is.tue.mpg.de/)) and external library ([Mujoco](http://www.mujoco.org/)) may require additional licence to obtain, and this codebase contains data processing scripts to process these data once obtained. 

Notice that internally, we call the task of **Egocentric Pose Estimation** "relive", as in "reliving your past experiences through egocentric view", so all the code developed for egocentric pose estimation is contained in the folder called "relive" (which is the project name). We develop the Universal Humanoid Controller independently, under the project name of "copycat", as in "mimicking and copying target pose". Thus, the two main folders for this project is "relive" and "coypcat". 

## Dependencies

The environment we used for running this project can be found in ```requirements.txt```. Notice that to estimate physically plausible human pose the [Mujoco](http://www.mujoco.org/) physics simulation is needed to train and evaluate our code.

## Datasets

The datasets we use for training and evaluating our method can be found here:

[[Real-world dataset](https://drive.google.com/drive/folders/1BBjPmjrm-FZLMw24Gsbl4CsodGgfsptY?usp=sharing)][[MoCap dataset](https://drive.google.com/drive/folders/1Mw1LQBNfor8a7Diw3eHLO--ZnREw57kB?usp=sharing)]

The folders contain the a data file that contains the pre-computed object pose and camera trajectory; another data file contains the pre-computed image features; a meta file is also included for loading the respective datasets.

To download the Mocap dataset, run the following script: 

```
bash download_data.sh
```

## Important files

* ```relive/models/traj_ar_smpl_net.py```:  definition of our kinematic model's network.
* ```relive/models/policy_ar.py```:  wrapper around our kinematic model to form the kinematic policy.
* ```relive/envs/humanoid_ar_v1.py```: main Mujoco environment for training and evaluating our kinematic policy.
* ```scripts/eval_pose_all.py```: evaluation code that computes all metrics reported in our paper from a pickled result file. 
* ```config/kin_poly.yml```: the configuration file used to train our kinematic policy.
* ```copycat/cfg/copycat.yml```: the configuration file used to train our universal humanoid controller.
* ```assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml```: the simulation configuration used in our experiments. It contains the definition for the humanoid and the objects (chair, box, table, etc.) for Mujoco. 

## Training

To train our dynamics-regulated kinematic policy, use the command:

```
python scripts/train_ar_policy.py --cfg kin_poly  --num_threads 35 
```

To train our kinematic policy using only supervised learning, use the command:

```
python scripts/exp_arnet_all.py.py --cfg kin_poly  
```

To train our universal humanoid controller, use the command:

```
python scripts/train_copycat.py.py --cfg copycat --num_threads 35
```

## Evaluation

To evaluate our dynamics-regulated kinematic policy, run:
```
python scripts/eval_ar_policy.py --cfg kin_poly --iter 1000  
```

To compute metrics, run:
```
python scripts/eval_pose_all.py --cfg kin_poly --algo kin_poly --iter 1000
```

To evaluate our universal humanoid controller, run:
```
python scripts/eval_copycat.py --cfg copycat --iter 10000
```

*Note that additional directory fixup may be needed for running these commands. Directorys that needs updating are named "/insert_directory_here/"*

## Pre-trained models

[[Kinematic policy](https://drive.google.com/file/d/1oQZzWVfWPrGzX0XyB0k4h7z6WLtSEsjX/view?usp=sharing)][[Universal Humanoid Controller](https://drive.google.com/file/d/1Hw2E8H0hHx9JwQXNsmWM0OjE1XTgFkmd/view?usp=sharing)]

## Licence

To be updated. 
