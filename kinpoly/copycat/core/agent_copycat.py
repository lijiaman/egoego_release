import multiprocessing
import math
import time
import os
import torch
os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import pickle
from collections import defaultdict
import glob
import os
import sys
import os.path as osp

from copycat.khrylib.rl.core import LoggerRL
from copycat.khrylib.utils.memory import Memory
from copycat.khrylib.utils.torch import *
from copycat.khrylib.rl.core import estimate_advantages
from copycat.khrylib.rl.agents import AgentPPO
from tqdm import tqdm

class AgentCopycat(AgentPPO):

    def __init__(self, cfg,  **kwargs):
        super().__init__( **kwargs)
        self.cfg = cfg
        freq_path = osp.join(cfg.output_dir, f"freq_dict.pt")
        self.freq_dict = {k:[] for k in self.data_loader.data_keys} if not osp.exists(freq_path) else joblib.load(freq_path)
        # self.freq_dict = {k:[] for k in self.data_loader.data_keys}
    
    def save_checkpoint(self, epoch):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = '%s/iter_%04d.p' % (cfg.model_dir, epoch + 1)
            model_cp = {'policy_dict': self.policy_net.state_dict(),
                        'value_dict': self.value_net.state_dict(),
                        'running_state': self.running_state}
            pickle.dump(model_cp, open(cp_path, 'wb'))
            joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def eval_policy(self, data_mode = "train", i_iter = 0):
        cfg = self.cfg
        if data_mode == "train":
            data_loader = self.data_loader
            
        coverage = 0
        num_jobs = self.num_threads
        jobs = data_loader.data_keys
        chunk = np.ceil(len(jobs)/num_jobs).astype(int)
        jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]

        data_res_coverage = {}
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                queue = multiprocessing.Queue()
                for i in range(len(jobs) - 1):
                    worker_args = (jobs[i+1], data_loader, queue)
                    worker = multiprocessing.Process(target=self.eval_seqs, args=worker_args)
                    worker.start()
                res = self.eval_seqs(jobs[0], data_loader, None)
                data_res_coverage.update(res)
                for i in tqdm(range(len(jobs)-1)):
                    res = queue.get()
                    data_res_coverage.update(res)
        
        
            for k, res in data_res_coverage.items(): 
                
                if res["percent"] == 1: 
                    coverage += 1
                    [self.freq_dict[k].append([res['percent'], 0]) for _ in range(1) if k in self.freq_dict] # full samples are more important...
                else:
                    [self.freq_dict[k].append([res['percent'], 0]) for _ in range(3) if k in self.freq_dict] # full samples are more important...
                
            eval_path = osp.join(cfg.output_dir, f"eval_dict_{data_mode}.pt")
            eval_dict = joblib.load(eval_path) if osp.exists(eval_path) else defaultdict(list)
            eval_dict[i_iter] = {k: v["percent"] for k, v in data_res_coverage.items()}
            joblib.dump(eval_dict, eval_path)

        return f"Coverage {data_mode} of {coverage} out of {data_loader.get_len()}"


    def eval_seqs(self, take_keys, data_loader, queue):
        res = {}
        for take_key in take_keys:
            res[take_key]  = self.eval_seq(take_key, data_loader)

        if queue == None:
            return res  
        else:
            queue.put(res)

    def eval_cur_seq(self):
        return self.eval_seq(self.fit_ind, self.data_loader)

    def eval_seq(self, take_key, loader):
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                self.env.load_expert(loader.get_sample_from_key(take_key = take_key, full_sample = True, fr_start = 0))
                state = self.env.reset()
                
                if self.running_state is not None:
                    state = self.running_state(state)

                for t in range(10000):
                    epos = self.env.get_expert_attr('qpos', self.env.get_expert_index(t)).copy()
                    res['gt'].append(epos)
                    res['pred'].append(self.env.data.qpos.copy())

                    state_var = tensor(state).unsqueeze(0).clone()
                    action = self.policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()

                    next_state, env_reward, done, info = self.env.step(action)
                    
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    res['reward'].append(c_reward)
                    # self.env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state, update=False)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}
                        res['percent'] = info['percent']
                        return res
                    state = next_state


    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        
        if hasattr(self.env, 'np_random'):
            self.env.np_random.rand(pid)
        memory = Memory()
        logger = self.logger_cls()
        freq_dict = defaultdict(list)
        while logger.num_steps < min_batch_size:
            # self.env.load_expert(self.data_loader.sample_seq(freq_dict = self.freq_dict, full_sample = False))
            self.env.load_expert(self.data_loader.sample_seq(freq_dict = self.freq_dict, full_sample = True))
            
            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                next_state, env_reward, done, info = self.env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                    
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if pid == 0 and self.render:
                    self.env.render()
                if done:
                    freq_dict[self.data_loader.curr_key].append([info['percent'], self.data_loader.fr_start])
                    break
                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict
    

    def sample(self, min_batch_size):
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules)
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads-1):
                    worker_args = (i+1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0], freq_dict = self.sample_worker(0, None, thread_batch_size)
                self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger, freq_dict = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                    
                    self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                self.freq_dict = {k: v if len(v) < 5000 else v[-5000:] for k, v in self.freq_dict.items()}
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger