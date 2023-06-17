import multiprocessing
import math
from relive.utils.tools import fix_height
import time
import os
import torch
os.environ["OMP_NUM_THREADS"] = "1"
from collections import defaultdict
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import glob
import os
import sys
import pdb
import os.path as osp
import joblib
sys.path.append(os.getcwd())

from copycat.khrylib.rl.core import LoggerRL
from copycat.khrylib.utils.memory import Memory
from copycat.khrylib.utils.torch import *
from copycat.khrylib.rl.core import estimate_advantages
from relive.core.trajbatch_ego import TrajBatchEgo
from copycat.khrylib.rl.agents import AgentPPO
from relive.data_loaders.statear_smpl_dataset import StateARDataset
from relive.utils.flags import flags




class AgentAR(AgentPPO):

    def __init__(self, cfg, env_wild = None, wild = False,  test_time = False, training = True, fit_ind = 0, checkpoint_epoch = 0,  **kwargs):
        super().__init__( **kwargs)
        self.cfg = cfg
        self.wild = wild
        self.training = training
        self.traj_cls = TrajBatchEgo
        self.test_data_loaders = []
        self.setup_data_loader()
        # self.setup_env() # ZL: in progress
        # self.setup_policy() # ZL: in progress
        self.fit_ind = fit_ind
        self.test_time = test_time
        self.env_wild = env_wild
        
        freq_path = osp.join(cfg.result_dir, f"freq_dict_{'wild_' if wild else '' }test.pt") if test_time else osp.join(cfg.result_dir, "freq_dict.pt")
        try:
            self.freq_dict = {k:[] for k in self.data_loader.takes} if not osp.exists(freq_path) else joblib.load(freq_path)
        except:
            print("error parsing freq_dict, using empty one")
            self.freq_dict = {k:[] for k in self.data_loader.takes}
        print("******************************")
        print(f"test_time: {test_time}")
        print(f"fit_ind: {fit_ind}")
        
        print(f"sampling temp: {self.cfg.policy_specs.get('sampling_temp', 0.5)}")
        print(f"sampling freq: {self.cfg.policy_specs.get('sampling_freq', 0.5)}")
        print(f"init_update_iter: {self.cfg.policy_specs.get('num_init_update', 3)}")
        print(f"step_update_iter: {self.cfg.policy_specs.get('num_step_update', 10)}")
        print("******************************")

        if not self.test_time and checkpoint_epoch == 0:
            self.policy_net.update_init_supervised(self.cfg, self.data_loader, device = self.device , dtype = self.dtype, num_epoch = self.cfg.policy_specs.get('warm_update_init', 500))
            self.policy_net.train_full_supervised(cfg, self.data_loader, device = self.device, dtype = self.dtype, scheduled_sampling = 0.3, num_epoch = self.cfg.policy_specs.get('warm_update_full', 50))
            self.policy_net.setup_optimizers()
            self.save_checkpoint(checkpoint_epoch)
    
    def setup_env(self):
        cfg = self.cfg

    def setup_policy(self):
        cfg = self.cfg 
        data_sample = self.data_loader.sample_seq()
        policy_net = PolicyAR(cfg, data_sample, device= self.device, dtype = self.dtype)


    def setup_data_loader(self):
        cfg = self.cfg
        self.test_data_loaders.append(StateARDataset(cfg, "test"))

        # ZL: todo, fix this junk 
        from relive.utils.statear_smpl_config import Config
        cfg_wild = Config("all", cfg.id, wild = True, create_dirs=False, mujoco_path = "%s.xml")
        self.test_data_loaders.append(StateARDataset(cfg_wild, "test"))



    def load_checkpoint(self, i_iter):
        if i_iter > 0:
            if self.wild:
                if self.test_time: cp_path = '%s/iter_wild_test_%04d.p' % (cfg.policy_model_dir, args.iter )
                else: cp_path = '%s/iter_wild_%04d.p' % (cfg.policy_model_dir, args.iter)
            else:
                if self.test_time: cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)
                else: cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)

            if not osp.exists(cp_path):
                cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, args.iter)

            # cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, args.iter)
            # cp_path = f'{cfg.policy_model_dir}/iter_test_6270.p'

            logger.info('loading model from checkpoint: %s' % cp_path)
            model_cp = pickle.load(open(cp_path, "rb"))
            policy_net.load_state_dict(model_cp['policy_dict'])
            
            # policy_net.old_arnet[0].load_state_dict(copy.deepcopy(policy_net.traj_ar_net.state_dict())) # ZL: should use the new old net as well
            
            value_net.load_state_dict(model_cp['value_dict'])
            running_state = model_cp['running_state']

    def save_checkpoint(self, i_iter):
        cfg = self.cfg
        # self.tb_logger.flush()
        policy_net, value_net, running_state = self.policy_net, self.value_net, self.running_state
        with to_cpu(policy_net, value_net):
            if self.wild:
                if self.test_time: cp_path = '%s/iter_wild_test_%04d.p' % (cfg.policy_model_dir, i_iter + 1)
                else: cp_path = '%s/iter_wild_%04d.p' % (cfg.policy_model_dir, i_iter + 1)
            else:
                if self.test_time: cp_path = '%s/iter_test_%04d.p' % (cfg.policy_model_dir, i_iter + 1)
                else: cp_path = '%s/iter_%04d.p' % (cfg.policy_model_dir, i_iter + 1)

            model_cp = {'policy_dict': policy_net.state_dict(),
                        'value_dict': value_net.state_dict(),
                        'running_state': running_state}
            pickle.dump(model_cp, open(cp_path, 'wb'))


    def next_fit_seq(self):
        self.fit_ind += 1
        if self.fit_ind == self.data_loader.len:
            exit()
        context_sample = self.data_loader.get_seq_by_ind(self.fit_ind, full_sample = True)

    def eval_policy(self, data_mode = "train", i_iter = 0):
        cfg = self.cfg
        if data_mode == "train":
            data_loaders = [self.data_loader]
        elif data_mode == "test":
            data_loaders = self.test_data_loaders


        res_msgs = [] 
        for data_loader in data_loaders:
            coverage = 0
            num_jobs = self.num_threads
            jobs = list(range(data_loader.get_len()))
            np.random.shuffle(jobs)

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
                # print(res["percent"], data_loader.takes[k])
                if res["percent"] == 1: 
                    coverage += 1
                    if data_mode == "train":
                        [self.freq_dict[data_loader.takes[k]].append([res['percent'], 0]) for _ in range(1)]
                else:
                    if data_mode == "train":
                        [self.freq_dict[data_loader.takes[k]].append([res['percent'], 0]) for _ in range(3)]

            eval_path = osp.join(cfg.result_dir, f"eval_dict_{'wild_' if self.wild else '' }test.pt") if self.test_time else osp.join(cfg.result_dir, f"eval_dict_{data_mode}.pt")
            eval_dict = joblib.load(eval_path) if osp.exists(eval_path) else defaultdict(list)
            eval_dict[i_iter] = {data_loader.takes[k]: v["percent"] for k, v in data_res_coverage.items()}
            joblib.dump(eval_dict, eval_path)
            res_msgs.append(f"Coverage {data_mode} of {coverage} out of {data_loader.get_len()}")

        return res_msgs


    def eval_seqs(self, fit_ids, data_loader, queue):
        res = {}
        for cur_id in fit_ids:
            res[cur_id]  = self.eval_seq(cur_id, data_loader)

        if queue == None:
            return res  
        else:
            queue.put(res)

    def eval_cur_seq(self):
        return self.eval_seq(self.fit_ind, self.data_loader)

    def eval_seq(self, fit_ind, loader):
        curr_env = self.env if not loader.cfg.wild  else self.env_wild
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                self.policy_net.set_mode('test')
                curr_env.set_mode('test')
                
                context_sample = loader.get_seq_by_ind(fit_ind, full_sample = True)
                self.ar_context = ar_context = self.policy_net.init_context(context_sample)

                curr_env.load_context(ar_context)
                state = curr_env.reset()
                
                if self.running_state is not None:
                    state = self.running_state(state)
                for t in range(10000):
                    res['target'].append(curr_env.target['qpos'])
                    res['pred'].append(curr_env.get_humanoid_qpos())
                    res['obj_pose'].append(curr_env.get_obj_qpos())

                    state_var = tensor(state).unsqueeze(0)
                    trans_out = self.trans_policy(state_var)
                    
                    action = self.policy_net.select_action(trans_out, mean_action = True)[0].numpy()
                    action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                    next_state, env_reward, done, info = curr_env.step(action)
                    
                    
                    # c_reward, c_info = self.custom_reward(curr_env, state, action, info)
                    # res['reward'].append(c_reward)
                    # curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}
                        # print(info['percent'], ar_context['ar_qpos'].shape[1], loader.curr_key, np.mean(res['reward']))
                        res['percent'] = info['percent']
                        return res
                    state = next_state

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls()
        self.policy_net.set_mode('test')
        self.env.set_mode('train')
        freq_dict = defaultdict(list)

        while logger.num_steps < min_batch_size:
            if self.test_time:
                context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp =  self.cfg.policy_specs.get("sampling_temp", 0.5), sampling_freq = self.cfg.policy_specs.get("sampling_freq", 0.9), full_sample = True if self.data_loader.get_seq_len(self.fit_ind) < 1000 else False)
                # context_sample = self.data_loader.get_seq_by_ind(self.fit_ind, full_sample = True if self.data_loader.get_seq_len(self.fit_ind) < 1000 else False)
                # context_sample = self.data_loader.get_seq_by_ind(self.fit_ind, full_sample = True)
            else:
                context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = self.cfg.policy_specs.get("sampling_temp", 0.5), sampling_freq = self.cfg.policy_specs.get("sampling_freq", 0.9))
                # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = self.cfg.policy_specs.get("sampling_temp", 0.5), sampling_freq = self.cfg.policy_specs.get("sampling_freq", 0.9), full_sample = True if self.data_loader.get_seq_len(self.fit_ind) < 1000 else False)
                # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = 0.5)
                # context_sample = self.data_loader.sample_seq()

            ar_context = self.policy_net.init_context(context_sample, fix_height = False) # should not try to fix the height during training!!!

            self.env.load_context(ar_context)
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
                #################### ZL: Jank Code.... ####################
                if self.test_time:
                    gt_qpos = self.env.ar_context['ar_qpos'][self.env.cur_t + 1]
                else:
                    gt_qpos = self.env.ar_context['qpos'][self.env.cur_t + 1]
                curr_qpos = self.env.get_humanoid_qpos()
                #################### ZL: Jank Code.... ####################
                
                next_state, env_reward, done, info = self.env.step(action)
                res_qpos = self.env.get_humanoid_qpos()
                

                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                
                if flags.debug:
                    np.set_printoptions(precision=4, suppress=1)
                    print(c_reward, c_info)

                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp, gt_qpos, curr_qpos, res_qpos)

                if pid == 0 and self.render:
                    self.env.render()
                if done:
                    freq_dict[self.data_loader.curr_key].append([info['percent'], self.data_loader.fr_start])
                    # print(self.data_loader.curr_key, info['percent'])
                    break

                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()
        

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict



    def push_memory(self, memory, state, action, mask, next_state, reward, exp, gt_target_qpos, curr_qpos, res_qpos):
        v_meta = np.array([self.data_loader.curr_take_ind, self.data_loader.fr_start, self.data_loader.fr_num])
        memory.push(state, action, mask, next_state, reward, exp, v_meta, gt_target_qpos, curr_qpos, res_qpos)

    # def push_memory(self, memory, state, action, mask, next_state, reward, exp):
    #     v_meta = np.array([self.data_loader.curr_take_ind, self.data_loader.fr_start, self.data_loader.fr_num])
    #     memory.push(state, action, mask, next_state, reward, exp, v_meta)
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
                
                self.freq_dict = {k: v if len(v) < 5000 else v[-5000:] for k, v in self.freq_dict.items()}
                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = torch.from_numpy(batch.v_metas).to(self.dtype).to(self.device)
        gt_target_qpos = torch.from_numpy(batch.gt_target_qpos).to(self.dtype).to(self.device)
        curr_qpos = torch.from_numpy(batch.curr_qpos).to(self.dtype).to(self.device)
        res_qpos = torch.from_numpy(batch.res_qpos).to(self.dtype).to(self.device)
        

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states[:, :self.policy_net.state_dim]))

        self.policy_net.set_mode('train')
        self.policy_net.initialize_rnn((masks, v_metas))
        """get advantage estimation from the trajectories"""
        print("==================================================>")

        if self.cfg.policy_specs.get("rl_update", False):
            print("RL:")
            advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)
            self.update_policy(states, actions, returns, advantages, exps)
        
        if not self.test_time:
            if self.cfg.policy_specs.get("init_update", False) or self.cfg.policy_specs.get("step_update", False) or self.cfg.policy_specs.get("full_update", False): print("Supervised:")
            if self.cfg.policy_specs.get("init_update", False):
                self.policy_net.update_init_supervised(self.cfg, self.data_loader, device = self.device , dtype = self.dtype, num_epoch = int(self.cfg.policy_specs.get("num_init_update", 5)))
            if self.cfg.policy_specs.get("step_update", False):
                self.policy_net.update_supervised(states, gt_target_qpos, curr_qpos, num_epoch = int(self.cfg.policy_specs.get("num_step_update", 10)))

            if self.cfg.policy_specs.get("step_update_dyna", False):
                self.policy_net.update_supervised_dyna(states, res_qpos, curr_qpos, num_epoch = int(self.cfg.policy_specs.get("num_step_dyna_update", 10)))
                
            if self.cfg.policy_specs.get("full_update", False):
                self.policy_net.train_full_supervised(self.cfg, self.data_loader, device = self.device, dtype = self.dtype, num_epoch = 1, scheduled_sampling = 0.3)
            self.policy_net.step_lr()
            
        # else:
            # self.policy_net.update_supervised(states, gt_target_qpos, curr_qpos, num_epoch = 2)


        return time.time() - t0


    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            # trans_value = self.trans_value(states[:, :self.policy_net.obs_lim])
            trans_value = self.trans_value(states)
            
            values_pred = self.value_net(trans_value)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        pbar = tqdm(range(self.opt_num_epochs))
        for _ in pbar:
            ind = exps.nonzero(as_tuple=False).squeeze(1)
            self.update_value(states, returns)
            surr_loss, ratio = self.ppo_loss(states, actions, advantages, fixed_log_probs, ind)
            self.optimizer_policy.zero_grad()
            surr_loss.backward()
            self.clip_policy_grad()
            self.optimizer_policy.step()
            pbar.set_description_str(f"PPO Loss: {surr_loss.cpu().detach().numpy():.3f}| Ration: {ratio.mean().cpu().detach().numpy():.3f}")

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss, ratio

    def action_loss(self, actions, gt_actions):
        pass
