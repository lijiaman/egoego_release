import time
from relive.utils.torch_ext import *
from relive.core.common import *
from relive.core.agent_ego import AgentEgo


class AgentVGAIL(AgentEgo):

    def __init__(self, discrim_net=None, discrim_vs_net=None, discrim_criterion=None,
                 optimizer_discrim=None, discrim_num_update=10, **kwargs):
        super().__init__(**kwargs)
        self.discrim_net = discrim_net
        self.discrim_vs_net = discrim_vs_net
        self.discrim_criterion = discrim_criterion
        self.optimizer_discrim = optimizer_discrim
        self.discrim_num_update = discrim_num_update
        self.sample_modules += [discrim_net, discrim_vs_net]
        self.update_modules += [discrim_net, discrim_vs_net]

    def pre_sample(self):
        super().pre_sample()
        self.discrim_vs_net.set_mode('test')

    def pre_episode(self):
        super().pre_episode()
        self.discrim_vs_net.initialize(tensor(self.env.get_episode_cnn_feat()))

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = batch.v_metas

        self.policy_vs_net.set_mode('train')
        self.value_vs_net.set_mode('train')
        self.policy_vs_net.initialize((masks, self.env.cnn_feat, v_metas))
        self.value_vs_net.initialize((masks, self.env.cnn_feat, v_metas))
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states))

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        self.update_policy(states, actions, returns, advantages, exps)
        self.update_discriminator(states, masks, v_metas)

        return time.time() - t0

    def update_discriminator(self, states, masks, v_metas):
        """perform discriminator update"""
        self.discrim_vs_net.set_mode('train')
        self.discrim_vs_net.initialize((masks, self.env.cnn_feat, v_metas))
        expert_states = self.get_expert_states(v_metas, masks)
        for i in range(self.discrim_num_update):
            g_vs_out = self.discrim_vs_net(states)
            e_vs_out = self.discrim_vs_net(expert_states)
            g_o = self.discrim_net(g_vs_out)
            e_o = self.discrim_net(e_vs_out)

            self.optimizer_discrim.zero_grad()
            l_g = self.discrim_criterion(g_o, ones((g_o.size(0), 1), device=self.device))
            l_e = self.discrim_criterion(e_o, zeros((e_o.size(0), 1), device=self.device))
            discrim_loss = l_g + l_e
            discrim_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discrim_net.parameters(), 40)
            self.optimizer_discrim.step()
        e_loss = l_e.detach().item()
        return e_loss

    def get_expert_states(self, v_metas, masks):
        expert_states = []
        end_indice = np.where(masks.cpu().numpy() == 0)[0]
        v_metas = v_metas[end_indice, :]
        end_indice = np.insert(end_indice, 0, -1)
        episode_lens = np.diff(end_indice)
        for v_meta, len in zip(v_metas, episode_lens):
            exp_ind, start_ind = v_meta
            e_states = self.env.expert_arr[exp_ind]['obs'][start_ind: start_ind + len, :]
            expert_states.append(e_states)
        expert_states = np.vstack(expert_states)
        if self.running_state is not None:
            expert_states = (expert_states - self.running_state.rs.mean[None, :]) / self.running_state.rs.std[None, :]
        return torch.from_numpy(expert_states).to(self.dtype).to(self.device)
