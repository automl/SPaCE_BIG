import numpy as np
from gym import spaces
from deep_sprl.environments.contextual_point_mass import ContextualPointMass

class CPMWrapper():
    def __init__(self, instance_feats, test):
        self.instances = instance_feats
        self.inst_id = -1
        if test:
            self.instance_set_size = 1
            self.curr_set = instance_feats
            self.indices = np.arange(len(instance_feats))
        else:
            self.instance_set_size = 0.01
            self.curr_set = [self.instances[0]]
            self.indices = [0]
        self.env = ContextualPointMass()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        #self.observation_space = spaces.Box(
        #    low=-np.inf * np.ones(4), high=np.ones(4) * np.inf
        #)
        self.test = None
        self.c_step = 0

    def step(self, action):
        self.c_step += 1
        obs, r, done, info = self.env.step(action)
        if self.c_step >= 300:
            done = True
        return obs, r, done, info

    def reset(self):
        self.c_step = 0
        self.inst_id = (self.inst_id + 1) % len(self.curr_set)
        self.env = ContextualPointMass(self.curr_set[self.inst_id])
        obs = self.env.reset()
        return obs

    def get_num_instances(self):
        return len(self.instances)

    def get_instance_set(self):
        return self.indices, self.curr_set

    def get_instance_set_size(self):
        return int(np.around(len(self.instances) * self.instance_set_size))

    def increase_set_size(self):
        self.instance_set_size += 0.01

    def set_instance_set(self, indices):
        size = int(np.around(len(self.instances) * self.instance_set_size))
        if size <= 0:
            size = 1
        self.curr_set = np.array(self.instances)[indices[:size]]
        self.indices = indices
