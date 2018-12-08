# -*- coding: utf-8 -*-
# Python2 compatibility
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

from gym.utils import seeding
import numpy as np
import os.path as path
import pyglet
import copy
import uuid
import time
import gym
import os
import gym.spaces
import pybullet_envs

from plots.plots import ScatterPlot, QuiverPlot, Plot, SurfacePlot

class PointMass(gym.Env):
    '''
    Just a simple 2D PointMass with a jet, trying to go towards a goal location
    '''
    max_speed = 5.
    max_torque = 5.
    max_position = 20.
    treshold = 2.
    dt = .1
    mass = .2
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }
    reward_styles = {
        'velocity': dict(vel=1, pos=0, goal=100, pexp=0, pot=0, const=0 ),
        'distsq'  : dict(vel=0, pos=1, goal=0,   pexp=0, pot=0, const=0 ),
        'distsq+g': dict(vel=0, pos=1, goal=5,   pexp=0, pot=0, const=0 ),
        'distexp' : dict(vel=0, pos=0, goal=0,   pexp=1, pot=0, const=0 ),
        'distsq+e': dict(vel=0, pos=1, goal=0,   pexp=1, pot=0, const=0 ),
        'pot-mvel': dict(vel=0, pos=0, goal=0,   pexp=0, pot=1, const=-5),
        'pot'     : dict(vel=0, pos=0, goal=0,   pexp=0, pot=1, const=0),
    }

    def __init__(self, max_steps=100, randomize_goal=True, writer=None, reset=True, reward_style='velocity'):
        self.max_steps = max_steps
        self.randomize_goal = randomize_goal
        self.images = []  # for saving video if needed
        
        if reward_style in self.reward_styles:
            self.reward_style = self.reward_styles[reward_style]
        else:
            raise ValueError(
                'Incorrent `reward_style` argument %s. Should be one of [%s]'
                % (reward_style, ', '.join(self.reward_styles.keys()))
            )

        self.writer = writer

        self.viewer = None
        self.plot = None

        # self.obs_size = 2
        self.act_size = 1
        
        high_action = self.max_torque * np.ones(self.act_size)
        self.action_space = gym.spaces.Box(low=-high_action, high=high_action, dtype=np.float32)

        self.obs_high = np.concatenate([self.max_position * np.ones(2), self.max_speed * np.ones(1)])
        self.observation_space = gym.spaces.Box(low=-self.obs_high, high=self.obs_high, dtype=np.float32)

        self.seed()
        if reset:
            self.reset()
    
    # @property
    # def observation_space(self):
    #     high_position = np.concatenate([self.max_position * np.ones(4), self.max_speed * np.ones(2)])
    #     return gym.spaces.Box(low=-high_position, high=high_position, dtype=np.float32)

    # @property
    # def action_space(self):
    #     high_action = self.max_torque * np.ones(self.act_size)
    #     return gym.spaces.Box(low=-high_action, high=high_action, dtype=np.float32)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def step(self, u):
        self.i_step += 1

        # state: (px, py, gx, gy, vx, vy)
        p = self.state[0:1]
        g = self.state[1:2]
        v = self.state[2:3]
        
        reward = 0

        u = np.array(u)  # PyTorch poblems: fixing it from both sides

        if np.linalg.norm(u) > self.max_torque:
            u = u / np.linalg.norm(u) * self.max_torque
        
        self.last_u = u

        p_pos = p

        p, v = self._integrate(p, v, u)

        distance = np.linalg.norm(g-p)
        reward += self.reward_style['vel']  * (np.dot(v, g-p) / distance - .001*(np.linalg.norm(u)**2))
        reward -= self.reward_style['pos']  * (distance / self.max_position) ** 2
        reward += self.reward_style['pexp'] * np.exp(-1 * distance)
        reward += self.reward_style['pot'] * (np.linalg.norm(g-p_pos[0:1]) - distance) / self.max_speed
        reward += self.reward_style['const']



        reached = distance < self.treshold
        if reached:
            reward += self.reward_style['goal']
        
        # done = reached or (self.i_step >= self.max_steps)
        done = False

        self.state = np.concatenate([p, g, v])
        return self._get_obs(), float(reward), done, {'termination': 'time'}

    def _integrate(self, p, v, u):
        # just a simple (dumb) explicit integration ... 
        v = v + u * self.dt / self.mass
        if np.linalg.norm(v) > self.max_speed:
            v = v / np.linalg.norm(v) * self.max_speed

        p = np.clip(p + v * self.dt + u * self.dt * self.dt / 2, -self.max_position, self.max_position)

        return p, v
    
    def save_video_if_possible(self):
        import PIL
        import cv2

        if self.images:
            try:
                os.makedirs('vids')
            except:
                pass
            
            vidsize = (self.images[0].width, self.images[0].height)

            vout = cv2.VideoWriter(
                'vids/pointmass__recording__%s__%s.avi' % (
                    time.strftime("%Y-%m-%d_%H-%M-%S"),
                    str(uuid.uuid4())[0:6]
                ),
                cv2.VideoWriter_fourcc(*'MPEG'),
                int(1/self.dt + 1e-3),
                vidsize
            )
            for image in self.images:
                vout.write(
                    np.array(
                        PIL.Image.frombytes(
                            'RGB',
                            vidsize,
                            image.get_data('BGR', -(image.width * len('RGB')))
                        )
                    )
                )
            vout.release()

    def reset(self):
        self.images = []  # clean up the recorder
        high = self.obs_high
        self.state = self.np_random.uniform(low=-high, high=high)
        if np.linalg.norm(self.state[-1:]) > self.max_speed:
            self.state[-1:] = self.state[-1:] / np.linalg.norm(self.state[-1:]) * self.max_speed
        if not self.randomize_goal:
            self.state[1:2] = 0
        self.last_u = np.array([0])
        self.i_step = 0
        # self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
        # theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
    
    def _get_geoms(self):
        from gym.envs.classic_control import rendering
        point = rendering.make_circle(1)
        point.set_color(.1, .8, .3)
        self.point_transform = rendering.Transform()
        point.add_attr(self.point_transform)

        goal = rendering.make_circle(1)
        goal.set_color(.9, .1, .1)
        self.goal_transform = rendering.Transform()
        goal.add_attr(self.goal_transform)
        fname = path.join(path.dirname(__file__), "assets/arrow.png")
        self.img = rendering.Image(fname, 1., 1.)
        self.img_trans = rendering.Transform()
        self.img.add_attr(self.img_trans)

        return [point, goal]
    
    def _do_transforms(self):
        self.point_transform.set_translation(self.state[0], 0)
        self.goal_transform.set_translation(self.state[1], 0)
        # if self.last_u:
        self.img_trans.set_translation(self.state[0], 0)
        self.img_trans.set_rotation(np.arctan2(0, self.last_u[0]))
        scale = np.linalg.norm(self.last_u) / self.max_torque * 2
        self.img_trans.set_scale(scale, scale)


    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.max_position, self.max_position, -self.max_position, self.max_position)
            for geom in self._get_geoms():
                self.viewer.add_geom(geom)

        self.viewer.add_onetime(self.img)

        self._do_transforms()
        # import time
        # time.sleep(self.dt)

        self.viewer.render(return_rgb_array = mode=='rgb_array')
        self.images.append(pyglet.image.get_buffer_manager().get_color_buffer().get_image_data())

    def visualize_solution(self, policy=None, value_func=None, i_iter=None):
        '''
            @brief Visualizes policy and value functions
            the policy/value are visualized only for states where goal = v = 0

            @param policy: a function that takes the state as input and outputs the action (as numpy array)
            @param valud_func: a function that takes the state as input and outputs the value (as float)
        '''
        nb_points = 12
        xlim = [self.observation_space.low[0], self.observation_space.high[0]]
        vlim = [self.observation_space.low[2], self.observation_space.high[2]]

        if self.plot is None:
            self.plot = Plot(1,2)
            self.splot = SurfacePlot(
                parent=self.plot,
                xlim=xlim, ylim=vlim,
                value_range=[-self.max_torque*2, self.max_torque*2]
            )
            self.qplot = QuiverPlot(parent=self.plot, xlim=xlim, ylim=vlim)
        
        x = np.linspace(xlim[0], xlim[1], nb_points)
        v = np.linspace(vlim[0], vlim[1], nb_points)
        points = np.array(np.meshgrid(x,v)).transpose().reshape((-1,2))
        val = np.ones(points.shape[0])
        d = np.zeros((points.shape[0], 2))
        for i, p in enumerate(points):
            # state = np.concatenate([p, [0] * (self.observation_space.shape[0] - 2)])
            state = np.concatenate([p[0:1], [0], p[1:2]])
            if value_func is not None:
                val[i] = value_func(state)
            if policy is not None:
                d[i][0] = policy(state)

        X, V = np.meshgrid(x,v)
        self.splot.update(X, V, val.reshape(len(x), -1))
        self.qplot.update(points, d)
        
        # if i_iter is not None and self.writer is not None:
        #     self.writer.add_image('Vis/Nets', self.plot.get_image(), i_iter)

    def close(self):
        if self.viewer:
            self.viewer.close()
