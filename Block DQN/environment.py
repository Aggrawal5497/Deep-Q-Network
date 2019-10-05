# -*- coding: utf-8 -*-

import gym
import cv2
import numpy as np
from collections import deque
from network import ToTensor, PrepareData, QNetwork, PrepareNextStateData
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optimizers

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action, times = 4):
        states = []
        treward = 0
        for _ in range(times):
            next_state, reward, done, info = self.env.step(action)
            states.append(next_state)
            treward += reward
        
        nstates = np.array(states)
        nstates = nstates.transpose((1, 2, 0))
        return nstates, reward, done, info
    
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        img = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        img = img[50:, :] / 255.0
        return img

def sampleRandomSamples(num_samples = 64):
    return np.random.choice(list(range(5000)), size=num_samples)

def learn(replay_mem, gamma = 0.98):
    samples = sampleRandomSamples()
    epoch_data = [replay_mem[val] for val in samples]
    s1, actions, rewards, dones, s2 = map(np.array, zip(*epoch_data))
    
    y = rewards
    episode_not_ends = np.where(dones == False)
    
    nstatedataset = PrepareNextStateData(s2[episode_not_ends], transforms.Compose([ToTensor()]))
    nstateDataloader = DataLoader(nstatedataset, batch_size = len(s2), shuffle = False)
    
    for idx, states in enumerate(nstateDataloader):
        action = target_net(states.cuda())
        action = action.max(1).values
    
    y[episode_not_ends] = y[episode_not_ends] + gamma * action.detach().cpu().numpy()
    
    dataset = PrepareData(s1, actions, y, s2, transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    t_loss = 0
    for i, batch in enumerate(dataloader):
        opt.zero_grad()
        q_value = network(batch['states'].cuda())
        Q_s_a = q_value.gather(1, batch['actions'].cuda())
        loss_value = loss(Q_s_a, batch['y_value'].cuda())
        loss_value.backward()
        opt.step()
        t_loss += loss_value.cpu().item()
        
    return t_loss

replay_buffer = deque(maxlen = 5000)
network = QNetwork((160, 160, 4), 4)
network = network.cuda()

target_net = QNetwork((160, 160, 4), 4)
target_net = target_net.cuda()
target_net.load_state_dict(network.state_dict())
target_net.eval()

loss = torch.nn.SmoothL1Loss()
opt = optimizers.Adam(network.parameters(), lr=1e-05)
epsilon = 1.0
decay = 0.99

convert_to_tensor = ToTensor()

env = gym.make('Breakout-v0')
env = ObservationWrapper(env)
env = BasicWrapper(env)
for i in range(1000):
    start_state = env.reset() 
    current_state = np.stack([start_state for _ in range(4)], axis=-1)
    treward = 0
    tloss = 0
    k = 0
    while True:
        if np.random.randn() < epsilon:
            action = env.action_space.sample()
        else:
            x = convert_to_tensor(current_state)
            action = network(x.unsqueeze(0).cuda())
            action = action.argmax()
            action = action.detach().cpu().item()
        next_state , reward, done, _ = env.step(action)
        treward += reward
        replay_buffer.append((current_state, action, reward, done, next_state))
        
        current_state = next_state
        
        if len(replay_buffer) == 5000:
            l = learn(replay_buffer)
            tloss += l

        k += 1   
        if done:
            break
    if len(replay_buffer) == 5000:
        epsilon = epsilon * decay
    print("iteration : {}, Episode reward : {}, Loss : {:.5f}".format(i+1, treward, tloss / k))
        
    
    if (i+1) % 20 == 0 and len(replay_buffer) == 5000:
        target_net.load_state_dict(network.state_dict())
        
        start_state = env.reset() 
        #env.render()
        current_state = np.stack([start_state for _ in range(4)], axis=-1)
        treward = 0
        while True:
            x = convert_to_tensor(current_state)
            action = target_net(x.unsqueeze(0).cuda())
            action = action.argmax()
            action = action.detach().cpu().item()
            next_state , reward, done, _ = env.step(action)
            treward += reward  
            current_state = next_state
            #env.render()
            if done:
                break
        env.close()
        print("Current achievable reward : {}".format(treward))

