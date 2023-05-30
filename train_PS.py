# code for parameter sampling 

import env
import copy
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os
import json

epi = 2000
CFenv = env.ConnectFourEnv()  # connect4 환경 생성
opAgent = env.ConnectFourRandomAgent()  # 상대 agent



def dict2json(data, filename='parameter sampling.json'):

    with open('loss_plot/'+filename, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results = {}
for i in range(optimization_trial):
    
    CFenv.reset()
    # 탐색한 하이퍼파라미터의 범위 지정===============
    lr = 10 ** np.random.uniform(-5, -2)
    batch_size = int(2 ** np.random.uniform(4, 10))
    target_update = int(10 ** np.random.uniform(0, 4))
    memory_len = int(2 ** np.random.uniform(10, 14))
    repeat_reward = int(10 ** np.random.uniform(0,1))
    
                
    # ================================================
    Qagent = env.ConnectFourDQNAgent(lr=lr, batch_size=batch_size, target_update=target_update, memory_len=memory_len, repeat_reward=repeat_reward)

    Qagent.train(epi=epi, env=CFenv, op_model=opAgent)

    plt.clf()
    plt.plot(Qagent.losses)
    plt.savefig('loss_plot/train_PS_loss_{}.png'.format(i))
    
    win, loss, draw = env.compare_model(Qagent, opAgent, n_battle=100)

    win_rate = win / (win + loss + draw)
    results[(
        "order:"+str(i),
        "lr:"+str(lr), 
        "batch_size:"+str(batch_size), 
        "target_update:"+str(target_update), 
        "memory_len:"+str(memory_len), 
        "repeat_reward:"+str(repeat_reward)
        )] = win_rate
    
    print("lr: "+str(lr))
    print("batch_size: "+str(batch_size))
    print("target_update: "+str(target_update))
    print("memory_len: "+str(memory_len))
    print("repeat_reward: "+str(repeat_reward))
    print("win_rate: "+str(win_rate))   
results = sorted(results.items(), key=lambda x: x[1], reverse=True) 
print(results)

filename = 'parameter sampling.json'

with open('loss_plot/'+filename, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)