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
import pandas as pd

epi = 1000  # 한 번 학습에 사용할 episode 수 
CFenv = env.ConnectFourEnv()  # connect4 환경 생성
model_num = 6  # 사용할 내 모델 넘버 
opAgent = env.HeuristicAgent()  # 상대 agent
optimization_trial = 100  # sampling 시도 횟수


def dict2json(data, filename='parameter sampling.json'):

    with open('loss_plot/'+filename, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



num = 1
while True:
    folder_path = "loss_plot/experiment_{}".format(num)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path+" 에 폴더를 만들었습니다.")
        break
    else: num += 1



# 하이퍼파라미터 무작위 탐색======================================
results = {}
for i in range(optimization_trial):
    
    CFenv.reset()
    # 탐색한 하이퍼파라미터의 범위 지정===============
    lr = 10 ** np.random.uniform(-5, -2)
    batch_size = int(2 ** np.random.uniform(4, 10))
    target_update = int(10 ** np.random.uniform(0, 4))
    memory_len = int(2 ** np.random.uniform(11, 15))
    repeat_reward = int(10 ** np.random.uniform(0,1))
    
                
    # ================================================
    Qagent = env.MinimaxDQNAgent(lr=lr, batch_size=batch_size, target_update=target_update, memory_len=memory_len, repeat_reward=repeat_reward,model_num=model_num)

    Qagent.train(epi=epi, env=CFenv, op_model=opAgent)

    plt.clf()
    plt.plot(Qagent.losses)
    plt.savefig('loss_plot/experiment_{}/train_PS_loss_{}.png'.format(num,i))
    
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

with open('loss_plot/experiment_{}/'.format(num)+filename, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


# json2excel.py 파일 코드를 따로 실행시키는 번거로움을 막기 위해 이식 

# Read the json file
with open('loss_plot/experiment_{}/'.format(num)+filename) as f:
    json_data = json.load(f)

# Extract the order, lr, batch_size, target_update, memory_len, and repeat_reward values to a list


data_list = []
for block in json_data:
    data = []
    for ele in block[0]:
        try:
            val = int(ele.split(':')[1])
        except:
            val = float(ele.split(':')[1])
        data.append(val)
    data.append(float(block[1]))

    data_list.append(data)

# Create a Pandas DataFrame from the list
df = pd.DataFrame(data_list, columns=['order', 'lr', 'batch_size', 'target_update', 'memory_len', 'repeat_reward', 'win_rate'])


# Write the DataFrame to an excel file
df.to_excel('loss_plot/experiment_{}/summary.xlsx'.format(num))

print(df)