import env
import AlphaZeroenv
import copy
import json
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os



CFenv = AlphaZeroenv.CFEnvforAlphaZero()  # connext4 환경 생성
agent = env.AlphaZeroAgent(env=CFenv, num_simulations=343, num_iterations=10, num_episodes=10, batch_size=16)

agent.train(epochs=2)

# 모델을 pth 파일로 저장
def save_model(model, filename='ResNetmodel'):
    num = 1
    while True:
        folder_path = "model/model_{}".format(num)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path+" 에 폴더를 만들었습니다.")
            break
        else: num += 1
        
    model_path = 'model/model_{}/'.format(num)+filename+'{}_'.format(num)+model.model_name+'.pth'
    if os.path.isfile(model_path):
        overwrite = input('Overwrite existing model? (Y/n): ')
        if overwrite == 'n':
            new_name = input('Enter name of new model:')
            model_path = 'model/model_{}/'.format(num)+new_name+'_'+model.model_name+'.pth'
    
    
    torch.save(model.state_dict(), model_path)


save_model(agent.model)
