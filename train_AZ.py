import env
from agent_structure import AlphaZero, AlphaZeroParallel
import copy
import json
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os
from models import AlphaZeroResNet
from functions import get_current_time

num_blocks, num_hidden = 5, 128
args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 5000,
    'num_parallel_games': 1000,
    'num_epochs': 2,
    'batch_size': 1024,
    'temperature': 1,
    'step_makes_temperature_0':20,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1,
    'train_time':get_current_time(),
    'num_blocks':num_hidden,
    'num_hidden':num_blocks
}
# args = {
#     'C': 4,
#     'num_searches': 100,
#     'num_iterations': 8,
#     'num_selfPlay_iterations': 800,
#     'num_parallel_games': 100,
#     'num_epochs': 2,
#     'batch_size': 64,
#     'temperature': 1,
#     'step_makes_temperature_0':9,
#     'dirichlet_epsilon': 0.5,
#     'dirichlet_alpha': 1,
#     'train_time':get_current_time(),
#     'num_blocks':num_hidden,
#     'num_hidden':num_blocks
# }
# args = {
#     'C': 4,
#     'num_searches': 100,
#     'num_iterations': 5,
#     'num_selfPlay_iterations': 100,
#     'num_parallel_games': 10,
#     'num_epochs': 2,
#     'batch_size': 64,
#     'temperature': 1,
#     'dirichlet_epsilon': 0.5,
#     'dirichlet_alpha': 1,
#     'train_time':get_current_time(),
#     'num_blocks':num_hidden,
#     'num_hidden':num_blocks
# }
def seed_everything(seed: int = 42):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)
        # 이건 학습 속도가 줄어든다고 함 
        torch.backends.cudnn.deterministic = False  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore

seed_everything()

game = env.ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlphaZeroResNet(
    num_blocks=num_blocks, \
    num_hidden=num_hidden
).to(device)

# model.load_state_dict(torch.load('model/alphazero/model_8/model_8_iter_7.pth'))

optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.OneCycleLR( \
            optimizer, 
            max_lr=0.2,
            steps_per_epoch=args['batch_size'], 
            epochs=args['num_epochs'],
            anneal_strategy='linear'
        )
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = None
# defalut parameter in code
# with 9 resnet block and 128 batch
# args = {
#     'C': 2,
#     'num_searches': 600,
#     'num_iterations': 8,
#     'num_selfPlay_iterations': 500,
#     'num_parallel_games': 100,
#     'num_epochs': 4,
#     'batch_size': 128,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }

# alphaZero = AlphaZero(model, optimizer, game, args)
# alphaZero.learn()

num = 1
while True:
    folder_path = "model/alphazero/model_{}".format(num)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path+" 에 폴더를 만들었습니다.")
        break
    else: num += 1
args['model_num'] = num

alphaZero = AlphaZeroParallel(model, optimizer,scheduler, game, args)
alphaZero.learn()





# with open('model/alphazero/model_{}/model_config_{}.json'.format(num,num), 'w') as f:
#     json.dump(args, f, indent=4, ensure_ascii=False)
# torch.save(alphaZero.model.state_dict(), "model/alphazero/model_{}/model_{}_iter_{}.pth".format(num,num,iter))



