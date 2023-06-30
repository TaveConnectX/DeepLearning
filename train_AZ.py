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
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore

seed_everything()

game = env.ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_blocks, num_hidden = 3, 64
model = AlphaZeroResNet(
    num_blocks=num_blocks, \
    num_hidden=num_hidden
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# args = {
#     'C': 2,
#     'num_searches': 200,
#     'num_iterations': 8,
#     'num_selfPlay_iterations': 300,
#     'num_parallel_games': 60,
#     'num_epochs': 5,
#     'batch_size': 64,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }
args = {
    'C': 2,
    'num_searches': 10,
    'num_iterations': 8,
    'num_selfPlay_iterations': 60,
    'num_parallel_games': 60,
    'num_epochs': 5,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}
# alphaZero = AlphaZero(model, optimizer, game, args)
# alphaZero.learn()

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()


args['train_time'] = get_current_time()
args['num_blocks'] = num_hidden
args['num_hidden'] = num_blocks

num = 1
while True:
    folder_path = "model/alphazero/model_{}".format(num)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path+" 에 폴더를 만들었습니다.")
        break
    else: num += 1

with open('model/alphazero/model_{}/model_config_{}.json'.format(num,num), 'w') as f:
    json.dump(args, f, indent=4, ensure_ascii=False)
torch.save(alphaZero.model.state_dict(), "model/alphazero/model_{}/model_{}.pth".format(num))

plt.plot(alphaZero.losses)
plt.savefig('model/alphazero/model_{}/loss_{}.png'.format(num,num))
plt.show()
plt.plot(alphaZero.vlosses)
plt.savefig('model/alphazero/model_{}/vloss_{}.png'.format(num,num))
plt.show()
plt.plot(alphaZero.plosses)
plt.savefig('model/alphazero/model_{}/ploss_{}.png'.format(num,num))
plt.show()

