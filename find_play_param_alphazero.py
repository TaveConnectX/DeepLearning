from env import ConnectFour, MCTS
from models import AlphaZeroResNet
import numpy as np
import torch
import time
import os
import copy
import random


def battle(k1, k2, mctss, result):

    # player1: mcts1, player -1: mcts2 
    mcts1 = mctss[k1]
    mcts2 = mctss[k2]
    outcome = []
    for start_player in [1,-1]:
        player = start_player
        state = CF.get_initial_state()
        while True:
            if player == -1:
                neutral_state = CF.change_perspective(state, player)
                mcts_probs = mcts2.search(neutral_state)
                print("player -1:",mcts_probs)
                # time.sleep(5)
                action = np.argmax(mcts_probs)
                
            elif player == 1:
                neutral_state = CF.change_perspective(state, player)
                mcts_probs = mcts1.search(neutral_state)
                print("player 1:",mcts_probs)
                # time.sleep(5)
                action = np.argmax(mcts_probs)
                
            state = CF.get_next_state(state, action, player)
            
            value, is_terminal = CF.get_value_and_terminated(state, action)
            
            if is_terminal:
                print(state)
                if value == 1:
                    print(player, "won")
                    if player == -1:
                        outcome.append(-1)
                    else: outcome.append(1)

                else:
                    print("draw")
                    outcome.append(0)
                break
                
            player = CF.get_opponent(player)

    
    if sum(outcome) == 2:
        result[k1] += 3
        result[k2] -= 3
        log = "2:0"
    elif sum(outcome) == -2:
        result[k1] -= 3
        result[k2] += 3
        log = "0:2"
    elif sum(outcome) == 1:
        result[k1] += 2
        result[k2] -= 2
        log = "1:0"
    elif sum(outcome) == -1:
        result[k1]  -= 2
        result[k2] += 2
        log = "0:1"
    elif outcome == [0,0]:
        result[k1] += 1
        result[k2] += 1
        log = "0:0"
    else:
        log = "1:1"

    print("{} vs {}: ".format(k1,k2)+log)
    print(result)
    return result



# 모델을 로드한다

CF = ConnectFour()
folder_path = "model/alphazero/"
print("what the...")

num_battles = 55
nb, hl, model_name = 9,128,'model_9/model_9_iter_0.pth'
args = {
    'C': 2,
    'num_searches': 50,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlphaZeroResNet(nb, hl).to(device)
model.load_state_dict(torch.load(folder_path+model_name, map_location=device))
model.eval()


# 경기결과 
result = {}

# 풀을 만든다
mctss = {}
Cs = [i/10 for i in range(15,25)]
for c in Cs:
    new_args = copy.deepcopy(args)
    new_args['C'] = c
    mctss[c] = MCTS(CF, new_args, model)
    result[c] = 0

# 랜덤으로 뽑아서 대결을 한다
for i in range(num_battles):
    
    keys = list(mctss.keys())
    k1, k2 = random.sample(keys,2)
    print(i, k1, k2)

    # k1, k2가 배틀을 함 
    result = battle(k1,k2,mctss,result)



# 점수 순으로 나열한다.
print(result)
sorted_result = sorted(result.items(), key=lambda x: -x[1])

print(sorted_result)
