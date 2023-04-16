import env
import copy
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt


epi = 10000
op_update = 1000
CFenv = env.ConnectFourEnv()
Qagent = env.ConnectFourDQNAgent()
Qagent2 = env.ConnectFourDQNAgent(eps=1)
losses = []

def board_normalization(arr):
    """Replace all occurrences of 2 with -1 in a numpy array"""
    arr2 = copy.deepcopy(arr)
    arr2[arr2 == 2] = -1
    return arr2


CFenv.reset()


for i in range(epi):
    if i%100==0: 
        CFenv.print_board()
        print("epi",i)

    CFenv.reset()
    state_ = board_normalization(CFenv.board.flatten()) + np.random.randn(1, Qagent.state_size)/100.0
    state = torch.from_numpy(state_).float()
    done = False
    while not done :
        q_value = Qagent.policy_net(state)
        q_value_ = q_value.data.numpy()

        action = Qagent.select_action(state)
        observation, reward, done =  CFenv.step(action)
        op_state_ = board_normalization(observation.flatten()) + np.random.randn(1, Qagent.state_size)
        op_state = torch.from_numpy(op_state_).float()
        if done:
            Qagent.append_memory(state,action, reward, op_state, done)
            break

        op_action = Qagent2.select_action(op_state)
        op_observation, op_reward, op_done = CFenv.step(op_action)
        
        next_state_ = board_normalization(op_observation.flatten()) + np.random.randn(1, Qagent.state_size)
        next_state = torch.from_numpy(next_state_).float()
        if op_done:
            Qagent.append_memory(state,action, -op_reward, next_state, op_done)
            break

        exp = (state, action, reward, next_state, done)
        Qagent.append_memory(*exp)
        state = next_state
        

        Qagent.replay()

        if done: break

    if Qagent.eps > 0.1: Qagent.eps -= (1/epi)
    if i%op_update==0: 
        Qagent2.policy_net.load_state_dict(Qagent.policy_net.state_dict())
        Qagent2.eps = 0



plt.plot(Qagent.losses)
plt.show()

# for testing
mode = input("put 1 for test:\n")
if mode == '1':
    Qagent.eps = 0
    CFenv.reset()
    print("let's test model")
    CFenv.print_board()

    while CFenv.win==0:
        if CFenv.player==1:
            col = int(input("어디에 둘 지 고르세요[0~{}]:\n".format(CFenv.n_col-1)))
            if col>=CFenv.n_col or col<0:
                print("잘못된 숫자입니다. 다시 골라주세요")
                continue
            elif CFenv.board[0,col] != 0:
                print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
                continue

            CFenv.step_human(col)
        else:
            time.sleep(1)
            state_ = board_normalization(CFenv.board.flatten())
            state = torch.from_numpy(state_).float()
            action = Qagent.select_action(state)
            CFenv.step(action)
            CFenv.print_board()


            
    if CFenv.win==3:
        print("draw!")

    else: print("player {} win!".format(int(CFenv.win)))


