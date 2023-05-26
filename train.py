import env
import copy
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os

epi = 5000
# 상대를 agent의 policy로 동기화 시키는건 편향이 세지므로 일단 제외
# op_update = 100
CFenv = env.ConnectFourEnv()  # connext4 환경 생성
Qagent = env.ConnectFourDQNAgent(model_num=1)  #학습시킬 agent
# Qagent2 = env.ConnectFourDQNAgent(eps=1)  # it means Qagent2 has random policy
Qagent2 = env.HeuristicAgent()  # 상대 agent

# DQNAgent class 에 내장
# losses = []  # loss 값 plot을 위한 list

# 구조 변경에 따른 삭제 예정
# noise = False  # board를 normalize 할 때 noise 추가 여부.
# flatten = False  # cnn일 땐 False, linear일 땐 true

# train() 을 env.py에 추가함에 따라 필요 없어짐 
# target_update = 500  # target net을 update 하는 주기(단위: episode)

# env.py 로 이동
# # env의 board를 normalize 해주는 함수 
# # 2를 -1로 바꿔서 board를 -1~1로 바꿔줌
# def board_normalization(noise,env=CFenv, flatten=True):
#     # cnn을 사용하지 않는다면, 2차원 board를 1차원으로 바꿔줘야됨 
#     if flatten:
#         arr = copy.deepcopy(env.board.flatten())
#     else: arr = copy.deepcopy(env.board)


#     """Replace all occurrences of 2 with -1 in a numpy array"""
#     arr[arr == 2] = -1
    
#     # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
#     if env.player == 2: arr = -1 * arr

#     if noise:
#         arr += np.random.randn(*arr.shape)/100.0

#     return arr

# 모델을 pth 파일로 저장
def save_model(model, filename='DQNmodel'):
    model_path = 'model/'+filename+'.pth'
    if os.path.isfile(model_path):
        overwrite = input('Overwrite existing model? (Y/n): ')
        if overwrite == 'n':
            new_name = input('Enter name of new model:')
            model_path = 'model/'+new_name+'.pth'
    
    torch.save(model.state_dict(), 'model/'+filename+'.pth')

# 모델 load. 매개변수만 load 하는게 overload가 적다고 하여 이 방법을 선택하였음 
def load_model(model, filename='DQNmodel'):
    model.load_state_dict(torch.load('model/'+filename+'.pth'))


# env.py 로 이동 
# # 두 모델의 승률을 비교하는 함수
# # n_battle 만큼 서로의 policy로 대결하여 
# # [model1's win, model2's win, draw] 리스트를 리턴 
# def compare_model(model1, model2, n_battle=10):
#     # epsilon을 복원하지 않으면, 학습 내내 고정됨 
#     eps1, eps2 = model1.eps, model2.eps
#     # 현재 model2를 random policy로 둘 예정이므로, eps=1을 사용
#     model1.eps, model2.eps = 0, 1.  # no exploration
#     models = [model1, model2]
#     records = [0,0,0]  # model1 win, model2 win, draw
#     comp_env = env.ConnectFourEnv()

#     for round in range(n_battle):
#         comp_env.reset()

#         while not comp_env.done:
#             # 성능 평가이므로, noise를 주지 않음 
#             state_ = board_normalization(noise=False,env=comp_env, flatten=flatten)
#             state = torch.from_numpy(state_).float()
#             player = comp_env.player
#             action = models[player-1].select_action(state, valid_actions=comp_env.valid_actions, player=player)
#             comp_env.step(action)
        
#         if comp_env.win == 1: records[0] += 1
#         elif comp_env.win == 2: records[1] += 1
#         else: records[2] += 1

#     model1.eps, model2.eps = eps1, eps2  # restore exploration

#     return records


        
Qagent.train(epi=epi, env=CFenv, op_model=Qagent2)


plt.plot(Qagent.losses)
plt.show()


record = env.compare_model(Qagent, Qagent2, n_battle=100)
print(record)
if record[0] >= record[1]:
    print("Q1의 승률이 {}, Q1을 선택하겠습니다".format(record[0]/sum(record)))
else:
    print("Q2의 승률이 {}, Q2를 선택하겠습니다".format(record[1]/sum(record)))
    Qagent = Qagent2

# for testing
mode = input("put 1 for test:\n")
if mode == '1':
    Qagent.eps = 0  # no exploration
    CFenv.reset()
    print("let's test model")
    CFenv.print_board()

    while CFenv.win==0:
        if CFenv.player==1:
            col = input("어디에 둘 지 고르세요[0~{}]:\n".format(CFenv.n_col-1))
            if col.isdecimal(): col = int(col)
            else:
                print("잘못된 입력입니다. 다시 입력해주세요.") 
                continue

            if col>=CFenv.n_col or col<0:
                print("잘못된 숫자입니다. 다시 골라주세요")
                continue
            elif CFenv.board[0,col] != 0:
                print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
                continue

            CFenv.step_human(col)
        else:
            time.sleep(1)
            state_ = env.board_normalization(False,CFenv, Qagent.policy_net.model_type)
            state = torch.from_numpy(state_).float()
            action = Qagent.select_action(state, valid_actions=CFenv.valid_actions, player=CFenv.player)
            CFenv.step(action)
            CFenv.print_board()


            
    if CFenv.win==3:
        print("draw!")

    else: print("player {} win!".format(int(CFenv.win)))


save = input("save model? (Y/n): ")
if save != 'n':
    save_model(Qagent.policy_net)

