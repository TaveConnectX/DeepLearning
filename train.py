import env
import copy
import numpy as np
import torch
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os

epi = 50000
# 상대를 agent의 policy로 동기화 시키는건 편향이 세지므로 일단 제외
# op_update = 100
CFenv = env.ConnectFourEnv()  # connext4 환경 생성
Qagent = env.ConnectFourDQNAgent_CNN()  #학습시킬 agent
# Qagent2 = env.ConnectFourDQNAgent(eps=1)  # it means Qagent2 has random policy
Qagent2 = env.ConnectFourDQNAgent_CNN()  # 상대 agent
losses = []  # loss 값 plot을 위한 list
noise=False  # board를 normalize 할 때 noise 추가 여부.
flatten = False  # cnn일 땐 False, linear일 땐 true
target_update = 500  # target net을 update 하는 주기(단위: episode)

# 2를 -1로 바꿔서 board를 -1~1로 바꿔줌
def board_normalization(noise,env=CFenv, flatten=True):
    # cnn을 사용하지 않는다면, 2차원 board를 1차원으로 바꿔줘야됨 
    if flatten:
        arr = copy.deepcopy(env.board.flatten())
    else: arr = copy.deepcopy(env.board)


    """Replace all occurrences of 2 with -1 in a numpy array"""
    arr[arr == 2] = -1
    
    # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
    if env.player == 2: arr = -1 * arr

    if noise:
        arr += np.random.randn(*arr.shape)/100.0

    return arr

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


# 두 모델의 승률을 비교하는 함수
# n_battle 만큼 서로의 policy로 대결하여 
# [model1's win, model2's win, draw] 리스트를 리턴 
def compare_model(model1, model2, n_battle=10):
    # epsilon을 복원하지 않으면, 학습 내내 고정됨 
    eps1, eps2 = model1.eps, model2.eps
    # 현재 model2를 random policy로 둘 예정이므로, eps=1을 사용
    model1.eps, model2.eps = 0, 1.  # no exploration
    models = [model1, model2]
    records = [0,0,0]  # model1 win, model2 win, draw
    comp_env = env.ConnectFourEnv()

    for round in range(n_battle):
        comp_env.reset()

        while not comp_env.done:
            # 성능 평가이므로, noise를 주지 않음 
            state_ = board_normalization(noise=False,env=comp_env, flatten=flatten)
            state = torch.from_numpy(state_).float()
            player = comp_env.player
            action = models[player-1].select_action(state, valid_actions=comp_env.valid_actions, player=player)
            comp_env.step(action)
        
        if comp_env.win == 1: records[0] += 1
        elif comp_env.win == 2: records[1] += 1
        else: records[2] += 1

    model1.eps, model2.eps = eps1, eps2  # restore exploration

    return records

CFenv.reset()

# 1p는 Qmodels[1], 2p는 Qmodels[2] 로 직관적으로 사용하기 위해 
Qmodels = [0, Qagent, Qagent2]

for i in range(epi):
    # 100번마다 loss, eps 등의 정보 표시
    if i!=0 and i%100==0: 
        CFenv.print_board(clear_board=False)
        print("epi:",i, ", agent's step:",Qagent.steps)
        # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
        record = compare_model(Qagent, Qagent2, n_battle=100)
        print(record)
        print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
        print("loss:",sum(Qagent.losses[-101:-1])/100.)
        print("epsilon:",Qagent.eps)

    # 주기적으로 target net  업데이트 
    if i%target_update==0:
        print("target net update")
        Qagent.update_target_net()
        Qagent2.update_target_net()

    CFenv.reset()
    # if CFenv.player == 2:
    #     first_move = np.random.choice(range(CFenv.n_col))
    #     CFenv.board[CFenv.n_row-1, first_move] = 1
    state_ = board_normalization(noise,CFenv, flatten)
    state = torch.from_numpy(state_).float()  # np array to torch
    done = False
    
    past_state, past_action, past_reward, past_done = state, None, None, done

    while not done :
        # print(i)
        # for param in Qagent.parameters():
        #     print(type(param),  param.size())
        
        # q_value = Qagent.policy_net(state)
        # q_value_ = q_value.data.numpy()
        player = CFenv.player
        op_player = 2//player

        action = Qmodels[player].select_action(state, valid_actions=CFenv.valid_actions, player=player)

        observation, reward, done =  CFenv.step(action)
        op_state_ = board_normalization(noise,CFenv, flatten)
        op_state = torch.from_numpy(op_state_).float()
        
        
        if past_action != None:  # 맨 처음이 아닐 때 
            # 경기가 끝났을 때(중요한 경험)
            if done:
                repeat = 1
                # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                if reward > 0: repeat = 5
                for j in range(repeat):
                    # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                    Qagent.append_memory(state,action, reward, op_state*-1, done)
                    # Qmodels[player].append_memory(state,action, reward, op_state*-1, done)
                    # 내가 이겼으므로 상대는 음의 보상을 받음 
                    #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
                    Qagent.append_memory(past_state, past_action, -reward, op_state, done)

                # print for debugging
                # print("for player")
                # print("state:\n",torch.round(state).int())
                # print("action:",action)
                # print("reward:",reward)
                # print("next_state\n",torch.round(op_state*-1).int())
                # print()
                # print("for opponent")
                # print("state:\n",torch.round(past_state).int())
                # print("action:",past_action)
                # print("reward:",-reward)
                # print("next_state\n",torch.round(op_state).int())
                # print()

            # 경기가 끝나지 않았다면
            else:
                Qagent.append_memory(past_state, past_action, past_reward, op_state, past_done)
                # Qmodels[op_player].append_memory(past_state, past_action, past_reward, op_state, past_done)

                # print for debugging
                # print("for opponent")
                # print("state:\n",torch.round(past_state).int())
                # print("action:",past_action)
                # print("reward:",past_reward)
                # print("next_state\n",torch.round(op_state).int())
                # print()

        
        # op_action = Qmodels[player].select_action(op_state,valid_actions=CFenv.valid_actions, player=player)
        # op_observation, op_reward, op_done = CFenv.step(op_action)
        
        # next_state_ = board_normalization(op_observation.flatten()) + np.random.randn(1, Qagent.state_size)/100.0
        # next_state = torch.from_numpy(next_state_).float()
        # # 2p가 돌을 놓자마자 끝남 
        # if op_done:
        #     Qmodels[player].append_memory(op_state,op_action, op_reward, next_state, op_done)
        #     Qmodels[op_player].append_memory(state,action, -op_reward, next_state, op_done)
        # else:
        #     exp = (state, action, reward, next_state, done)
        #     Qmodels[player].append_memory(*exp)

        # info들 업데이트 해줌 
        past_state = state
        past_action = action
        past_reward = reward
        past_done = done
        state = op_state
        
        # replay buffer 를 이용하여 mini-batch 학습
        Qmodels[player].replay()
        Qmodels[op_player].replay()
        # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
        #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
        #     print("action:",Qagent.memory[-1][1])
        #     print("reward:",Qagent.memory[-1][2])
        #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
            
        # 게임이 끝났다면 나가기 
        if done: break
    # print("eps:",Qagent.eps)
    # epsilon-greedy
    # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
    if Qagent.eps > 0.1: 
        Qagent.eps -= (1/epi)
        # Qagent2.eps -= (1/epi)
        

    # if i>= epi//2 and i%op_update==0: 
    #     records = compare_model(Qagent, Qagent2)

    #     if records[0] >= records[1]:
    #         Qagent2.policy_net.load_state_dict(Qagent.policy_net.state_dict())
            
    #     else: Qagent.policy_net.load_state_dict(Qagent2.policy_net.state_dict())

    #     Qagent2.eps = Qagent.eps
        



plt.plot(Qagent.losses)
plt.show()


record = compare_model(Qagent, Qagent2, n_battle=100)
print(record)
if record[0] >= record[1]:
    print("Q1의 승률이 {}, Q1을 선택하겠습니다".format(record[0]/sum(record)))
else:
    print("Q2의 승률이 {}, Q2를 선택하겠습니다".format(record[1]/sum(record)))
    Qagent = Qagent2


# for testing
mode = input("put 1 for test:\n")
if mode == '1':
    Qagent.eps = 0
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
            state_ = board_normalization(False,CFenv, flatten)
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

