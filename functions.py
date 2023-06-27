import os 
import time 
import json
import datetime
import torch
import copy 
import numpy as np 
from operator import add, neg
# 모델을 pth 파일로 저장
def save_model(model, filename='Model', folder_num=None):
    if folder_num is None:
        num = 1  # folder의 이름에 쓰일 숫자
        while True:
            folder_path = "model/model_{}".format(num)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(folder_path+" 에 폴더를 만들었습니다.")
                break
            else: num += 1
    
    else: num=folder_num

    model_path = 'model/model_{}/'.format(num)+filename+'{}_'.format(num)+model.model_name+'.pth'
    if os.path.isfile(model_path):
        overwrite = input('Overwrite existing model? (Y/n): ')
        if overwrite == 'n':
            new_name = input('Enter name of new model:')
            model_path = 'model/model_{}/'.format(num)+new_name+'_'+model.model_name+'.pth'
    
    
    torch.save(model.state_dict(), model_path)


# 모델 load. 매개변수만 load 하는게 overload가 적다고 하여 이 방법을 선택하였음 
def load_model(model, filename='Model'):
    if not '.pth' in filename or not '.pt' in filename:
        filename += '.pth'
    model.load_state_dict(torch.load(filename))

# console 창을 비우는 함수 
def clear():
    os.system('cls' if os.name=='nt' else 'clear')

# env의 board를 normalize 해주는 함수 
# 2를 -1로 바꿔서 board를 -1~1로 바꿔줌
def board_normalization(noise:bool,env, use_conv:bool):
    # cnn을 사용하지 않는다면, 2차원 board를 1차원으로 바꿔줘야됨 
    if use_conv:
        arr = copy.deepcopy(env.board)
    else: 
        arr = copy.deepcopy(env.board.flatten())


    """Replace all occurrences of 2 with -1 in a numpy array"""
    arr[arr == 2] = -1
    
    # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
    if env.player == 2: arr = -1 * arr

    if noise:
        arr += np.random.randn(*arr.shape)/100.0

    return arr

def get_distinct_actions(env):
    board = env.board
    valid_actions = env.valid_actions
    board = np.array(board).reshape(6,7)

    distinct_actions = []
    for a in valid_actions:
        if board[1,a] != 0:
            distinct_actions.append(a)

    return distinct_actions

# 한 칸만 남았으면 pair 액션이 불가능하므로 체크가 필요 
def is_full_after_my_turn(valid_actions, distinct_actions):
    if len(valid_actions)==1 and len(distinct_actions)==1:
        return True
    else: return False

# def get_minimax_action(q_value,valid_actions, distinct_actions):
#     q_dict = {}
#     for a in valid_actions:
#         q_dict[a] = (None, np.inf)
#         for b in valid_actions:
#             if a in distinct_actions and a==b: continue
#             idx = 7*a + b
#             # print(a,b)
#             # print(q_value[idx])
#             # print(q_dict[a][1])
#             if q_value[idx] <= q_dict[a][1]:
#                 q_dict[a] = (b, q_value[idx])

#     max_key = None
#     max_value = float('-inf')
#     for a, (b, q) in q_dict.items():
#         if q > max_value:
#             max_key = a
#             max_value = q

#     return (max_key, q_dict[max_key][0])



def get_valid_actions(board):
    valid_actions = []
    for i in range(7):
        if board[0,i] == 0:
            valid_actions.append(i)
    return valid_actions

def get_next_board(board, action, player):
    next_board = copy.deepcopy(board)
    if board[0,action] != 0:
        print("invalid action")
        exit()
    for row in range(5,-1,-1):
        if next_board[row,action] == 0:
            next_board[row,action] = player
            return next_board

# # 두 모델의 승률을 비교하는 함수
# # n_battle 만큼 서로의 policy로 대결하여 
# # [model1's win, model2's win, draw] 리스트를 리턴 
# def compare_model(model1, model2, n_battle=10):
#     # epsilon을 복원하지 않으면, 학습 내내 고정됨 
#     eps1 = model1.eps
#     model1.eps = 0  # no exploration
#     players = {1:model1, 2:model2}
#     records = [0,0,0]  # model1 win, model2 win, draw
#     comp_env = ConnectFourEnv()
#     step = 0
#     for round in range(n_battle):
#         comp_env.reset()

#         while not comp_env.done:
#             # 성능 평가이므로, noise를 주지 않음 
#             turn = comp_env.player
#             state_ = board_normalization(noise=False,env=comp_env, use_conv=players[turn].use_conv)
#             state = torch.from_numpy(state_).float()
            
#             action = players[turn].select_action(state, comp_env, player=turn)
#             if isinstance(action, tuple): action = action[0]
#             comp_env.step(action)
#             step += 1
#         if comp_env.win == 1: records[0] += 1
#         elif comp_env.win == 2: records[1] += 1
#         else: records[2] += 1
#         # print(step)
#     model1.eps = eps1  # restore exploration

#     return records


# # model1과 model2의 policy에 따라
# # 어떻게 플레이 하는지를 직접 확인가능
# def simulate_model(model1, model2):
#     eps1 = model1.eps
#     model1.eps = 0  # no exploration
#     players = {1:model1, 2:model2}

#     test_env = ConnectFourEnv()

#     test_env.reset()
#     while not test_env.done:
#         turn = test_env.player
#         state_ = board_normalization(noise=False, env=test_env, use_conv=players[turn].use_conv)
#         state = torch.from_numpy(state_).float()

#         action = players[turn].select_action(state, test_env, player=turn)
#         test_env.step(action)
#         test_env.print_board(clear_board=False)
#         print("{}p put piece on {}".format(turn, action))
#         time.sleep(1)   
#     print("winner is {}p".format(test_env.win))

#     model1.eps = eps1  # restore exploration

def get_current_time():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_model_config(file_name=None):
    if file_name is None:
        file_name = 'config.json'
    with open(file_name, 'r') as f:
        config = json.load(f)
    return config

def get_model_and_config_name(folder_path):
    file_names = os.listdir(folder_path)
    for file in file_names:
        if '.pth' in file or '.pt' in file:
            model_name = file
        elif '.json' in file:
            model_config_name = file
    return model_name, model_config_name


# z: [(idx1, value1), (idx2, value2), ...] 으로 이루어져있어야함 
def softmax_policy(z, temp): 
    if not isinstance(z, torch.Tensor):
        z = torch.Tensor(z)

    # temp=0 이면 greedy한 action 선택
    if temp==0:
        maxidx = z.argmax(dim=0)[1]
        action, value = z[maxidx]
        if True in torch.isnan(z[maxidx]):
            print(z)
            print(action, value)
            print(maxidx)
        return int(action), value
    temp += torch.finfo(z.dtype).tiny
    # z = np.array(z)
    if z.dim() == 1:
        z = torch.tensor([(idx, value) for idx, value in np.ndenumerate(z)])
    if z.dim() == 2:
        idxs = z[:,0]
        values = z[:,1]

    values = np.array(values.cpu())
    # print(z)
    values = values / temp
    # print(values)
    max_v = np.max(values)
    #print(max_v)
    exp_v = np.exp(values-max_v) 
    #print(exp_v)
    sum_exp_v = np.sum(exp_v)
    #print(sum_exp_v)
    y = exp_v / sum_exp_v
    # print(y)
    # print("sum of y:",np.sum(y))
    action = np.random.choice(idxs.cpu(), p=y)
    value = next(pair[1] for pair in z if pair[0] == action)

    return int(action), value

def set_optimizer(optimizer,parameters, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=lr)
    elif optimizer == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr=lr)
    else:
        raise ValueError("optimizer is not defined")
    
def get_nash_prob_and_value(payoff_matrix, iterations=100):
    payoff_matrix = np.array(payoff_matrix)
    '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
    transpose_payoff = np.transpose(payoff_matrix)
    row_cum_payoff = np.zeros(len(payoff_matrix))
    col_cum_payoff = np.zeros(len(transpose_payoff))

    col_count = np.zeros(len(transpose_payoff))
    row_count = np.zeros(len(payoff_matrix))
    active = 0
    # print("payoff_matrix:",payoff_matrix)
    # print("transpose payoff:",transpose_payoff)
    # print("row_cum_payoff:",row_cum_payoff)
    # print("col_cum_payoff:",col_cum_payoff)
    # print("colpos:",col_pos)
    # print("rowpos:",row_pos)
    # print("colcnt:",col_count)
    # print("rowcnt:",row_count)
    # print("active:",active)

    for i in range(iterations):
        row_count[active] += 1 
        col_cum_payoff += payoff_matrix[active]
        # print("col_cum_payoff:",col_cum_payoff)

        active = np.argmin(col_cum_payoff)
        # print("active:",active)

        col_count[active] += 1 

        row_cum_payoff += transpose_payoff[active]
        # print("row_cum_payoff:",row_cum_payoff)
        
        active = np.argmax(row_cum_payoff)
        # print("active:",active)
        
        
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations  
    row_prob = row_count / iterations
    col_prob = col_count / iterations
    
    return row_prob, col_prob, value_of_game