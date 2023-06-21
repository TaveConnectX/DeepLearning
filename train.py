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
from functions import get_model_config, save_model, \
                        get_current_time, get_model_and_config_name, load_model
from agent_structure import ConnectFourDQNAgent, HeuristicAgent, set_op_agent


config = get_model_config()

# 상대를 agent의 policy로 동기화 시키는건 편향이 세지므로 일단 제외
# op_update = 100
CFenv = env.ConnectFourEnv()  # connext4 환경 생성
# agent with
# agent = env.AlphaZeroAgent(env=CFenv)

Qagent = ConnectFourDQNAgent(
    state_size=CFenv.n_col * CFenv.n_row,
    action_size=CFenv.n_col
)
if config['selfplay']:
    folder_path = 'model/model_for_selfplay'
    model_name, model_config = get_model_and_config_name(folder_path)
    # print(SL_model_name,SL_model_config)
    # 불러온 config 파일로 모델 껍데기를 만듦
    prev_model_config = get_model_config(folder_path+'/'+model_config)
    Qagent = ConnectFourDQNAgent({
        'use_conv':prev_model_config['use_conv'], \
        'use_minimax':prev_model_config['use_minimax'], \
        'use_resnet':prev_model_config['use_resnet']
    })
    # 불러온 모델 파일로 모델 업로드
    load_model(Qagent.policy_net, filename=folder_path+'/'+model_name)
# Qagent2 = env.ConnectFourDQNAgent(eps=1)  # it means Qagent2 has random policy
# if Qagent is MinimaxDQNAgent and Qagent2 is None,
# Qagent will train with its own.
Qagent2 = set_op_agent(config['op_agent'])  # 상대 agent


# # 모델을 pth 파일로 저장
# def save_model(model, filename='DQNmodel'):
#     global num
#     model_path = 'model/model_{}/'.format(num)+filename+'{}_'.format(num)+model.model_name+'.pth'
#     if os.path.isfile(model_path):
#         overwrite = input('Overwrite existing model? (Y/n): ')
#         if overwrite == 'n':
#             new_name = input('Enter name of new model:')
#             model_path = 'model/model_{}/'.format(num)+new_name+'_'+model.model_name+'.pth'
    
    
#     torch.save(model.state_dict(), model_path)

# # 모델 load. 매개변수만 load 하는게 overload가 적다고 하여 이 방법을 선택하였음 
# def load_model(model, filename='DQNmodel'):
#     model.load_state_dict(torch.load('model/'+filename+'.pth'))


Qagent.train(epi=config['epi'], env=CFenv, op_model=Qagent2)



# 여긴 나중에 evaluation으로 바꿔보자 
if Qagent2 is None: Qagent2 = env.HeuristicAgent()
record = env.compare_model(Qagent, Qagent2, n_battle=100)
print(record)
print("win rate of Qagent: {}%".format(record[0]))



model_config = {
    'model_type': Qagent.policy_net.model_name,
    
    'epi': config['epi'],
    'gamma': Qagent.gamma,
    'learning rate': Qagent.lr,
    'batch_size': Qagent.batch_size,
    'target_update': Qagent.target_update,
    'memory_len': Qagent.memory_len,
    'repeat_reward': Qagent.repeat_reward,
    'win_rate': record[0]/sum(record),
}
config['model_name'] = Qagent.policy_net.model_name
config['win_rate'] = record[0]/sum(record)
if not config['selfplay']:
    config['op_model_type'] = Qagent2.policy_net.model_name
else: config['op_model_type'] = 'HeuristicAgent'
config['train_time'] = get_current_time()


num = 1
while True:
    folder_path = "model/model_{}".format(num)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path+" 에 폴더를 만들었습니다.")
        break
    else: num += 1

with open('model/model_{}/model_config_{}.json'.format(num,num), 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


plt.plot(Qagent.losses)
plt.savefig('model/model_{}/loss_{}.png'.format(num,num))
plt.show()

for i in range(3):
    if i==2: plt.plot(Qagent.record[i]/34*100)
    else: plt.plot(Qagent.record[i]/33*100)
plt.savefig('model/model_{}/win_rate_{}.png'.format(num,num))
plt.show()

save_model(Qagent.policy_net, folder_num=num)

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
            action = Qagent.select_action(state, CFenv, player=CFenv.player)
            if isinstance(action, tuple):
                action = action[0]
            CFenv.step(action)
            CFenv.print_board()


            
    if CFenv.win==3:
        print("draw!")

    else: print("player {} win!".format(int(CFenv.win)))


# 폴더별로 관리하게 바꿈으로써 밑의 코드는 필요 없어짐 
# save = input("save model? (Y/n): ")
# if save != 'n':
#     save_model(Qagent.policy_net)

