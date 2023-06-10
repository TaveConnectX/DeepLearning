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

epi = 10
# 상대를 agent의 policy로 동기화 시키는건 편향이 세지므로 일단 제외
# op_update = 100
CFenv = AlphaZeroenv.CFEnvforAlphaZero()  # connext4 환경 생성
agent = env.AlphaZeroAgent(env=CFenv)

agent.train(epochs=100)
# Qagent = env.ConnectFourDQNAgent(
#     lr=0.004315892712310481,
#     batch_size=21,
#     target_update=54,
#     memory_len=10395,
#     repeat_reward=1,
#     model_num=2
# )  #학습시킬 agent
# Qagent2 = env.ConnectFourDQNAgent(eps=1)  # it means Qagent2 has random policy
# Qagent2 = env.ConnectFourRandomAgent()  # 상대 agent


# 모델을 pth 파일로 저장
def save_model(model, filename='DQNmodel'):
    global num
    model_path = 'model/model_{}/'.format(num)+filename+'{}_'.format(num)+model.model_name+'.pth'
    if os.path.isfile(model_path):
        overwrite = input('Overwrite existing model? (Y/n): ')
        if overwrite == 'n':
            new_name = input('Enter name of new model:')
            model_path = 'model/model_{}/'.format(num)+new_name+'_'+model.model_name+'.pth'
    
    
    torch.save(model.state_dict(), model_path)

# 모델 load. 매개변수만 load 하는게 overload가 적다고 하여 이 방법을 선택하였음 
def load_model(model, filename='DQNmodel'):
    model.load_state_dict(torch.load('model/'+filename+'.pth'))




        
Qagent.train(epi=epi, env=CFenv, op_model=Qagent2)




record = env.compare_model(Qagent, Qagent2, n_battle=100)
print(record)
print("win rate of Qagent: {}%".format(record[0]))



model_config = {
    'model_type': Qagent.policy_net.model_name,
    'op_model_type': Qagent2.policy_net.model_name,
    'epi': epi,
    'gamma': Qagent.gamma,
    'learning rate': Qagent.lr,
    'batch_size': Qagent.batch_size,
    'target_update': Qagent.target_update,
    'memory_len': Qagent.memory_len,
    'repeat_reward': Qagent.repeat_reward,
    'win_rate': record[0]/sum(record),
}

num = 1
while True:
    folder_path = "model/model_{}".format(num)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path+" 에 폴더를 만들었습니다.")
        break
    else: num += 1

with open('model/model_{}/model_config_{}.json'.format(num,num), 'w') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)


plt.plot(Qagent.losses)
plt.savefig('model/model_{}/loss_{}.png'.format(num,num))
plt.show()

save_model(Qagent.policy_net)

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


# 폴더별로 관리하게 바꿈으로써 밑의 코드는 필요 없어짐 
# save = input("save model? (Y/n): ")
# if save != 'n':
#     save_model(Qagent.policy_net)

