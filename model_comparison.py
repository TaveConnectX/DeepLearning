import os
import time
import torch
import env
from functions import compare_model
# simulate 면 1초마다 보여줌, 선턴 후턴 다르게 한번씩 
is_simulation = False

# agent class 할당, 여기도 모델 파일에 따라 자동화할 수 있는 방법이 없을까?
agents = {}
agents[1] = (
    env.MinimaxDQNAgent(eps=0, model_num=6)
)
agents[2] = (
    env.MinimaxDQNAgent(eps=0.3, model_num=6)
)
# models for comparison folder 에서 모델 이름 가져오기
# 폴더 안에는 model 파일만 두개가 존재해야함
# 바꿔야하는 경로를 지정합니다.
folder_path = "model/model_for_comparison/"
file_names = os.listdir(folder_path)
for i in range(len(file_names)):
    file_name = file_names[i]
    if not '.pt' in file_name: 
        print("it is impossible")
        exit()
    else:

        agents[i+1].policy_net.load_state_dict(torch.load(folder_path+file_name))
        agents[i+1].update_target_net()
        print("agent {} model name is {}".format(i+1, file_name))

if is_simulation:
    for first_player in [1,2]:
        CFenv = env.ConnectFourEnv(first_player=first_player)
        while CFenv.win == 0:
            time.sleep(1)
            turn = CFenv.player
            state_ = env.board_normalization(False,CFenv, agents[turn].policy_net.model_type)
            state = torch.from_numpy(state_).float()
            action = agents[turn].select_action(state, CFenv, player=turn)
            if isinstance(action, tuple): action = action[0]
            CFenv.step(action)
            CFenv.print_board(clear_board=False)

else:
    record = compare_model(agents[1], agents[2], n_battle=100)
    print("win rate of agent 1: {}%".format(record[0]/sum(record)*100))






