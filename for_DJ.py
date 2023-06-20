
from functions import get_model_and_config_name, board_normalization, \
                        load_model
import env
from agent_structure import ConnectFourDQNAgent

import torch


# Value Network를 구현하기 위한 모델들은 model/models_for_V_net에 저장되어 있음
# SL model: model/models_for/V_net/model_SL
# RL model: model/models_for/V_net/model_RL

# 각 폴더엔 모델 파일 '.pth' 과 모델의 정보를 담은 '.json' 파일로 이루어져 있음 


# 1. Supervised Learning Model 불러오기
# models_for_V_net 에서 모델과 config 를 불러온다.
folder_path = 'model/models_for_V_net'
SL_model_name, SL_model_config = get_model_and_config_name(folder_path+'/model_SL')
# print(SL_model_name,SL_model_config)
# 불러온 config 파일로 모델 껍데기를 만듦
SL_agent = ConnectFourDQNAgent(config_file_name=folder_path+'/model_SL/'+SL_model_config)
# 불러온 모델 파일로 모델 업로드
load_model(SL_agent.policy_net, filename=folder_path+'/model_SL/'+SL_model_name)

# 2. Reinforcement Learning Model 불러오기
RL_model_name, RL_model_config = get_model_and_config_name(folder_path+'/model_RL')
# print(RL_model_name,RL_model_config)
RL_agent = ConnectFourDQNAgent(config_file_name=folder_path+'/model_RL/'+RL_model_config)
load_model(RL_agent.policy_net, filename=folder_path+'/model_RL/'+RL_model_name)

SL_agent.eps, RL_agent.eps = 0.0, 0.0  # greedy 한 action을 취하기 위해서






# 환경 생성 
VEnv = env.ConnectFourEnv()

# 쓸만한 기능들

# 환경 초기화
VEnv.reset()
# 가능한 행동들
print(VEnv.valid_actions)
# 현재 플레이어
print(VEnv.player)
# 해당 agent가 convolution layer를 쓰는지 확인
print(SL_agent.use_conv)

# 보드를 가지고 오기
print(VEnv.board)

# 보드 정규화, noise의 여부, 환경, 모델이 conv를 쓰는지 여부를 넣어줌 
# noise를 true로 써서 실험해보는 것도 좋을 듯?
state_ = board_normalization(noise=False, env=VEnv, use_conv=SL_agent.use_conv)
state = torch.from_numpy(state_).float()
print(state)

# agent의 action 선택, 인자엔 state와 가능한 액션, 현재 플레이어를 넣어줌 
action = SL_agent.select_action(state=state, env=VEnv, player= VEnv.player)

# minimax-DQN을 사용할 경우 (내 액션, 상대 예상 액션) 으로 리턴하기 때문에 앞에 하나만 필요함
if SL_agent.use_minimax: 
    op_action_prediction = action[1]  # 상대 예상 액션은 이 파일에서 필요 없을 것으로 보임(아마?)
    action = action[0]  # action은 이제 0~6 의 정수가 됨 
print(action)

# 선택한 action으로 그 환경에서 진행했을 때 다음 보드판, 보상, 끝난 여부 출력
# 환경은 reset 하지 않는한 계속 유지되므로 여기선 observation을 따로 저장해둘 필요 없을 듯? 
observation, reward, done = VEnv.step(action)
print(observation, reward, done, VEnv.player)  # player가 바뀐것을 알려주기 위해 출력해봄 

# 더 필요한 함수나 기능, 또는 궁금한 점이 있다면, 
# agent_structure.py의 ConnectFourDQNAgent class의 train() 이나
# env.py 의 ConnectFourEnv class 를 참고해도 됨
# 애매하거나 모르는 부분이 있으면 뭐든 물어보십쇼 






