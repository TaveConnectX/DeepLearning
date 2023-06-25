from functions import get_model_and_config_name, board_normalization, \
                        load_model
import env
from agent_structure import ConnectFourDQNAgent
import random
import torch
import torch.nn as nn
import torch.optim as optim

'''
# 학습 데이터
data = [
    ([[[0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1]]], [1,0,0]), #win

    ([[[1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0]]], [0,1,0]), #draw

    ([[[1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0]]], [0,0,1]) #lose
]
'''

# 신경망 모델 정의
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(42, 3)  # 입력 크기: 7, 출력 크기: 클래스 수

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 2차원 배열을 1차원으로 평탄화
        x = self.fc(x)
        return x

# 모델 인스턴스 생성
model = Classifier()

# 손실 함수 및 최적화 기법 정의
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

'''
# 예측
test_data = [
    [[[0, 1, 0, 1, 1, 0, 1],
      [1, 0, 1, 0, 1, 0, 1],
      [0, 1, 0, 1, 1, 0, 1],
      [1, 0, 1, 0, 1, 0, 1],
      [0, 1, 0, 1, 1, 0, 1],
      [1, 0, 1, 0, 1, 0, 1]]],

    [[[1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0]]]
]

for inputs in test_data:
    inputs = torch.FloatTensor(inputs)
    outputs = model(inputs)
    probabilities = torch.sigmoid(outputs)  # 시그모이드를 적용하여 확률을 얻습니다.
    _, predicted = torch.max(probabilities.data, 1)
    print("Predicted Label:", predicted.item())
'''

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
data = list()
for i in range(0, 2000) :
    if(i % 100 == 0) :
        print("making_data : " + str(i))
    VEnv.reset()

    state_ = board_normalization(noise=False, env=VEnv, use_conv=SL_agent.use_conv)
    state = torch.from_numpy(state_).float()

    use_rl = False  # RL 에이전트를 사용할 턴인지 여부
    rl_start_turn = random.randint(15, 20)  # RL 에이전트를 사용하기 시작하는 턴

    for turn in range(1, 41):
        #print("Turn:", turn)
        # SL 에이전트 사용 (1 ~ rl_start_turn 수)
        if turn < rl_start_turn :
            action = SL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
            if SL_agent.use_minimax: 
                op_action_prediction = action[1]  # 상대 예상 액션은 이 파일에서 필요 없을 것으로 보임(아마?)
                action = action[0]  # action은 이제 0~6 의 정수가 됨 
        # RL 에이전트 사용 (rl_start_turn 수 이후)
        else:
            if use_rl:
                action = RL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
                if RL_agent.use_minimax: 
                    op_action_prediction = action[1]  # 상대 예상 액션은 이 파일에서 필요 없을 것으로 보임(아마?)
                    action = action[0]  # action은 이제 0~6 의 정수가 됨 
            else:
                action = random.randint(0, 6)  # 랜덤으로 액션 선택

        #print("Selected Action:", action)

        # 선택한 action으로 환경 진행
        state, reward, done = VEnv.step(action)

        # RL 에이전트 사용 여부 결정
        if turn == rl_start_turn:
            important_state = state
            use_rl = True
        
        if (use_rl == False) :
            important_state = state

        if done:
            if (reward == 0.0) :
                reward = 0
            
            if (VEnv.player == 2 and reward == 1) :
                reward = -1

            important_state = important_state.tolist()
            for i in range(len(important_state)) :
                for j in range(0, len(important_state[i])) :
                    if important_state[i][j] == 2 :
                        important_state[i][j] = -1
                    important_state[i][j] = int(important_state[i][j])
            tmp = list()
            tmp_forlist = list()
            tmp_forlist.append(important_state)
            tmp.append(tmp_forlist)
            if (reward == 1) :
                tmp.append([1,0,0])
            elif (reward == 0) : 
                tmp.append([0,1,0])
            elif (reward == -1) : 
                tmp.append([0,0,1])
            #print(data)
            data.append(tmp)
            break

# 학습
for i in range(0, 1600) :
    if(i % 100 == 0) :
        print("training : " + str(i))
    inputs = torch.FloatTensor(data[i][0])
    targets = torch.FloatTensor(data[i][1]).unsqueeze(0)  # 타겟에 차원을 추가합니다.

    # 경사 초기화
    optimizer.zero_grad()

    # 순전파 + 역전파 + 최적화
    #print(inputs)
    #print(targets)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# 테스트
count = 0
for i in range(1600, len(data)) :
    if(i % 100 == 0) :
        print("test : " + str(i))
    inputs = torch.FloatTensor(data[i][0])
    outputs = model(inputs)
    probabilities = torch.sigmoid(outputs)  # 시그모이드를 적용하여 확률을 얻습니다.
    _, predicted = torch.max(probabilities.data, 1)
    Predicted_label = predicted.item()
    #print("Predicted Label:", Predicted_label)
    if data[i][1] == [1,0,0] :
        real_label = 1
    elif data[i][1] == [0,1,0] :
        real_label = 0
    elif data[i][1] == [0,0,1] :
        real_label = 2
    #print("real Label:", real_label)
    if (Predicted_label == real_label) :
        count += 1
print("percent : " + str(count/400*100))

    

'''
# 학습 데이터
data = [
    ([[[0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 1]]], [1,0,0]), #win

    ([[[1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0],
       [1, 0, 1, 1, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0]]], [0,1,0]), #draw

    ([[[1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0]]], [0,0,1]) #lose
]
print(type(data[0]))
'''