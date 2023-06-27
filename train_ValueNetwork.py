from functions import get_model_and_config_name, board_normalization, \
                        load_model
import env
from agent_structure import ConnectFourDQNAgent
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

'''
# 학습 데이터 예시 형식
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

SL_agent.eps, RL_agent.eps = 0.05, 0.05  # 되도록 greedy 한 action을 취하기 위해서


# 환경 생성 
VEnv = env.ConnectFourEnv()

# 전역변수 설정
total_count = 10
learn_count = int(total_count * 0.8)
test_count = total_count - learn_count

data = list()
for i in range(0, total_count) :
    # 200마다 진행사항 보고
    if(i % 200 == 0) :
        print("making_data : " + str(i))
    # 환경 초기화
    VEnv.reset()

    state_ = board_normalization(noise=False, env=VEnv, use_conv=SL_agent.use_conv)
    state = torch.from_numpy(state_).float()

    use_rl = False  # RL 에이전트를 사용할 턴인지 여부
    rl_start_turn = random.randint(15, 20)  # RL 에이전트를 사용하기 시작하는 턴 (랜덤으로 지정)
    
    for turn in range(1, 41):
        #print("Turn:", turn)
        # SL 에이전트 사용 (1 ~ rl_start_turn 수)
        # print(VEnv.board)
        if turn < rl_start_turn :
            action = SL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
            if SL_agent.use_minimax: 
                action = action[0]  # action은 이제 0~6 의 정수가 됨 
        # RL 에이전트 사용 (rl_start_turn 수 이후)
        else:
            if use_rl:
                action = RL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
                if RL_agent.use_minimax: 
                    action = action[0]  # action은 이제 0~6 의 정수가 됨 
            else:
                # 완전 랜덤으로 고르면 꽉찬 열에 둘 수 있어서 수정함 
                action = np.random.choice(VEnv.valid_actions)
                # action = random.randint(0, 6)  # 랜덤으로 액션 선택

        #print("Selected Action:", action)

        # 선택한 action으로 환경 진행
        state, reward, done = VEnv.step(action)

        # RL 에이전트 사용 여부 결정
        if turn == rl_start_turn:
            important_state = state
            use_rl = True
            print(turn,important_state)
        
        if (use_rl == False) :
            important_state = state

        if done:
            # 리워드 값 정리
            if (reward == 0.0) :
                reward = 0
            
            # 상대가 이긴 경우 내가 진 것으로 변경
            if (VEnv.player == 2 and reward == 1) :
                reward = -1

            # random 상태의 state값 저장
            important_state = important_state.tolist()
            # 데이터 가공 형식에 맞게 변형, 2를 -1로
            for i in range(len(important_state)) :
                for j in range(0, len(important_state[i])) :
                    if important_state[i][j] == 2 :
                        important_state[i][j] = -1
                    important_state[i][j] = int(important_state[i][j])
            tmp = list()
            tmp_forlist = list()
            # 데이터 가공 형식에 맞게 변형
            tmp_forlist.append(important_state)
            tmp.append(tmp_forlist)
            # reword값을 각각의 배열로 변경
            if (reward == 1) :
                tmp.append([1,0,0])
            elif (reward == 0) : 
                tmp.append([0,1,0])
            elif (reward == -1) : 
                tmp.append([0,0,1])
            # 최종 리스트에 추가
            data.append(tmp)
            break

# 데이터 잘 수집되었는지 확인 
# for i,d in enumerate(data) :
#     # print(i, d)
#     b = np.array(d[0])  # board
#     cnt = np.count_nonzero(b) # 0이 아닌 수의 개수 
#     r = d[1]  # result: win lose draw
#     print(cnt,b,r)

# 학습
for i in range(0, learn_count) :
    # 500마다 진행사항 보고
    if(i % 500 == 0) :
        print("training : " + str(i))
    inputs = torch.FloatTensor(data[i][0])
    targets = torch.FloatTensor(data[i][1]).unsqueeze(0)  

    # 경사 초기화
    optimizer.zero_grad()

    # 순전파 + 역전파 + 최적화
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# 모델 저장
torch.save(model.state_dict(), "ValueNetwork.pth")
print("saved model")

# 테스트
# 예측과 결과가 맞아떨어진 카운트
count = 0
for i in range(learn_count, len(data)) :
    # 200마다 진행사항 보고
    if(i % 200 == 0) :
        print("test : " + str(i))
    # 입력형태에 맞게 변형
    inputs = torch.FloatTensor(data[i][0])
    outputs = model(inputs)

    # 시그모이드를 적용하여 확률을 횓극
    probabilities = torch.sigmoid(outputs)  
    _, predicted = torch.max(probabilities.data, 1)
    Predicted_label = predicted.item()
    # 라벨 기존 데이터 형식과 맞게 변환
    if data[i][1] == [1,0,0] :
        real_label = 0
    elif data[i][1] == [0,1,0] :
        real_label = 1
    elif data[i][1] == [0,0,1] :
        real_label = 2
    # 일치하는 경우 count 추가
    if (Predicted_label == real_label) :
        count += 1
# 최종 정확도 출력
print("percent : " + str(count/40000*100))

# percent : 65.5  (2만개) (학습 1.6만, 테스트 0.4만)
# percent : 81.86 (10만개 (학습 8만, 테스트 2만))
# percent : 83.765 (20만개 (학습 16만, 테스트 4만)) "ValueNetwork.pth"
