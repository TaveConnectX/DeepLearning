import torch
from agent_structure import ConnectFourDQNAgent, evaluate_model
from functions import load_model, get_model_and_config_name, \
                    get_model_config
# 폴더에서 모델 불러오기
folder_path = 'model/model_for_evaluate'
model_name, model_config = get_model_and_config_name(folder_path)

# 불러온 config 파일로 모델 껍데기를 만듦
prev_model_config = get_model_config(folder_path+'/'+model_config)
kwargs={
    'use_conv':prev_model_config['use_conv'], \
    'use_minimax':prev_model_config['use_minimax'], \
    'use_resnet':prev_model_config['use_resnet'],
    'next_state_is_op_state':prev_model_config['next_state_is_op_state'],
    'use_nash':prev_model_config['use_nash']
}
agent = ConnectFourDQNAgent(config_file_name=None, **kwargs)
#print(agent.use_resnet)
# 불러온 모델 파일로 모델 업로드
load_model(agent.policy_net, filename=folder_path+'/'+model_name)
agent.eps = 0


state = [
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0,-1,-1, 0, 0, 0],
    [ 0, 0, 1, 1, 0, 0, 0],
    [ 0, 0,-1,-1, 0, 0, 0],
    [ 1, 0, 1, 1, 0, 0, 0]
]
state = torch.tensor(state).float().unsqueeze(0).unsqueeze(0).to('cuda')
print(agent.policy_net(state).reshape(7,7))


# 수능처럼 점수를 계산하자
# random model: 2점
# heuristic model: 3점
# minimax model: 4점

# 2점 3
# 3점 14
# 4점 13

# compare model로 점수 계산 
num_tests = 10
n_battles = [3,14,13]
records = [[],[],[]]
scores = []
for i in range(num_tests):
    print(i, scores)
    evaluate_model(agent, records, n_battles=n_battles)
    twos = records[0][-1]
    threes = records[1][-1]
    fours = records[2][-1]

    scores.append(twos*2+threes*3+fours*4)

print(records)
print(max(scores))
print(min(scores))
print(sum(scores)/len(scores))

