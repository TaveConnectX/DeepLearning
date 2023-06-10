import os
import torch
import env
import json
import matplotlib.pyplot as plt
import time

# 모델을 pth 파일로 저장
def save_model(model, filename='selfplayModel'):
    global num
    model_path = 'model/model_{}/'.format(num)+filename+'{}_'.format(num)+model.model_name+'.pth'
    if os.path.isfile(model_path):
        overwrite = input('Overwrite existing model? (Y/n): ')
        if overwrite == 'n':
            new_name = input('Enter name of new model:')
            model_path = 'model/model_{}/'.format(num)+new_name+'_'+model.model_name+'.pth'
    
    
    torch.save(model.state_dict(), model_path)


CFenv = env.ConnectFourEnv()
epi = 10000
# 일정 step 마다 train 된 agent를 pool에 넣음
add_pool = 300
agent = env.ConnectFourDQNAgent(
    lr=0.004315892712310481,
    batch_size=21,
    target_update=54,
    memory_len=10395,
    repeat_reward=1,
    model_num=2
)
# pretrain 된 모델 불러오기
agent.policy_net.load_state_dict(torch.load('model/model_for_selfplay/DQNmodel2_CNN-v1'+'.pth'))
agent.update_target_net()
ra = env.ConnectFourRandomAgent()
pool = [ra]  # pool 만들기, 처음엔 randomagent만 들어있음


agent.train_selfplay(epi=epi, env=CFenv, pool=pool, add_pool=add_pool)




op_agent = env.HeuristicAgent()
record = env.compare_model(agent, ha, n_battle=100)
print(record)
print("win rate of agent: {}%".format(record[0]))



model_config = {
    'model_type': agent.policy_net.model_name,
    'op_model_type': op_agent.policy_net.model_name,
    'epi': epi,
    'gamma': agent.gamma,
    'learning rate': agent.lr,
    'batch_size': agent.batch_size,
    'target_update': agent.target_update,
    'memory_len': agent.memory_len,
    'repeat_reward': agent.repeat_reward,
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


plt.plot(agent.losses)
plt.savefig('model/model_{}/loss_{}.png'.format(num,num))
plt.show()

save_model(agent.policy_net)

# for testing
mode = input("put 1 for test:\n")
if mode == '1':
    agent.eps = 0  # no exploration
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
            state_ = env.board_normalization(False,CFenv, agent.policy_net.model_type)
            state = torch.from_numpy(state_).float()
            action = agent.select_action(state, valid_actions=CFenv.valid_actions, player=CFenv.player)
            CFenv.step(action)
            CFenv.print_board()


            
    if CFenv.win==3:
        print("draw!")

    else: print("player {} win!".format(int(CFenv.win)))

