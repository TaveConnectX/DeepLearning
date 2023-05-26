import env
import numpy as np
import torch
import time

CF = env.ConnectFourEnv()
mode = input("to play with human, type 'human'(else just enter):")
CF.print_board()


# 나중에 RandomAgent가 추가된다면 ConnectFourDQNAgent(), RandomAgent(), HeuristicAgent() 등등 선택 가능
agent = env.HeuristicAgent()

while CF.win==0:
    if CF.player==1:
        col = int(input("어디에 둘 지 고르세요[0~{}]:".format(CF.n_col-1)))
        if col>=CF.n_col or col<0:
            print("잘못된 숫자입니다. 다시 골라주세요")
            continue
        elif CF.board[0,col] != 0:
            print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
            continue

        CF.step_human(col)
    else:
        # random agent가 추가되면 step_cpu() 를 사용하지 않아도 될듯
        # 현재 cpu는 random
        #CF.step_cpu()
        time.sleep(1)
        state_ = env.board_normalization(False,CF, agent.policy_net.model_type)
        state = torch.from_numpy(state_).float()
        action = agent.select_action(state, valid_actions=CF.valid_actions, player=CF.player)
        CF.step(action)
        CF.print_board()


        
if CF.win==3:
    print("draw!")

else: print("player {} win!".format(int(CF.win)))