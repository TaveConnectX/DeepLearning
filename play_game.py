import env
import numpy as np

CF = env.ConnectFourEnv()
mode = input("to play with human, type 'human'(else just enter):")
CF.print_board()

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
        CF.step_cpu()


        
if CF.win==3:
    print("draw!")

else: print("player {} win!".format(int(CF.win)))