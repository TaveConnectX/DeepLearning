import numpy as np
import os


def clear():
    os.system('cls' if os.name=='nt' else 'clear')

class ConnectFourEnv:
    def __init__(self, n_row=6, n_col=7, mode='cpu'):
        self.n_row = n_row
        self.n_col = n_col
        self.board = np.zeros((n_row, n_col))
        self.mode = mode
        self.player = np.random.choice([1,2])
        self.win = 0

    # col에 조각 떨어뜨리기 
    def drop_piece(self, col):
        # 떨어뜨리려는 곳이 이미 가득 차있을 때
        if not self.board[0][col] == 0:
            print("already full")
            return

        else:
            for row in range(self.n_row-1,-1,-1):
                if self.board[row][col] == 0:
                    self.board[row][col] = self.player
                    break
                else: continue

    def change_player(self):
        self.player = int(2//self.player)

    def print_board(self):
        clear()
        for row in range(self.n_row):
            row_str = "|"
            for col in range(self.n_col):
                if self.board[row][col] == 0:
                    row_str += " "
                elif self.board[row][col] == 1:
                    row_str += "X"
                elif self.board[row][col] == 2:
                    row_str += "O"
                row_str += "|"
            print(row_str)
        print("+" + "-" * (len(self.board[0]) * 2 - 1) + "+")
        print("player {}'s turn!".format(int(self.player)))

    # (x,y) 에서 8방향으로 적절한 좌표 3개를 제공 
    def coor_for_8_direction(self, x, y):
        coors = []
        left, right, up, down = (False,)*4
        # (x,y) 기준 오른쪽
        if y+3<self.n_col:
            right=True
            coors.append(((x,y+1),(x,y+2),(x,y+3)))
        # (x,y) 기준 왼쪽
        if y-3>=0:
            left=True
            coors.append(((x,y-1),(x,y-2),(x,y-3)))
        # (x,y) 기준 위 
        if x-3>=0:
            up=True
            coors.append(((x-1,y),(x-2,y),(x-3,y)))
        # (x,y) 기준 아래 
        if x+3<self.n_row:
            down=True
            coors.append(((x+1,y),(x+2,y),(x+3,y)))
        # (x,y) 기준 오른쪽 위 
        if right and up:
            coors.append(((x-1,y+1),(x-2,y+2),(x-3,y+3)))
        # (x,y) 기준 오른쪽 아래 
        if right and down:
            coors.append(((x+1,y+1),(x+2,y+2),(x+3,y+3)))
        # (x,y) 기준 왼쪽 위 
        if left and up:
            coors.append(((x-1,y-1),(x-2,y-2),(x-3,y-3)))
        # (x,y) 기준 왼쪽 아래 
        if left and down:
            coors.append(((x+1,y-1),(x+2,y-2),(x+3,y-3)))

        return coors 


    # 승패가 결정됐는지 확인하는 함수
    # 0: not end
    # 1: player 1 win
    # 2: player 2 win
    # 3: draw 
    def check_win(self):

        for x in range(self.n_row-1,-1,-1):
            for y in range(self.n_col):
                piece = self.board[x][y]
                if piece == 0: continue

                coor_list = self.coor_for_8_direction(x,y)
                for coors in coor_list:
                    print("coors:",coors)
                    if piece == self.board[coors[0]] == self.board[coors[1]] == self.board[coors[2]]:
                        self.win = piece
                        return

        if not 0 in self.board[0,:]:
            self.win = 3

    def step_human(self, col):
        self.drop_piece(col)
        self.change_player()
        self.print_board()
        self.check_win()

    def step_cpu(self):
        self.drop_piece(np.random.choice(range(self.n_col)))
        self.change_player()
        self.print_board()
        self.check_win()
    
        

    






