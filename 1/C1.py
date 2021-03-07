import numpy as np

class SudokuBoard:
    def __init__(self):
        self.board = np.zeros((9,9), dtype=np.int32 )

    def is_valid_placing(num,pos):
        return


sb = SudokuBoard()
print(sb.board.shape)
print(np.all(sb.board==0))
print(sb.board[3,3])
sb.board[0,2]=1
sb.board[0,1]=1
print(np.all(sb.board[::1,1]==0))