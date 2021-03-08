import numpy as np

class SudokuBoard:
    def __init__(self):
        self.board = np.zeros((9,9), dtype=np.uint8 )

    def is_valid_placing(self,num,pos):
        fst = pos[0] // 3
        snd = pos[0] // 3 
        return np.all(self.board[fst::3, snd::3]!=num) and np.all(self.board[fst::3,snd::3] != num) and self.board[pos[0],pos[1]] == 0 and np.all(self.board[pos[0]]!=num)

    def insert(self,num,pos):
        if self.is_valid_placing(num,pos):
            self.board[pos[0],pos[1]] = num
            return True
        else:
            return False

    def get_row_with_most_numbers(self):
        return np.count_nonzero(self.board,axis=1).argmax()

    def get_possible_cols(self,num):
        return np.intersect1d(np.where(np.all((self.board!=num), axis=0)),np.where(np.all((self.board!=num), axis=1)))

    def sum_of_missing_numbers_per_square(self):
        return np.subtract(np.full((1,9),45),(np.sum(self.board, axis=1))).reshape(3,3)




sb = SudokuBoard()
print(sb.board.shape)
print(np.all(sb.board==0))
sb.board[3,3]=5
print(sb.board)
print(sb.is_valid_placing(5,(0,0)))
print(sb.is_valid_placing(5,(5,4)))
print(sb.get_row_with_most_numbers())
print(sb.get_possible_cols(5))
print(sb.sum_of_missing_numbers_per_square())
#print(np.indices((9,9)))
'''
[[0 0 0  [0 0 0  [0 0 0
  0 0 0   0 0 0   0 0 0
  0 0 0]  0 0 0]  0 0 0]
              
   
 [0 0 0  [0 0 0  [0 0 0
  5 0 0   0 0 0   0 5 0
  0 0 0]  0 0 0]  0 0 0]

  [0 0 0 [0 0 0  [0 0 0
   0 0 0  0 0 0   0 0 0
   0 0 0] 0 0 0]  0 0 0]]
   
   
   
   
   
 '''