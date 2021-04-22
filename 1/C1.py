import numpy as np

class SudokuBoard:
    def __init__(self):
        self.board = np.zeros((9,9), dtype=np.uint8 )

    def is_valid_placing(self,num,pos):
        fst = (pos[0] // 3 ) * 3
        snd = (pos[1] // 3 ) * 3
        return np.all(self.board[::1, pos[1]]!=num) and np.all(self.board[pos[0], ::1 ]!= num) and self.board[pos[0],pos[1]] == 0 and np.all(self.board[fst:fst+3:1,snd:snd+3:1]!=num)

    def insert(self,num,pos):
        if self.is_valid_placing(num,pos):
            self.board[pos[0],pos[1]] = num
            return True
        else:
            return False

    def get_row_with_most_numbers(self):
        return np.argmax(np.count_nonzero(self.board,axis=1))

    def get_possible_cols(self,num):

        result = np.array(np.where(np.all(self.board!=num,axis=0)))[0]
        return result

    def sum_of_missing_numbers_per_square(self):
        blocks = np.sum(np.array(self.board).reshape((3,3,3,3)).transpose((0,2,1,3)).reshape(9,9),axis = 1)
        result = np.subtract(np.full((9),45),blocks). reshape(3,3)
        return result



import unittest

class TestSudoku(unittest.TestCase):
    def test_is_valid_placing(self):
        sb = SudokuBoard()
        self.assertEqual(sb.board.shape, (9,9))
        self.assertTrue(np.all(sb.board == 0))
        self.assertTrue(sb.is_valid_placing(5, (0,0)))
        self.assertTrue(sb.is_valid_placing(5, (8,0)))
        self.assertTrue(sb.is_valid_placing(5, (0,8)))
        self.assertTrue(sb.is_valid_placing(5, (4,6)))
        self.assertTrue(sb.is_valid_placing(5, (8,8)))

        sb.board[3,3] = 5
        self.assertFalse(sb.is_valid_placing(7, (3,3)))
        self.assertFalse(sb.is_valid_placing(5, (5,4)))
        self.assertFalse(sb.is_valid_placing(5, (7,3)))
        self.assertFalse(sb.is_valid_placing(5, (3,7)))
        
        self.assertTrue(sb.is_valid_placing(5, (2,4)))
        self.assertTrue(sb.is_valid_placing(6, (3,4)))

    def test_insert(self):
        sb = SudokuBoard()
        self.assertTrue(sb.insert(4, (2,7)))
        self.assertEqual(sb.board[2,7], 4)
        self.assertFalse(sb.insert(4, (2,7)))

    def test_get_row_with_most_numbers(self):
        sb = SudokuBoard()
        sb.insert(3, (5,6))
        self.assertEqual(sb.get_row_with_most_numbers(), 5)
        
        to_insert = [(3, (1,0)),(6, (1,7)), (4, (2,1)),(1, (3,7))]
        [sb.insert(*t) for t in to_insert] 
        self.assertEqual(sb.get_row_with_most_numbers(), 1)

        to_insert = [(9, (2,3)), (4, (5,8)), (6, (5,1))]
        [sb.insert(*t) for t in to_insert] 
        self.assertEqual(sb.get_row_with_most_numbers(), 5)

    def test_get_possible_cols(self):
        sb = SudokuBoard()
        self.assertEqual(len(sb.get_possible_cols(5)), 9)
        
        sb.insert(5, (5,8))
        self.assertEqual(len(sb.get_possible_cols(5)), 8)

        sb.insert(6, (4,3))
        self.assertEqual(len(sb.get_possible_cols(5)), 8)

        to_insert = [(5, (1,0)),(5, (2,7)), (5, (4,4))]
        [sb.insert(*t) for t in to_insert]
        self.assertEqual(len(sb.get_possible_cols(5)), 5)

    def test_sum_of_missing_numbers_per_square(self):
        sb = SudokuBoard()
        sum_missing = sb.sum_of_missing_numbers_per_square()
        self.assertTrue(np.all(sum_missing == 45))
        self.assertEqual(sum_missing.shape, (3,3))

        to_insert = [(6, (6,5)),(7, (8,5)), (8, (7,5))]
        [sb.insert(*t) for t in to_insert]
        sum_missing = sb.sum_of_missing_numbers_per_square()
        self.assertEqual(sum_missing[2,1], 24)
        self.assertEqual(np.sum(sum_missing == 45), 8)

        to_insert = [(1, (3,0)),(2, (4,2)), (3, (4,1))]
        [sb.insert(*t) for t in to_insert]
        sum_missing = sb.sum_of_missing_numbers_per_square()
        
        self.assertEqual(sum_missing[1,0], 39)
        self.assertEqual(sum_missing[2,1], 24)
        self.assertEqual(np.sum(sum_missing == 45), 7)

def suite():
    suite = unittest.TestSuite()
    testfuns = ["test_is_valid_placing", "test_insert", 
                "test_get_row_with_most_numbers",
                "test_get_possible_cols",
                "test_sum_of_missing_numbers_per_square",
                ]
    [suite.addTest(TestSudoku(fun)) for fun in testfuns]
    return suite

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite())

#print(np.indices((9,9)))
'''
[[0 0 0  [0 0 0  [3 0 0
  0 0 0   0 0 0   0 0 0
  0 0 0]  0 0 0]  0 0 0]
              
   
 [0 0 0  [0 0 0  [0 0 0
  5 0 0   0 0 0   0 0 0
  0 0 0]  0 0 0]  0 0 0]

  [0 0 0 [0 0 0  [0 0 0
   0 0 6  0 0 8   0 0 7  
   0 0 0] 0 0 0]  0 0 0]]


[[0 0 0  [0 0 0  [0 0 0
  0 0 0   0 0 0   0 y 0 # 2,4 nyilván jó
  0 0 0]  0 0 0]  0 0 0]
                 
 [0 0 0  [0 0 0  [0 0 0
  5 y 0   0 0 0   0 0 0  # beszúrtuk az 5-öst a 4. blokkba (3,3)
  0 x 0]  0 0 0]  0 0 0] # 3,7 érthetően nem jó, ugyanaz a blokk

 [0 0 0 [0 0 0  [0 0 0
  0 0 0  x 0 0   0 0 0  # 7,3 nem jó, de miért?
  0 0 0] 0 0 0]  0 0 0]]
   
   
   
   
   
 '''