import numpy as np
from copy import deepcopy

class State :

     #棋盘状态一律用numpy来表示
     #状态的初始化
     #如果state = None,则初始化状态，步数为0；否则继承状态,步数加一
     #黑棋子状态：黑棋所占交叉点为1，其余为0
     #白棋子状态：白棋所占交叉点为1，其余为0
     #位置状态：黑子所占为1，空为0，白棋为-1
     def __init__(self,state = None):
         if state is None:
             self.blackBoard = np.zeros(shape=[19, 19])
             self.whiteBoard = np.zeros(shape=[19, 19])
             self.positionStateBoard = np.zeros(shape=[19, 19])
             self.step = 0
             # 黑棋先手,所以它的子节点是黑棋走一步，因为要继承它的状态，在继承的时候，子节点的颜色就是黑色
             self.currentColor = "white"
             #默认是不放弃的状态
             self.abandon = "no"
         else:
             self.blackBoard = deepcopy(state.blackBoard)
             self.whiteBoard = deepcopy(state.whiteBoard)
             self.positionStateBoard = deepcopy(state.positionStateBoard)
             if state.currentColor == "black" :
                 self.currentColor = "white"
             else:
                 self.currentColor = "black"
             self.step = state.step + 1
             #默认是不放弃的状态
             self.abandon = "no"

     # 改变黑棋子状态，黑棋下一步
     def moveBlackStoneOnce(self, positionX, positionY):
         self.blackBoard[positionX, positionY] = 1
         self.positionStateBoard[positionX, positionY] = 1

     # 改变白棋子状态，白棋下一步
     def moveWhiteStoneOnce(self, positionX, positionY):
         self.whiteBoard[positionX, positionY] = 1
         self.positionStateBoard[positionX, positionY] = -1

     #设置
     def setStep(self,step):
         self.step = step

     def setBlackBoard(self,blackBoard):
         blackBoardConfirm = np.matrix(blackBoard)
         assert blackBoardConfirm.shape[0] == 19 and blackBoardConfirm.shape[1] == 19
         self.blackBoard = deepcopy(blackBoard)
     
     def setWhiteBoard(self,whiteBoard):
         whiteBoardConfirm = np.matrix(whiteBoard)
         assert whiteBoardConfirm.shape[0] == 19 and whiteBoardConfirm.shape[1] == 19
         self.whiteBoard = deepcopy(whiteBoard)
     
     def setpositionStateBoard(self,positionStateBoard):
         positionStateBoardConfirm = np.matrix(positionStateBoard)
         assert positionStateBoardConfirm.shape[0] == 19 and positionStateBoardConfirm.shape[1] == 19
         self.positionStateBoard = deepcopy(positionStateBoard)
     
     def setCurrentColor(self,color):
         self.currentColor = color

     def setAbandonState(self,abandon):
         self.abandon = abandon




     



