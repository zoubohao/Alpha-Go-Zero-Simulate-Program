import numpy as np
from State import State
from copy import deepcopy

class Rules :


    #使用Tromp–Taylor作为基本计分规则，在输赢的计算上改进，使之更容易计算
    #围棋的基本两条规则：
    #1、不许重复棋盘的形态
    #2、任何一个存在于棋盘上的棋子要有气，没气就必须提子
    def __init__(self,state):
        self.blackBoard = deepcopy(state.blackBoard)
        self.whiteBoard = deepcopy(state.whiteBoard)
        #别把引用复制了，不然就是同一个东西
        self.checkWhiteBoard = deepcopy(self.whiteBoard)
        self.checkBlackBoard = deepcopy(self.blackBoard)
        self.libertiesInOneWhiteCluster = []
        self.libertiesInOneBlackCluster = []
        self.whiteCluster = []
        self.blackCluster = []
        self.positionStateBoard = deepcopy(state.positionStateBoard)
        self.currentColor = state.currentColor


    #实现棋盘上棋子的气，保证每簇棋子有气存在
    #获得白棋簇，并给出这个簇的气，簇记录在self.whiteCluster中，气记录在self.libertiesInOneWhiteCluster中
    def whiteOneClusterSearch(self,positionX,positionY):
        listAdjacent = []
        self.whiteCluster.append([positionX,positionY])
        try:
            if positionX - 1 >= 0 and positionY >= 0:
                if self.checkWhiteBoard[positionX - 1, positionY] == 1:
                    listAdjacent.append([positionX - 1, positionY])
                    self.whiteCluster.append([positionX - 1, positionY])
                else:
                    if self.whiteBoard[positionX - 1, positionY] != 1:
                        self.libertiesInOneWhiteCluster.append([positionX - 1, positionY])
        except:
            pass

        try:
            if positionX >= 0 and positionY -1 >= 0:
                if self.checkWhiteBoard[positionX, positionY - 1] == 1:
                    listAdjacent.append([positionX, positionY - 1])
                    self.whiteCluster.append([positionX, positionY - 1])
                else:
                    if self.whiteBoard[positionX, positionY - 1] != 1:
                        self.libertiesInOneWhiteCluster.append([positionX, positionY - 1])
        except:
            pass

        try:
            if positionX >= 0 and positionY + 1 >= 0:
                if self.checkWhiteBoard[positionX, positionY + 1] == 1:
                    listAdjacent.append([positionX, positionY + 1])
                    self.whiteCluster.append([positionX, positionY + 1])
                else:
                    if self.whiteBoard[positionX, positionY + 1] != 1:
                        self.libertiesInOneWhiteCluster.append([positionX, positionY + 1])
        except:
            pass

        try:
            if positionX+1 >= 0 and positionY  >= 0:
                if self.checkWhiteBoard[positionX + 1, positionY] == 1:
                    listAdjacent.append([positionX + 1, positionY])
                    self.whiteCluster.append([positionX + 1, positionY])
                else:
                    if self.whiteBoard[positionX + 1, positionY] != 1:
                        self.libertiesInOneWhiteCluster.append([positionX + 1, positionY])
        except:
            pass

        self.checkWhiteBoard[positionX, positionY] = 0

        for position in listAdjacent:
            self.whiteOneClusterSearch(position[0],position[1])

    #扫描整个棋盘，去除死去的白棋
    def whiteClustersSearchAndEradicate(self):
        for i in range(19):
            for j in range(19):
                if self.checkWhiteBoard[i,j] == 1:
                    self.whiteOneClusterSearch(i,j)
                    checkLiberties = []
                    for liberty in self.libertiesInOneWhiteCluster :
                        #这个气没有被黑棋填上，该簇可以存活
                        #print("libery is ",liberty)
                        if self.blackBoard[liberty[0],liberty[1]] == 0:
                            break
                        else:
                            checkLiberties.append("FULL")
                    #print("length is ",len(checkLiberties))
                    #不然就去除这个簇中的白棋
                    if len(checkLiberties) == len(self.libertiesInOneWhiteCluster):
                        for position in self.whiteCluster:
                            self.whiteBoard[position[0],position[1]] = 0
                            self.positionStateBoard[position[0],position[1]] = 0
                    self.whiteCluster = []
                    self.libertiesInOneWhiteCluster = []

    def blackOneClusterSearch(self,positionX,positionY):
        listAdjacent = []
        self.blackCluster.append([positionX,positionY])
        try:
            if positionX - 1 >= 0 and positionY >= 0:
                if self.checkBlackBoard[positionX - 1, positionY] == 1:
                    listAdjacent.append([positionX - 1, positionY])
                    self.blackCluster.append([positionX - 1, positionY])
                else:
                    if self.blackBoard[positionX - 1, positionY] != 1:
                        self.libertiesInOneBlackCluster.append([positionX - 1, positionY])
        except:
            pass

        try:
            if positionX >= 0 and positionY - 1 >= 0:
                if self.checkBlackBoard[positionX, positionY - 1] == 1:
                    listAdjacent.append([positionX, positionY - 1])
                    self.blackCluster.append([positionX, positionY - 1])
                else:
                    if self.blackBoard[positionX, positionY - 1] != 1:
                        self.libertiesInOneBlackCluster.append([positionX, positionY - 1])
        except:
            pass

        try:
            if positionX >= 0 and positionY + 1 >= 0:
                if self.checkBlackBoard[positionX, positionY + 1] == 1:
                    listAdjacent.append([positionX, positionY + 1])
                    self.blackCluster.append([positionX, positionY + 1])
                else:
                    if self.blackBoard[positionX, positionY + 1] != 1:
                        self.libertiesInOneBlackCluster.append([positionX, positionY + 1])
        except:
            pass

        try:
            if positionX +1 >= 0 and positionY  >= 0:
                if self.checkBlackBoard[positionX + 1, positionY] == 1:
                    listAdjacent.append([positionX + 1, positionY])
                    self.blackCluster.append([positionX + 1, positionY])
                else:
                    if self.blackBoard[positionX + 1, positionY] != 1:
                        self.libertiesInOneBlackCluster.append([positionX + 1, positionY])
        except:
            pass

        self.checkBlackBoard[positionX, positionY] = 0

        for position in listAdjacent:
            self.blackOneClusterSearch(position[0],position[1])


    def blackClustersSearchAndEradicate(self):
        for i in range(19):
            for j in range(19):
                if self.checkBlackBoard[i,j] == 1:
                    self.blackOneClusterSearch(i,j)
                    checkLiberties = []
                    for liberty in self.libertiesInOneBlackCluster :
                        #这个气没有被白棋填上，该簇可以存活
                        if self.whiteBoard[liberty[0],liberty[1]] == 0:
                            break
                        else:
                            checkLiberties.append("FULL")
                    #不然就去除这个簇中的黑棋
                    if len(checkLiberties) == len(self.libertiesInOneBlackCluster):
                        for position in self.blackCluster:
                            self.blackBoard[position[0],position[1]] = 0
                            self.positionStateBoard[position[0],position[1]] = 0
                    self.blackCluster = []
                    self.libertiesInOneBlackCluster = []

    #实现棋盘上的棋子不能重复上一回合的形，防止重复打劫，在MCTS模块中有
    #计算黑棋与白棋的子数，并将每个簇的气也考虑进去。
    #如果一个气的周围有白棋和黑棋的话，各加0.5，不然这个气就是黑的或者白的
    #如果到棋局终点还有没有下完的地方，除了气其他空白区域不计入计算
    def winnerJudge(self):
        blackCount = 0
        blackLiberitesTotalSet = set()
        whiteCount = 0
        whiteLiberitesTotleSet = set()
        for i in range(19):
            for j in range(19):
                if self.checkBlackBoard[i,j] == 1:
                    self.blackOneClusterSearch(i,j)
                    #去除重复的黑棋
                    blackPointsSet = set()
                    for position in self.blackCluster:
                        position = tuple(position)
                        blackPointsSet.add(position)
                    #将簇清空，为下一次簇的添加做准备
                    self.blackCluster = []
                    blackCount = blackCount + blackPointsSet.__len__()
                    #去除重复的棋眼
                    blackLiberitesSet = set()
                    for position in self.libertiesInOneBlackCluster :
                        position = tuple(position)
                        blackLiberitesSet.add(position)
                    #气的清空，为下一次气的添加做准备
                    self.libertiesInOneBlackCluster = []
                    #遍历黑棋气眼
                    for positionL in blackLiberitesSet:
                        if not blackLiberitesTotalSet.__contains__(positionL) :
                            # 这个气没有被白棋堵上
                            if self.whiteBoard[positionL[0], positionL[1]] == 0:
                                # 检查这个气周围有没有白棋
                                # 有白棋则黑棋占领数目加0.5
                                try:
                                    if self.whiteBoard[positionL[0] - 1, positionL[1]] == 1:
                                        blackCount = blackCount + 0.5
                                        blackLiberitesTotalSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.whiteBoard[positionL[0], positionL[1] - 1] == 1:
                                        blackCount = blackCount + 0.5
                                        blackLiberitesTotalSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.whiteBoard[positionL[0] + 1, positionL[1]] == 1:
                                        blackCount = blackCount + 0.5
                                        blackLiberitesTotalSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.whiteBoard[positionL[0], positionL[1] + 1] == 1:
                                        blackCount = blackCount + 0.5
                                        blackLiberitesTotalSet.add(positionL)
                                        continue
                                except:
                                    pass
                                # 周围都没白棋，黑棋占领数目加1
                                blackCount = blackCount + 1
                                blackLiberitesTotalSet.add(positionL)
                #计算白棋子数
                if self.checkWhiteBoard[i,j] == 1:
                    self.whiteOneClusterSearch(i,j)
                    #去除重复的白棋
                    whitePointsSet = set()
                    for position in self.whiteCluster:
                        position = tuple(position)
                        whitePointsSet.add(position)
                    self.whiteCluster = []
                    whiteCount =whiteCount + whitePointsSet.__len__()
                    #去除重复的棋眼
                    whiteLiberitesSet = set()
                    for position in self.libertiesInOneWhiteCluster :
                        position = tuple(position)
                        whiteLiberitesSet.add(position)
                    self.libertiesInOneWhiteCluster = []
                    #遍历白棋气眼
                    for positionL in whiteLiberitesSet:
                        if  not whiteLiberitesTotleSet.__contains__(positionL) :
                            # 这个气没有被黑棋堵上
                            if self.blackBoard[positionL[0], positionL[1]] == 0:
                                # 检查这个气周围有没有黑棋
                                # 有黑棋则白棋占领数目加0.5
                                try:
                                    if self.blackBoard[positionL[0] - 1, positionL[1]] == 1:
                                        whiteCount = whiteCount + 0.5
                                        whiteLiberitesTotleSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.blackBoard[positionL[0], positionL[1] - 1] == 1:
                                        whiteCount = whiteCount + 0.5
                                        whiteLiberitesTotleSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.blackBoard[positionL[0] + 1, positionL[1]] == 1:
                                        whiteCount = whiteCount + 0.5
                                        whiteLiberitesTotleSet.add(positionL)
                                        continue
                                except:
                                    pass
                                try:
                                    if self.blackBoard[positionL[0], positionL[1] + 1] == 1:
                                        whiteCount = whiteCount + 0.5
                                        whiteLiberitesTotleSet.add(positionL)
                                        continue
                                except:
                                    pass
                                # 周围都没黑棋，白棋占领数目加1
                                whiteCount = whiteCount + 1
                                whiteLiberitesTotleSet.add(positionL)
        return blackCount , whiteCount





#############################################################################
#测试#

if __name__ == "__main__":
    testState = State()
    blackBoard = np.zeros(shape=[19,19])
    positionBoard = np.zeros(shape=[19,19])
    whiteBoard = np.zeros(shape=[19,19])
    whiteBoard[0,1] = 1
    positionBoard[0,1] = -1
    whiteBoard[0,3] = 1
    positionBoard[0,3] = -1
    blackBoard[0,2] = 1
    positionBoard[0,2] = 1

    testState.setBlackBoard(blackBoard)
    testState.setWhiteBoard(whiteBoard)
    testState.setpositionStateBoard(positionBoard)
    testState.setCurrentColor("white")
    print("black board" , blackBoard)
    print("white board",whiteBoard)
    testState.moveWhiteStoneOnce(1, 2)
    rules = Rules(state=testState)
    #rules.blackClustersSearchAndEradicate()
    #rules.whiteClustersSearchAndEradicate()
    blackCount , whiteCount = rules.winnerJudge()
    print(blackCount)
    print(whiteCount)
    print(rules.blackBoard)
    print(rules.whiteBoard)

























