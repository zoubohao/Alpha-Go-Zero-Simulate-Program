from State import State
from Node import Node
from RulesOfGo import Rules
import gc
import math
import numpy as np
from  MCTS import MCTS
import random
from copy import  deepcopy


class SelfPlay:

    def __init__(self,path = None):
        #初始化棋盘状态
        nullState = State()
        #设置初始化棋盘节点
        self.nullRootNode = Node(state=nullState)
        self.nullRootNode.setSelected("yes")
        if path is None:
            self.MCTS = MCTS()
        else:
            self.MCTS = MCTS(path= path)
        self.policyNet = self.MCTS.policyNet
        self.valueNet = self.MCTS.valueNet
        self.inputData = self.MCTS.inputData
        self.probabilityData = self.MCTS.probabilityData
        self.valueData = self.MCTS.valueData
        self.learningRate = self.MCTS.learningRate
        self.lossT = self.MCTS.lossT
        self.optT = self.MCTS.optT
        self.sess = self.MCTS.sess
        self.lossP , self.lossV = self.MCTS.lossP , self.MCTS.lossV
        self.training = self.MCTS.training
        #将初始化节点加入记录
        self.stepNodeRecord = [self.nullRootNode]
        #记录反馈数据状态,好训练网络
        self.stepInputRecord = []
        self.stepProbabilityRecord = []
        self.stepValueRecord = []
        #每下一步MCTS迭代次数,默认1600次
        self.iteration = 1600
        #阈值
        #低于5%
        self.wResignValue = 0.95
        self.bResignValue = -0.95
        #这局是否需要放弃
        self.resignAbandon = self.randomJudgeResign()


    #有90的可能不会放弃这局棋
    def randomJudgeResign(self):
        ranNumber = random.random()
        if ranNumber >= 0.1:
            return False
        else:
            return True

    def terminalCondition(self):
        terminal = False
        #达到最大步数
        if len(self.stepNodeRecord) == 19 * 19 + 1 + 1:
            print("Reach the Max steps  . GAME OVER ")
            terminal = True
            return terminal
        #双方都放弃pass
        if self.stepNodeRecord[-1].state.abandon == "yes" and \
                self.stepNodeRecord[-2].state.abandon == "yes":
            print("Both pass the game . GAME OVER ")
            terminal = True
            return terminal
        if self.resignAbandon :
            # 当前节点的价值Q小于阈值
            currentRootNode = self.stepNodeRecord[-2]
            currentChildNodes = currentRootNode.childNodes
            VvalueOfNodes = [childNode.V for childNode in currentChildNodes]
            if currentRootNode.state.currentColor == "black":
                # 黑色的阈值要小于规定阈值
                # 黑输的可能性很大0.95
                if currentRootNode.V <= self.bResignValue:
                    # 且其子节点白旗最大的Q值都小于-0.90
                    # 子节点白赢得可能性很大0.90
                    if max(VvalueOfNodes) <= -0.90:
                        # 且随机数的值大于0.1
                        # 有90%的可能性放弃
                        print("Black resign . GAME OVER ")
                        terminal = True
                        return terminal
            else:
                # 白色的阈值要大于规定阈值
                if currentRootNode.V >= self.wResignValue:
                    if min(VvalueOfNodes) >= 0.90:
                        print("White resign . GAME OVER")
                        terminal = True
                        return terminal
        return terminal

    def inputDataConstruction(self):
        nodeLength = len(self.stepNodeRecord)
        #这里用的C是1表示黑棋，0表示白棋
        #使用t时刻的位置状态棋谱
        #使用NCHW格式的输入
        #选择最后一个被添加的节点
        #使用最后一步的节点
        C = self.stepNodeRecord[-1]
        cState = C.state
        inputData = []
        if nodeLength < 8 :
            for node in self.stepNodeRecord:
                nodeState = node.state
                nodeStateBlackBoard = deepcopy(nodeState.blackBoard)
                nodeStateWhiteBoard = deepcopy(nodeState.whiteBoard)
                inputData.append(nodeStateBlackBoard)
                inputData.append(nodeStateWhiteBoard)
            lenOfInputData = len(inputData)
            residual = 16 - lenOfInputData
            for i in range(residual):
                inputData.append(np.zeros(shape=[19,19]))

        else:
            dataNodes = self.stepNodeRecord[-8:]
            for node in dataNodes:
                nodeState = node.state
                nodeStateBlackBoard = np.array(nodeState.blackBoard)
                nodeStateWhiteBoard = np.array(nodeState.whiteBoard)
                inputData.append(nodeStateBlackBoard)
                inputData.append(nodeStateWhiteBoard)
        #如果叶子节点是黑色走棋
        if cState.currentColor == "black":
            cPositionStateBoard = np.ones(shape=[19,19])
            inputData.append(cPositionStateBoard)
        #如果是白色走棋
        if cState.currentColor == "white":
            cPositionStateBoard = np.zeros(shape=[19,19])
            inputData.append(cPositionStateBoard)
        inputData = np.reshape(a=inputData,newshape=[1,18,19,19])
        return inputData


    #垃圾手动进行回收
    def cutOffRemainedTreeNodes(self,nextSelectedNode):
        parentNode = nextSelectedNode.parentNode
        self.nodesSelect(parentNode)
        gc.collect()

    #对垃圾节点的选择
    def nodesSelect(self,node):
        childNodes = node.childNodes
        if childNodes is not None :
            for nodeC in childNodes:
                if nodeC.selected != "yes":
                    self.nodesSelect(nodeC)
            if node.selected != "yes":
                del node.state.abandon
                del node.state.blackBoard
                del node.state.whiteBoard
                del node.state.positionStateBoard
                del node.state.currentColor
                del node.state
                del node.parentNode
                del node.selected
                del node.position
                del node.childNodes
                del node.Q
                del node.P
                del node.W
                del node.N
                del node
        else:
            if node.selected != "yes" :
                del node.state.abandon
                del node.state.blackBoard
                del node.state.whiteBoard
                del node.state.positionStateBoard
                del node.state.currentColor
                del node.state
                del node.parentNode
                del node.selected
                del node.position
                del node.Q
                del node.P
                del node.W
                del node.N
                del node.childNodes
                del node


    def terminalNodeSelect(self,rootNode):
          #对垃圾节点的选择
        try:
            childNodes = rootNode.childNodes
            if childNodes is not None:
                for nodeC in childNodes:
                    self.terminalNodeSelect(nodeC)
                del rootNode.state.abandon
                del rootNode.state.blackBoard
                del rootNode.state.whiteBoard
                del rootNode.state.positionStateBoard
                del rootNode.state.currentColor
                del rootNode.state
                del rootNode.parentNode
                del rootNode.selected
                del rootNode.position
                del rootNode.childNodes
                del rootNode.Q
                del rootNode.P
                del rootNode.W
                del rootNode.N
                del rootNode
            else:
                del rootNode.state.abandon
                del rootNode.state.blackBoard
                del rootNode.state.whiteBoard
                del rootNode.state.positionStateBoard
                del rootNode.state.currentColor
                del rootNode.state
                del rootNode.parentNode
                del rootNode.selected
                del rootNode.position
                del rootNode.Q
                del rootNode.P
                del rootNode.W
                del rootNode.N
                del rootNode.childNodes
                del rootNode
        except  :
            pass



    def moveUntilTerminal(self):
        terminal = False
        countStep = 0
        self.MCTS.inforStep = 2
        while terminal is False:
            #设置每一次MCTS的温度大小
            #前30步棋为1
            #剩余的让温度T趋近于0
            #使用e的-x次方
            if countStep <= 30:
                self.MCTS.setTemperature(1)
            else:
                #让温度徐徐降低
                self.MCTS.setTemperature(math.exp(-(countStep - 30.0) * 0.005))
            countStep = countStep + 1
            #设置select步骤根节点起始点
            self.MCTS.setRootNode(self.stepNodeRecord[-1])
            print("\n")
            print("The color of current root node is ",self.stepNodeRecord[-1].state.currentColor)
            print("The position board of current root node is \n",self.stepNodeRecord[-1].state.positionStateBoard)
            for i in range(self.iteration):
                if i % self.MCTS.inforStep == 0 :
                    print("Current MCTS IS AT ", i)
                self.MCTS.select(i)
                self.MCTS.expand(i)
                self.MCTS.backUp()
            #返回根节点的子节点
            childNodesOfRootNode = self.MCTS.outputChildNodesOfRootNode()
            #保证根节点的有子节点
            if childNodesOfRootNode is not None :
                # 返回子节点对应的概率
                childNodesCorresposeProbability = self.MCTS.outputProbability()
                # 构造输入数据
                self.stepInputRecord.append(self.inputDataConstruction())
                # 创建19*19+1的概率记录，好做训练数据
                probabilityRecord = np.zeros(shape=[19 * 19 + 1])
                # 写入每一个步骤的可能性
                # 如果节点里面没有这个步骤，那么可能性为0
                # 遍历根节点的子节点
                for i, node in enumerate(childNodesOfRootNode):
                    # 得到这个子节点下棋的位置信息
                    position = node.position
                    # 如果这个节点是弃权节点，将弃权可能性放在最后
                    if node.state.abandon == "yes":
                        probabilityRecord[-1] = childNodesCorresposeProbability[i]
                    # 将棋盘2维位置信息转换为1维位置信息，将对应的概率赋值
                    else :
                        probabilityRecord[position[0] * 19 + position[1]] = childNodesCorresposeProbability[i]
                # 将概率信息添加到每一步的概率中
                #概率记录中最后一个是放弃的概率
                probabilityRecord = np.reshape(probabilityRecord,newshape=[1,19*19+1])
                self.stepProbabilityRecord.append(probabilityRecord)
                # 寻找概率最大的那个子节点
                indexOfMaxProbability = childNodesCorresposeProbability.index(max(childNodesCorresposeProbability))
                nextNode = childNodesOfRootNode[indexOfMaxProbability]
                ###############################
                #将下一个节点加入步数节点记录中
                self.stepNodeRecord.append(nextNode)
                self.MCTS.appendNodeToStepNodeRecord(node=nextNode)
                # 判断结束条件
                terminal = self.terminalCondition()
                ###############################
                #垃圾回收，将树的其他节点剪掉
                nextNode.setSelected("yes")
                self.cutOffRemainedTreeNodes(nextNode)
            #如果根节点没有子节点，那么循环终止
            else:
                terminal = True

    #输赢判定
    def winnerJudge(self):
        terminalNode = self.stepNodeRecord[-1]
        rule = Rules(state=terminalNode.state)
        blackOccupation , whiteOccupation = rule.winnerJudge()
        return blackOccupation , whiteOccupation

#####################################
###########主要使用下面的方法##########
#####################################
    #对自对弈做一个封装
    def selfPlayStart(self,iteration=None):
        if iteration is not None:
            self.iteration = iteration
        self.moveUntilTerminal()
        blackOccu , whiteOccu = self.winnerJudge()
        #先手的要补子，这里按中国的规则
        finalBlackOccu = blackOccu - 3.75
        finalWhiteOccu = whiteOccu + 3.75
        print("Final black occupation is ",finalBlackOccu)
        print("Final white occupation is ",finalWhiteOccu)
        #黑赢
        if finalBlackOccu > finalWhiteOccu :

            for i in range(len(self.stepInputRecord)):
                if i % 2 == 0:
                    self.stepValueRecord.append(np.reshape([-1.0],[1,1]))
                else:
                    self.stepValueRecord.append(np.reshape([1.01],[1,1]))
        #白赢
        elif finalBlackOccu < finalWhiteOccu:
            for i in range(len(self.stepProbabilityRecord)):
                if i % 2 == 0:
                    self.stepValueRecord.append(np.reshape([-1.01],[1,1]))
                else:
                    self.stepValueRecord.append(np.reshape([1.0],[1,1]))
        #平局
        else:
            for i in range(len(self.stepInputRecord)):
                if i % 2 == 0:
                    self.stepValueRecord.append(np.reshape([-1.0],[1,1]))
                else:
                    self.stepValueRecord.append(np.reshape([1.0],[1,1]))

    #返回所有的数据
    def allBoardStepDataReturn(self):
        return self.stepInputRecord , self.stepProbabilityRecord , self.stepValueRecord

    def clearAllStepData(self):
        self.terminalNodeSelect(rootNode=self.nullRootNode)
        gc.collect()
        nullRootNode = Node(state=State())
        nullRootNode.setSelected("yes")
        self.nullRootNode = nullRootNode
        self.stepNodeRecord = [self.nullRootNode]
        self.MCTS.stepNodeRecord = [self.nullRootNode]
        self.stepInputRecord = []
        self.stepProbabilityRecord = []
        self.stepValueRecord = []


################
#测试
if __name__ == "__main__":
    selfPlay = SelfPlay()
    selfPlay.selfPlayStart()
    I , P ,V = selfPlay.allBoardStepDataReturn()
    print(P)

















