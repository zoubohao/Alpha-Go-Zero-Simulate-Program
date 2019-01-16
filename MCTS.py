from Node import Node
from State import State
from Net import Net
import numpy as np
import tensorflow as tf
from RulesOfGo import Rules
from copy import deepcopy
import math
import time

class MCTS :

    def __init__(self,path = None):
        self.Cpuct = 0.6
        rootState = State()
        nullRootNode = Node(state=rootState)
        nullRootNode.setSelected("yes")
        self.rootNode = nullRootNode
        self.selectedNodes = []
        # 设置最初始的状态，空棋盘
        self.stepNodeRecord = [nullRootNode]
        #网络的正向传播计算概率输出
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.net = Net()
        self.policyNet , self.valueNet = self.net.policyNet,self.net.valueNet
        self.inputData = self.net.inputPlaceHolder
        self.probabilityData = self.net.porbPlaceHolder
        self.valueData = self.net.valuePlaceHolder
        self.learningRate = self.net.learningRate
        self.lossT , self.optT = self.net.lossT , self.net.opt
        self.lossP , self.lossV = self.net.lossP ,self.net.lossV
        self.training = self.net.training
        if path is None :
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess,path)
        #默认温度是1
        self.T = 1
        self.inforStep = 1

    #为重新使用这个树做准备，设置树的根节点
    def setRootNode(self,rootNode):
        self.rootNode = rootNode

    #设置温度T值
    def setTemperature(self,T):
        self.T = T

    def setCpuct(self,Cpuct):
        self.Cpuct = Cpuct

    #将下一步选择的节点添加进去
    def appendNodeToStepNodeRecord(self,node):
        self.stepNodeRecord.append(node)

    #输入数据构造
    def inputStateDataConstruction(self):
        #input数据不止是要当前选择节点的数据来构建，
        #还需要rootNode原来节点的信息来构建
        nodes = []
        for node in self.stepNodeRecord:
            nodes.append(node)
        for n,node in enumerate(self.selectedNodes):
            if n != 0 :
                nodes.append(node)
        nodeLength = len(nodes)
        #这里用的C是1表示黑棋，0表示白棋
        #使用t时刻的位置状态棋谱
        #使用NCHW格式的输入
        #选择最后一个被添加的节点
        #也就是叶子节点
        C = nodes[-1]
        cState = C.state
        inputData = []
        if nodeLength < 8 :
            for node in nodes:
                nodeState = node.state
                nodeStateBlackBoard = deepcopy(nodeState.blackBoard)
                nodeStateWhiteBoard = deepcopy(nodeState.whiteBoard)
                inputData.append(nodeStateBlackBoard)
                inputData.append(nodeStateWhiteBoard)
            lenOfInputData = len(inputData)
            residual = 16 - lenOfInputData
            for n in range(residual):
                inputData.append(np.zeros(shape=[19,19]))

        else:
            dataNodes = nodes[-8:]
            for node in dataNodes:
                nodeState = node.state
                nodeStateBlackBoard = deepcopy(nodeState.blackBoard)
                nodeStateWhiteBoard = deepcopy(nodeState.whiteBoard)
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

        inputData = np.reshape(a=inputData,newshape=[1,17,19,19])
        return inputData

    #走一步并且确认是否符合规则
    def moveAStepAndCheck (self,state,positionX,positionY):
        situation = True
        #如果是黑棋走
        if state.currentColor == "black":
            #计算当前棋盘上的黑子数目
            stateBlackBoardCount = sum(np.reshape(state.blackBoard, [19 * 19]))
            # 新子节点状态下一个黑子
            state.moveBlackStoneOnce(positionX, positionY)
            # 用rules去判断死去的子，使用更新的状态
            rules = Rules(state=state)
            rules.whiteClustersSearchAndEradicate()
            rules.blackClustersSearchAndEradicate()
            # 如果在去除了死子之后黑色的子还比在走了一步之后还要少或者相等
            # 那么就是走了自杀棋，不允许
            # 跳过这个节点的创建
            rulesBlackBoardCount = sum(np.reshape(rules.blackBoard,[19*19]))
            if rulesBlackBoardCount <= stateBlackBoardCount:
                situation = False
            #检查下了黑子之后，棋子的局面和前第2回合棋子的局面，防止重复的局面发生
            if len(self.selectedNodes) > 1:
                if (np.array(rules.positionStateBoard) ==
                    np.array(self.selectedNodes[-2].state.positionStateBoard)) \
                        .all():
                    situation = False
            return situation , rules
        #如果是白棋走
        else:
            stateWhiteBoardCount = sum(np.reshape(state.whiteBoard, [19 * 19]))
            state.moveWhiteStoneOnce(positionX, positionY)
            rules = Rules(state=state)
            rules.blackClustersSearchAndEradicate()
            rules.whiteClustersSearchAndEradicate()
            rulesWhiteBoardCount = sum(np.reshape(rules.whiteBoard,[19*19]))
            if rulesWhiteBoardCount <= stateWhiteBoardCount:
                situation = False
            if len(self.selectedNodes) > 1:
                if (np.array(rules.positionStateBoard) ==
                    np.array(self.selectedNodes[-2].state.positionStateBoard)) \
                        .all():
                    situation = False
            return situation , rules

    #puct算法
    def PUCTAlgorithm(self,P,N,sumN):
        return self.Cpuct * P *(math.sqrt(sumN)/ (1.0 + N))

    def select(self,times):
        #在完成回溯之后，将已选节点设置为空列表
        self.selectedNodes = []
        #每次从设置的树的根节点开始
        rootNode = self.rootNode
        currentChildNodes = rootNode.childNodes
        selectedNodes = [rootNode]
        #不是子节点,选择
        while currentChildNodes is not None:
            currentMaxNode = None
            currentMaxQaddU = 0
            caculate = 0
            sumN = 0
            for node in currentChildNodes:
                sumN = sumN + node.N
            for n,node in enumerate(currentChildNodes):
                if n == 0:
                    currentMaxNode = node
                    currentMaxQaddU = caculate = self.PUCTAlgorithm(node.P,node.N,sumN) + node.Q
                else:
                    caculate = self.PUCTAlgorithm(node.P,node.N,sumN) + node.Q
                    if caculate > currentMaxQaddU :
                        currentMaxNode = node
                        currentMaxQaddU = caculate
                #print("max is ",currentMaxQaddU)
                #print("Q add U is ",caculate)
            selectedNodes.append(currentMaxNode)
            currentChildNodes = currentMaxNode.childNodes
        self.selectedNodes = selectedNodes
        if times % self.inforStep == 0:
             print("One current simulate selection have selected nodes length : ", len(selectedNodes))

    # policy net work head
    # 输出19*19+1个节点
    # value head
    # 输出一个最终值
    def expand(self,times):
        inputData = self.inputStateDataConstruction()
        leafNode = self.selectedNodes[-1]
        leafNodeState = leafNode.state
        leafNodeChildNodes = []
        policyOutput = self.sess.run(self.policyNet,feed_dict={self.net.inputPlaceHolder:inputData,
                                                               self.training:False})
        #valueOutput = self.sess.run(self.valueNet,feed_dict={self.net.inputData:inputData})
        policyOutput = np.array(policyOutput)
        policyOutput = np.reshape(policyOutput,newshape=[19*19+1])
        #前19*19为棋盘的优先级输出
        positionProbabilityOutput = policyOutput[0:-1]
        positionProbabilityOutput = np.reshape(positionProbabilityOutput,newshape=[19,19])
        #print("position probability output .",positionProbabilityOutput)
        #最后一个为放弃子的优先级输出
        abandonOutput = policyOutput[-1]
        #当前叶子节点的价值输出
        #stateValueOutput = float(np.array(valueOutput))
        if times % self.inforStep == 0 :
            print("Policy Net Abandon output is ", abandonOutput)
            #print("Value Net StateValueOutput is ",stateValueOutput)
        #设置叶子节点的值，在expand中完成对叶子节点的回溯
        #leafNode.setW(w=stateValueOutput + leafNode.W)
        ##leafNode.setN(n=leafNode.N + 1)
        #leafNode.setQ(q=leafNode.W / leafNode.N + 0.0)
        #向rootNode,S0的概率中添加噪声,论文中是添加狄利克雷分布噪声
        #但是没有找到相关信息，所以添加截断高斯噪声
        #对叶子节点扩展############
        if leafNode.parentNode is None:
            positionBoard = deepcopy(leafNodeState.positionStateBoard)
            if times % self.inforStep == 0:
                print("Add noise .")
            for n in range(19):
                for j in range(19):
                    #位置是空的才可以走
                    if positionBoard[n,j] == 0:
                        #噪声方差
                        noise = self.sess.run(tf.truncated_normal(shape=[1],
                                                                 stddev=0.8
                                                                , dtype=tf.float32))
                        noise = np.array(noise)
                        positionProbabilityOutput[n,j] = (1-0.25)*positionProbabilityOutput[n,j]+noise*0.25
                        # 创建子状态
                        # 新的子状态继承叶子节点的状态
                        #自动转换棋子的颜色状态
                        #自动添加步数
                        newNodeState = State(state=leafNodeState)
                        situation , rulesCheck = self.moveAStepAndCheck(newNodeState,n,j)
                        if situation is False:
                            continue
                        else:
                            # 将更新过后的黑，白，位置面盘赋值给子状态
                            newNodeState.blackBoard = rulesCheck.blackBoard
                            newNodeState.whiteBoard = rulesCheck.whiteBoard
                            newNodeState.positionStateBoard = rulesCheck.positionStateBoard
                            # 创建节点,设置子节点参数
                            newNode = Node(state=newNodeState)
                            newNode.setPosition([n,j])
                            newNode.setParentNode(parentNode=leafNode)
                            newNode.setP(positionProbabilityOutput[n,j])
                            self.selectedNodes.append(newNode)
                            inputData = self.inputStateDataConstruction()
                            valueOut = self.sess.run(self.valueNet,feed_dict={self.inputData:inputData
                                ,self.training:False})
                            valueOut = valueOut[0][0]
                            #if times % self.inforStep == 0:
                            #     print("new node value out is ",valueOut)
                            newNode.setV(valueOut)
                            self.selectedNodes.pop()
                            leafNodeChildNodes.append(newNode)
            #放弃节点扩展
            newAbandonState = State(state=leafNodeState)
            newAbandonState.abandon = "yes"
            newAbandonNode = Node(state=newAbandonState)
            newAbandonNode.setPosition([-1])
            newAbandonNode.setParentNode(parentNode=leafNode)
            newAbandonNode.setP(p=abandonOutput)
            self.selectedNodes.append(newAbandonNode)
            inputData = self.inputStateDataConstruction()
            valueOut = self.sess.run(self.valueNet, feed_dict={self.inputData: inputData,self.training:False})
            valueOut = valueOut[0][0]
            if times % self.inforStep == 0:
                print("abandon value out is ", valueOut)
            newAbandonNode.setV(valueOut)
            self.selectedNodes.pop()
            leafNodeChildNodes.append(newAbandonNode)
            #所有子节点添加完毕后，将其加入叶子节点的子节点，实现扩展
            #必须要有节点
            if times % self.inforStep == 0:
                print("Expand Root ChildNodes number is " , len(leafNodeChildNodes))
            if len(leafNodeChildNodes) != 0 :
                leafNode.setChildNodes(leafNodeChildNodes)
        #其他节点不添加噪声
        else:
            if times % self.inforStep == 0:
                print("Not add noise .")
            #如果所选节点大于2步,
            #包括根节点在内的节点，如果2步双方都不下了，游戏结束，不能添加任何节点
            if len(self.selectedNodes) >= 2 :
                countAbandon = 0
                for n in range(1,3):
                    if self.selectedNodes[-n].state.abandon == "yes":
                        countAbandon = countAbandon + 1
                # 所选节点放弃状态小于2步，可以直接添加节点
                if countAbandon < 2 :
                    #普通节点添加
                    positionBoard = np.array(leafNodeState.positionStateBoard)
                    for n in range(19):
                        for j in range(19):
                            if positionBoard[n, j] == 0:
                                # 创建子子状态
                                # 新的子状态继承叶子节点的状态
                                # 自动改变颜色状态
                                # 自动添加步数
                                newNodeState = State(state=leafNodeState)
                                #在i,j位置下一步棋
                                situation, rulesCheck = self.moveAStepAndCheck(newNodeState, n, j)
                                #判断是否符合规则，不符合跳过
                                if situation is False:
                                    continue
                                else:
                                    #符合
                                    # 将更新过后的黑，白，位置面盘赋值给子状态
                                    newNodeState.blackBoard = rulesCheck.blackBoard
                                    newNodeState.whiteBoard = rulesCheck.whiteBoard
                                    newNodeState.positionStateBoard = rulesCheck.positionStateBoard
                                    # 创建节点
                                    newNode = Node(state=newNodeState)
                                    newNode.setPosition([n, j])
                                    newNode.setParentNode(parentNode=leafNode)
                                    newNode.setP(positionProbabilityOutput[n,j])
                                    self.selectedNodes.append(newNode)
                                    inputData = self.inputStateDataConstruction()
                                    valueOut = self.sess.run(self.valueNet, feed_dict={self.inputData: inputData
                                        , self.training: False})
                                    valueOut = valueOut[0][0]
                                    #if times % self.inforStep == 0:
                                    #     print("new node value out is ", valueOut)
                                    newNode.setV(valueOut)
                                    self.selectedNodes.pop()
                                    leafNodeChildNodes.append(newNode)
                    #放弃节点添加
                    newAbandonState = State(state=leafNodeState)
                    newAbandonState.abandon = "yes"
                    newAbandonNode = Node(state=newAbandonState)
                    newAbandonNode.setPosition([-1])
                    newAbandonNode.setParentNode(parentNode=leafNode)
                    newAbandonNode.setP(p=abandonOutput)
                    self.selectedNodes.append(newAbandonNode)
                    inputData = self.inputStateDataConstruction()
                    valueOut = self.sess.run(self.valueNet, feed_dict={self.inputData: inputData
                        , self.training: False})
                    valueOut = valueOut[0][0]
                    if times % self.inforStep == 0:
                         print("abandon value out is ",valueOut)
                    newAbandonNode.setV(valueOut)
                    self.selectedNodes.pop()
                    leafNodeChildNodes.append(newAbandonNode)
            #放弃大于等于2步，不添加任何节点
            else:
                pass
            if len(leafNodeChildNodes) != 0 :
                leafNode.setChildNodes(leafNodeChildNodes)
                if times % self.inforStep == 0:
                   print("Expand Current Leaf Node Child Nodes number is ", len(leafNodeChildNodes))
            else :
                if times % self.inforStep == 0:
                   print("Expand Current Leaf Node Child Nodes number is None .")


    #更新剩余节点的统计系数
    def backUp(self):
        reversedSelectNodes = list(reversed(self.selectedNodes))
        for n,node in enumerate(reversedSelectNodes):
            if n == 0 :
                node.setN(node.N + 1)
                node.setW(node.W + node.V)
                node.setQ(q=node.W / node.N + 0.0)
            else:
                node.setN(n=node.N + 1)
                node.setW(w=node.W + reversedSelectNodes[n - 1].W)
                node.setQ(q=node.W / node.N + 0.0)

    def tensorflowSessionClose(self):
        self.sess.close()


    #可能会输出None,当前根节点没有子节点的时候
    def outputChildNodesOfRootNode(self):
        return self.rootNode.childNodes

    def outputProbability(self):
        childNodesOfRootNode = self.rootNode.childNodes
        countPowInChildNodes = [math.pow(node.N,self.T)
                                for node in childNodesOfRootNode]
        sumCountsPow = sum(countPowInChildNodes)
        outputP = [countPow / sumCountsPow + 0.0
                   for countPow in countPowInChildNodes]
        return outputP


############################
#测试
if __name__ == "__main__" :

    MCTS = MCTS()
    start = time.time()
    for i in range(500):
        start = time.time()
        print(i)
        print("Select start . ")
        MCTS.select(i)
        print("Expand start")
        MCTS.expand(i)
        print("Back up start")
        MCTS.backUp()
        end= time.time()
        print("one MCTS search time is ",end - start)
    end = time.time()
    MCTS.tensorflowSessionClose()
    print("one step time is " , end - start)
    outputFirstP = MCTS.outputProbability()
    print(outputFirstP)
    print(len(outputFirstP))
















