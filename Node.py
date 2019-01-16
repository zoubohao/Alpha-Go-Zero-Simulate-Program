

class Node :

    #包含一个状态和所有合理的边，合理的，合理的，合理的！！！符合规则
    def __init__(self,state):
        self.state = state
        self.parentNode = None
        self.childNodes = None
        self.N = 0.0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0
        self.V = 0.0
        #节点的位置信息
        self.position = []
        self.selected = "no"


    def setP(self,p):
        self.P = p

    def setW(self,w):
        self.W = w

    def setQ(self,q):
        self.Q = q

    def setN(self,n):
        self.N = n

    def setV(self,v):
        self.V = v

    def setParentNode(self,parentNode):
        self.parentNode = parentNode

    def setChildNodes(self,childNodes):
        self.childNodes = childNodes

    def setState(self,state):
        self.state = state

    def setPosition(self,position):
        self.position = position

    def setSelected(self,selected):
        self.selected = selected








