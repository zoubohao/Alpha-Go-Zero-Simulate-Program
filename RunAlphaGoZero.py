from Training import Training

class RunAlphaGoZero:

    #path为保存数据的文件位置
    #如果为None，则初始化权重，不然使用保存的文件里面的权重
    def __init__(self,path = None):

        if path is None:
            self.training = Training()
            self.selfPlay = self.training.selfPlay
        else:
            self.training = Training(path=path)
            self.selfPlay = self.training.selfPlay

    def OnceSelfPlay(self , iteration = None):
        print("SELFPLAY START .")
        #iter是每一步中MCTS需要搜索的步数，默认300
        #Vresign是终止一盘棋局的最小阈值
        #当节点的V值小于它时，放弃对弈，认输
        if iteration is not None:
            self.selfPlay.selfPlayStart(iteration=iteration)
        else:
            self.selfPlay.selfPlayStart()
        inputData, possibilityData, valueData = self.selfPlay.allBoardStepDataReturn()
        self.selfPlay.clearAllStepData()
        print("SELF PLAY END . ")
        return inputData,possibilityData,valueData


    def Training(self,inputData,possibilityData,valueData,
                 holisticDataTrainingTimes,batch_size):
        print("TRAINING START . ")
        self.training.setTrainingTimes(holisticDataTrainingTimes)
        self.training.training(inputData=inputData,
                               probability=possibilityData,
                               value=valueData,batch_size=batch_size)
        print("TRAINING END .")


    def TensorflowSessionClose(self):
        self.training.closeSession()


    def TrainingWeightSave(self,path):
        #保存文件的路径为"e:\model.ckpt"
        self.training.weightsSave(path=path)






