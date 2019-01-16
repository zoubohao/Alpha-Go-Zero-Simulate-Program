from SelfPaly import SelfPlay
import random
import tensorflow as tf
import  numpy as np



class Training :

    def __init__(self,path = None):
        self.trainingTimes = 20
        if path is None :
            self.selfPlay = SelfPlay()
        else:
            self.selfPlay = SelfPlay(path = path)
        # 数据的输入和输出#
        self.netInput = self.selfPlay.inputData
        self.policyOutput = self.selfPlay.policyNet
        self.valueOutput = self.selfPlay.valueNet
        self.probabilityData = self.selfPlay.probabilityData
        self.valueData = self.selfPlay.valueData
        self.learningRate = self.selfPlay.learningRate
        self.lossT ,self.lossP ,self.lossV= self.selfPlay.lossT,self.selfPlay.lossP , self.selfPlay.lossV
        self.sess = self.selfPlay.sess
        self.optT = self.selfPlay.optT
        self.trainingAble = self.selfPlay.training

    def setTrainingTimes(self, times):
        self.trainingTimes = times

    def training(self, inputData, probability, value,batch_size):
        realInput = []
        realProb = []
        realValue = []
        for n, inputOneData in enumerate(inputData):
            if n % batch_size == 0 and n != 0:
                meidumI = []
                for i in range(batch_size):
                    meidumI.append(inputData[n - 1 - i])
                meidumI = np.reshape(meidumI, [batch_size, 17, 19, 19])
                realInput.append(meidumI)
                mediumP = []
                for i in range(batch_size):
                    mediumP.append(probability[n - 1 - i])
                mediumP = np.reshape(mediumP, [batch_size, 19 * 19 + 1])
                realProb.append(mediumP)
                mediumV = []
                for i in range(batch_size):
                    mediumV.append(value[n - 1 - i])
                mediumV = np.reshape(mediumV, [batch_size, 1])
                realValue.append(mediumV)
        learningRateSchdule = [0.0125,0.01,0.0085]
        trainingTime = 0
        total = self.trainingTimes * 20 * len(realInput)
        print("Gross training times are :",total)
        #一系列棋谱数据训练次数
        for i in range(self.trainingTimes):
            # 在每一个棋谱上训练,除去初始的空棋盘
            #从1开始
            for j in range(1,len(realInput)):
                #在每一步上训练的次数
                for n in range(20):
                    if trainingTime <= total / 3.0:
                        learningRate = learningRateSchdule[0]
                    elif total / 3.0 < trainingTime  <= 2 * total / 3.0:
                        learningRate = learningRateSchdule[1]
                    else:
                        learningRate = learningRateSchdule[2]

                    self.sess.run(self.optT,feed_dict={self.netInput : realInput[j],
                                                      self.probabilityData :realProb[j],
                                                      self.valueData : realValue[j],
                                                      self.learningRate : learningRate,
                                                       self.trainingAble:True})

                    if trainingTime % 100 == 0 :
                        print("Have trained " + str(trainingTime) + "times .")
                        print("value output is ", self.sess.run(self.valueOutput, feed_dict={self.netInput: realInput[j],
                                                                                self.probabilityData: realProb[j],
                                                                                valueData: realValue[j],
                                                                                self.learningRate: learningRate,
                                                                                self.trainingAble: False}))
                        print("value number is ", realValue[j])
                        print("Current total loss is :", self.sess.run(self.lossT,  feed_dict={self.netInput: realInput[j],
                                                                                self.probabilityData: realProb[j],
                                                                                valueData: realValue[j],
                                                                                self.learningRate: learningRate,
                                                                                self.trainingAble: False}))

                        print("Current P loss is :", self.sess.run(self.lossP,  feed_dict={self.netInput: realInput[j],
                                                                                self.probabilityData: realProb[j],
                                                                                valueData: realValue[j],
                                                                                self.learningRate: learningRate,
                                                                                self.trainingAble: False}))

                        print("Current V loss is :", self.sess.run(self.lossV,  feed_dict={self.netInput: realInput[j],
                                                                                self.probabilityData: realProb[j],
                                                                                valueData: realValue[j],
                                                                                self.learningRate: learningRate,
                                                                                self.trainingAble: False}))
                    trainingTime = trainingTime + 1

    def weightsSave(self,path):
        saver = tf.train.Saver()
        saver.save(sess=self.sess,save_path=path)

    def closeSession(self):
        self.sess.close()


############################
##测试
if __name__ == "__main__" :
    selfPlayAndTraining = Training()
    selfPlay = selfPlayAndTraining.selfPlay
    inputData = [selfPlay.inputDataConstruction(),selfPlay.inputDataConstruction(),selfPlay.inputDataConstruction()]
    possiData = [[random.random()*5.0 for i in range(19*19+1)],[random.random()*5.0 for i in range(19*19+1)]
        ,[random.random()*5.0 for i in range(19*19+1)]]
    valueData = [1,-1,1]
    selfPlayAndTraining.training(inputData,possiData,valueData)
    selfPlayAndTraining.weightsSave(path="e:\model.ckpt")
    training = SelfPlay(path="e:\model.ckpt")













