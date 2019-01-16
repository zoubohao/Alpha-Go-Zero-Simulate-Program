from LeelaZeroNet import LeelaZeroNet
import numpy as np
import tensorflow as tf
from Net import Net


net = Net()
policyNet , valueNet = net.policyNet , net.valueNet
loss = net.lossT
inputData , probData , valueData = net.inputPlaceHolder , net.porbPlaceHolder , net.valuePlaceHolder
training = net.training
lossP , lossV = net.lossP,net.lossV
opt = net.opt
lR = net.learningRate
sess = tf.Session()

inputD = []
prob = []
value = []

with open("d:/2.0","r") as f:
    lines = f.readlines()
    count = 0
    inputBoard = []
    for line in lines:
        line = line.strip("\n")
        if count < 16:
            boardList = list(line)
            board = ""
            length = len(boardList)
            for n,ele in enumerate(boardList):
                if n != length - 1:
                    binDis = bin(int(str(ele).upper(), 16))
                    binDis = list(binDis)
                    if len(binDis) < 6:
                        residual = 6 - len(binDis)
                        for i in range(residual):
                            binDis.insert(2, "0")
                    binDis.pop(0)
                    binDis.pop(0)
                    convert = "".join(binDis)
                    board = board + convert
                else:
                    board = board + ele
            position = list(board)
            position = [float(num) for num in position]
            position = np.array(position)
            position = np.reshape(position,newshape=[19,19])
            inputBoard.append(position)
            count = count  + 1
        elif count == 16:
            if line == "1":
                #zeros = np.zeros([19,19])
                position = np.ones(shape=[19, 19])
                #inputBoard.append(zeros)
                inputBoard.append(position)
            else:
                #ones = np.ones([19,19])
                position = np.zeros(shape=[19, 19])
                #inputBoard.append(ones)
                inputBoard.append(position)
            inputD.append(inputBoard)
            inputBoard = []
            count = count + 1
        elif count == 17:
            possi = line.split(" ")
            possi = [float(elem) for elem in possi]
            prob.append(possi)
            count = count + 1
        elif count == 18:
            value.append(float(line))
            count = 0


batch_size = 16
realInput = []
realProb = []
realValue = []
for n, inputOneData in enumerate(inputD):
    if n % batch_size == 0 and n != 0:
        meidumI = []
        for i in range(batch_size):
            meidumI.append(inputD[n-1-i])
        meidumI = np.reshape(meidumI,[batch_size,17,19,19])
        realInput.append(meidumI)
        mediumP = []
        for i in range(batch_size):
            mediumP.append(prob[n-1-i])
        mediumP = np.reshape(mediumP,[batch_size,19*19+1])
        realProb.append(mediumP)
        mediumV = []
        for i in range(batch_size):
            mediumV.append(value[n-1-i])
        mediumV = np.reshape(mediumV,[batch_size,1])
        realValue.append(mediumV)

sess.run(tf.global_variables_initializer())
total = 100 * len(realInput) * 5
print("Total training times are ",total)
trainingTime = 0
learningRateSchdule = [0.0125,0.01,0.0075]
# 一系列棋谱数据训练次数
for i in range(100):
    # 在每一个棋谱上训练,除去初始的空棋盘
    # 从1开始
    for j in range(1, len(realInput)):
        # 在每一步上训练的次数
        for n in range(5):

            if trainingTime <= total / 3.0:
                learningRate = learningRateSchdule[0]
            elif total / 3.0 < trainingTime <= 2 * total / 3.0:
                learningRate = learningRateSchdule[1]
            else:
                learningRate = learningRateSchdule[2]

            #sess.run(optP, feed_dict={inputData: inputD[j],
            #                          probData : prob[j],
            #                          lR: learningRate,
            #                          training:True})

            #sess.run(optV, feed_dict={inputData: inputD[j],
            #                         valueData: value[j],
            #                          lR: learningRate,
            #                         training:True})
            #print("probInput datas are ",probInputData)
            sess.run(opt, feed_dict={inputData: realInput[j],
                                    probData: realProb[j],
                                    valueData: realValue[j],
                                    lR: learningRate,
                                    training:True})

            if trainingTime % 50 == 0:
                print("Have trained " + str(trainingTime) + "times .")
                #print("policy number is ",probInputData)
                print("value number is ", realValue[j])
                print("value output is ",sess.run(valueNet,feed_dict={inputData: realInput[j],
                                    probData: realProb[j],
                                    valueData: realValue[j],
                                    lR: learningRate,
                                    training:False}))
                print("Current total loss is :",sess.run(loss, feed_dict={inputData: realInput[j],
                                    probData: realProb[j],
                                    valueData: realValue[j],
                                    lR: learningRate,
                                    training:False}))

                print("Current P loss is :", sess.run(lossP, feed_dict={inputData: realInput[j],
                                    probData: realProb[j],
                                    valueData: realValue[j],
                                    lR: learningRate,
                                    training:False}))

                print("Current V loss is :", sess.run(lossV, feed_dict={inputData: realInput[j],
                                    probData: realProb[j],
                                    valueData: realValue[j],
                                    lR: learningRate,
                                    training:False}))
            trainingTime = trainingTime + 1

saver = tf.train.Saver()
saver.save(sess,save_path="e:\modelTrainingFourTimes.ckpt")