from RunAlphaGoZero import RunAlphaGoZero


Run = RunAlphaGoZero()
for i in range(5):
    inputData, possibilityData, valueData = Run.OnceSelfPlay(iteration=1600)
    Run.Training(inputData, possibilityData, valueData, 5,batch_size=16)
Run.TrainingWeightSave(path="D:\AlphaGoWeight\modelTrainingFourTimes.ckpt")
Run.TensorflowSessionClose()

