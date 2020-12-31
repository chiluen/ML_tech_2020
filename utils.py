import numpy as np

def rmse(real, predict):
    rmse = 0
    real = np.array(real)
    for i in range(predict.shape[0]):
        rmse += (predict[i] - real[i])**2
    rmse /= predict.shape[0]
    return rmse