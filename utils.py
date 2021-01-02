import numpy as np

def rmse(real, predict):
    rmse = 0
    real = np.array(real)
    for i in range(predict.shape[0]):
        rmse += (predict[i] - real[i])**2
    rmse /= predict.shape[0]
    rmse = np.sqrt(rmse)

    return rmse

def binary_cls_error(real, predict):
    error = 0
    real = np.array(real)
    for i in range(predict.shape[0]):
        if real[i] != predict[i]:
            error += 1
    error /= predict.shape[0]
    return error
            
def mae(real, predict):
    error = 0
    real = np.array(real)
    for i in range(real.shape[0]):
        error += abs(real[i] - predict[i])
    error /= real.shape[0]
    return error