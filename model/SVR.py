import numpy as np
from sklearn.svm import SVR

class model_SVR_reg(SVR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train(self, x_train, y_train):
        self.fit(x_train, y_train)


#C=1.0, epsilon=0.0, verbose=1
