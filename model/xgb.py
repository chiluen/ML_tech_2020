import xgboost as xgb

class model_xgb_reg(xgb.XGBRegressor):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train):
        self.fit(x_train, y_train)

class model_xgb_cls(xgb.XGBClassifier):
    def __init__(self):
        super().__init__()
    def train(self, x_train, y_train):
        self.fit(x_train, y_train)
