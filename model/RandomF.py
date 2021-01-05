from sklearn.ensemble import RandomForestRegressor

class model_RF_reg(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train(self, x_train, y_train):
        self.fit(x_train, y_train)

#max_depth=20, random_state=0, n_estimators=200