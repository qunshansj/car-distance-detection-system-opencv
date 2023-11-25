
class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1  # 最小的加权错误率
        self.best_fea_id = 0  # 最优特征id
        self.best_thres = 0  # 选定特征的最优阈值
        self.best_op = 1  # 阈值符号，其中 1: >, 0: <
 
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        n = X.shape[1]
        for i in range(n):
            feature = X[:, i]  # 选定特征列
            fea_unique = np.sort(np.unique(feature))  # 将所有特征值从小到大排序
            for j in range(len(fea_unique)-1):
                thres = (fea_unique[j] + fea_unique[j+1]) / 2  # 逐一设定可能阈值
                for op in (0, 1):
                    y_ = 2*(feature >= thres)-1 if op==1 else 2*(feature < thres)-1  # 判断何种符号为最优
                    err = np.sum((y_ != y)*sample_weight)
                    if err < self.best_err:  # 当前参数组合可以获得更低错误率，更新最优参数
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self
    
    def predict(self, X):
        feature = X[:, self.best_fea_id]
        return 2*(feature >= self.best_thres)-1 if self.best_op==1 else 2*(feature < self.best_thres)-1
    
    def score(self, X, y, sample_weight=None):
        y_pre = self.predict(X)
        if sample_weight is not None:
            return np.sum((y_pre == y)*sample_weight)
        return np.mean(y_pre == y)
