def gauss_my(X, means, covs, n_component):  # 自作のGauss関数
    n_size, ndim = np.shape(X)
    a = lambda i: np.sqrt(np.linalg.det(covs[:,:,i]).T*(2*np.pi)**ndim)
    b1 = lambda i: X-np.tile(means[:, i], (n_size,1)).reshape(-1, ndim)
    b2 = lambda i: np.linalg.inv(np.tile(covs[:,:,i], (n_size,1,1))).reshape(-1, ndim)
    b3 = lambda i: np.dot(b1(i), b2(i)).reshape(-1, ndim)
    b4 = lambda i: np.exp(-0.5 * np.diag(np.dot(b3(i), b1(i).T)))
    b = np.vstack(b4(i) / a(i) for i in range(n_component))
    return b.reshape(n_size, 1, n_component)


def gauss_or(X, means, covs, n_component):  # scipyによるGauss関数
    mn_lst = np.array([stats.multivariate_normal(mean=means[:, i], cov=covs[:,:,i]) for i in range(n_component)])
    return np.array([mn_lst[i].pdf(X) for i in range(n_component)])


class GaussianMixture(object):

    def __init__(self, n_component):
        # ガウス分布の個数
        self.n_component = n_component
    
    # EMアルゴリズムを用いた最尤推定
    # f_factor: 忘却率
    # s_factor: スムージング定数
    def fit(self, X, f_factor, s_factor):
        # データの次元
        self.ndim = np.size(X, 1)

        # 前処理
        self.pre_process(X)
    
        # EステップとMステップを繰り返す
        for i in range(np.shape(X)[0]):
            xt = X[i, :].reshape(1, -1)
            params = np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))
            # Eステップ、負担率を計算
            self.resps = self.expectation(xt)
            # Mステップ、パラメータを更新
            self.maximization(xt, f_factor, s_factor)
            
            print("Step:" + str(i) + " / resps:" + str(self.resps) + " / w:" + str(self.weights))
            
            # # パラメータが収束したかを確認
            # if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))):
            #     break
            # else:
            #     print("parameters may not have converged")

    # 前処理関数
    def pre_process(self, X):
        # 推定のための中間パラメータ値の初期化
        self.weights_f = np.ones(self.n_component) / self.n_component
        self.means_f = np.random.uniform(X.min(), X.max(), (self.ndim, self.n_component))
        self.covs_f = np.repeat(10 * np.eye(self.ndim), self.n_component).reshape(self.ndim, self.ndim, self.n_component)
        # 推定パラメータ値の算出        
        self.weights = (self.weights_f + s_factor) / (self.n_component * s_factor + self.weights_f.sum())  # 式(5.15)
        self.means = self.means_f / self.weights_f  # 式(5.10)
        self.covs = self.covs_f / self.weights_f - \
                np.array([np.dot(self.means[:, i].reshape(-1, 1), self.means[:,i].reshape(-1, 1).T) for i in range(self.n_component)]).T  # 式(5.12) 
        self.resps = self.weights
    
    # ガウス関数
    def gauss(self, X):  
        print("gauss func called...")
        # ガウス関数による分布確率の取得
        # ------ Choose one ------------
        val = gauss_my(X, self.means, self.covs, self.n_component)
#         val = gauss_or(X, self.means, self.covs, self.n_component)
        # ------ Choose one ------------
        return val

    # Eステップ
    def expectation(self, X):
        if (math.isnan(self.gauss(X).sum()) or self.gauss(X).sum() == 0):
            print("##### Gauss includes NaN !!!!!!!! #####")
            return self.resps
        else:
            self.resps = self.weights * self.gauss(X)
            self.resps /= self.resps.sum(axis=-1, keepdims=True)  # 式(5.7)
            print("gauss: " + str(self.gauss(X)))
            print("resps: " + str(self.resps))
            return self.resps

    # Mステップ
    def maximization(self, X, f_factor, s_factor):
        # 推定のための中間パラメータ値の更新
        self.weights_f = ((1 - f_factor) * self.weights + f_factor * self.resps).reshape(1, self.n_component)  # 式(5.14) (5.17)
        self.means_f = ((1 - f_factor) * self.means + f_factor * np.tile(X.T, (1, self.n_component)) * self.resps).reshape(self.ndim, -1)  # 式(5.18) 
        self.covs_f = (1 - f_factor) * self.covs + \
            f_factor * self.resps * np.tile(np.dot(X.T, X), (self.n_component, 1, 1)).T  # 式(5.19) 
        # 推定パラメータ値の更新
        self.weights = (self.weights_f + s_factor) / (self.n_component * s_factor + self.weights_f.sum())  # 式(5.15)
        self.means = self.means_f / self.weights_f  # 式(5.10)
        self.covs = (self.covs_f / self.weights_f) - \
            np.array([np.dot(self.means[:, i].reshape(-1, 1), self.means[:,i].reshape(-1, 1).T) for i in range(self.n_component)]).T  # 式(5.12) 

    # 確率分布p(x)を計算
    def predict_proba(self, X):
        gauss = self.weights.reshape(-1, 1) * self.gauss(X)
        return np.sum(gauss, axis=-1)

    # クラスタリング
    def classify(self, X):
        joint_prob = self.weights.reshape(-1, 1) * self.gauss(X)
        return np.argmax(joint_prob, axis=0)

