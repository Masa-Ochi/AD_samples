class GaussianMixture(object):

    def __init__(self, n_component, f_factor=0.5):
        # ガウス分布の個数
        n_component = n_component
        # 忘却率
        f_factor = f_factor


    # EMアルゴリズムを用いた最尤推定
    def fit(self, X):
        # データの次元
        ndim = np.size(X, 1)
        # 混合係数の初期化
        weights = np.ones(n_component) / n_component
        # 平均の初期化
        means = np.random.uniform(X.min(), X.max(), (ndim, n_component))
        # 共分散行列の初期化
        covs = np.repeat(10 * np.eye(ndim), n_component).reshape(ndim, ndim, n_component)

        # EステップとMステップを繰り返す
        for i in range(np.shape(X)[0]):
            xt = X[i, :].reshape(2,1)
            params = np.hstack((weights.ravel(), means.ravel(), covs.ravel()))
            # Eステップ、負担率を計算
            resps = expectation(xt)
            # Mステップ、パラメータを更新
            maximization(xt, resps)
            # # パラメータが収束したかを確認
            # if np.allclose(params, np.hstack((weights.ravel(), means.ravel(), covs.ravel()))):
            #     break
            # else:
            #     print("parameters may not have converged")

    # ガウス関数
    def gauss(self, X):

        # precisions = np.linalg.inv(covs.T).T
        # diffs = X - means
        # assert diffs.shape == (len(X), ndim, n_component)
        # exponents = np.sum(np.einsum('nik,ijk->njk', diffs, precisions) * diffs, axis=1)
        # assert exponents.shape == (len(X), n_component)
        # return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(covs.T).T * (2 * np.pi) ** ndim)
        a = np.hstack(np.linalg.det(covs[:,:,i]).T*(2*np.pi)**ndim for i in range(np.shape(covs)[2]))
        b = np.sqrt(a)
        c = np.hstack(np.linalg.det(-0.5*(X.T-means[:, i]).T*np.linalg.inv(covs[:,:,i])*(X.T-means[:, i])) for i in range(np.shape(covs)[2]))
        return np.exp(c)/b

    # Eステップ
    def expectation(self, X):
        # 式(5.7)
        resps = weights * gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    # Mステップ
    def maximization(self, X, resps):

        # 式(5.17)
        weights = (1 - f_factor) * weights + f_factor * resps

        # 式(5.20)
        means = (1 - f_factor) * means + f_factor * np.dot(X.T, resps)
        covs = (1 - f_factor) * covs + \
            f_factor * np.tile(resps, (ndim, ndim, 1)) * np.tile(np.dot(X.T, X), (n_component, 1, 1)).T

    # 確率分布p(x)を計算
    def predict_proba(self, X):
        # PRML式(9.7)
        gauss = weights * gauss(X)
        return np.sum(gauss, axis=-1)

    # クラスタリング
    def classify(self, X):
        joint_prob = weights * gauss(X)
        return np.argmax(joint_prob, axis=1)


def create_toy_data(n_size):
    pi = np.array([0.5, 0.2, 0.3])
    mus = [[1, 1], [-1, -1], [0, 0]]
    stds = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((n_size, 2), dtype=np.float32)
    for n in range(n_size):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
    return x


def main():
    X = create_toy_data(100) # Xは時系列で昇順

    model = GaussianMixture(3)
    model.fit(X)
    labels = model.classify(X)

    x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
    probs = model.predict_proba(X_test)
    Probs = probs.reshape(100, 100)
    colors = ["red", "blue", "green"]
    plt.scatter(X[:, 0], X[:, 1], c=[colors[int(label)] for label in labels])
    plt.contour(x_test, y_test, Probs)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


# if __name__ == '__main__':
main()
