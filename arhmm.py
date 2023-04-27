from hmmlearn.hmm import GaussianHMM

class ARHMM(GaussianHMM):
    def __init__(self, n_components=1, covariance_type='full', n_lags=1, **kwargs):
        super().__init__(n_components=n_components, covariance_type=covariance_type, **kwargs)
        self.n_lags = n_lags

    def _get_lagged_data(self, X):
        X_lagged = []
        for i in range(len(X) - self.n_lags):
            X_lagged.append(X[i:i + self.n_lags])
        return np.array(X_lagged)

    def fit(self, X, **kwargs):
        X_lagged = self._get_lagged_data(X)
        return super().fit(X_lagged, **kwargs)

    def sample(self, n_samples=1, random_state=None):
        X_lagged = super().sample(n_samples=n_samples, random_state=random_state)[0]
        return X_lagged[:, -1]
