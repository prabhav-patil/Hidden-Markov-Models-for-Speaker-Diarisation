from hmmlearn import hmm
import numpy as np

class StickyHMM(hmm.GaussianHMM):
    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", n_iter=10, tol=1e-2, init_params="ste",
                 params="ste", covariance_type="diag", min_covar=None, random_state=None, verbose=False,
                 sticky_factor=0.99):
        super(StickyHMM, self).__init__(n_components, startprob_prior, transmat_prior,
                                         algorithm, n_iter, tol, init_params, params, covariance_type,
                                         min_covar, random_state, verbose)
        self.sticky_factor = sticky_factor
        
    def _generate_sample_from_state(self, state, random_state=None):
        return self._generate_sample_from_gaussian(self.means_[state], self.covars_[state], random_state)
    
    def _get_sticky_transmat(self):
        n_components = self.n_components
        transmat = (1.0 - self.sticky_factor) * self.transmat_ + self.sticky_factor / n_components
        for i in range(n_components):
            transmat[i, i] = 1.0 - self.sticky_factor + self.sticky_factor / n_components
        return transmat
    
    def _get_log_likelihood(self, obs):
        return np.array([self._compute_log_likelihood(obs[i:i+1])[0] for i in range(len(obs))])
    
    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["sticky"] = np.zeros((self.n_components, self.n_components))
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)
        n_samples, n_components = posteriors.shape
        sticky_posteriors = np.zeros((n_samples - 1, n_components, n_components))
        for t in range(n_samples - 1):
            alpha_t = fwdlattice[t, :]
            beta_t = bwdlattice[t+1, :]
            obs_t1 = obs[t+1:t+2]
            transmat = self._get_sticky_transmat()
            sticky_posteriors[t, :, :] = alpha_t[:, np.newaxis] * transmat * obs_t1 * beta_t
            sticky_posteriors[t, :, :] /= sticky_posteriors[t, :, :].sum()
        stats["sticky"] += sticky_posteriors.sum(axis=0)
        return stats
    
    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        sticky = stats["sticky"]
        self.transmat_ = (sticky / sticky.sum(axis=1)[:, np.newaxis])
    
    def fit(self, X, lengths=None):
        if lengths is None:
            lengths = [len(X)]
        super().fit(X, lengths)
    
    def predict(self, X, lengths=None):
        if lengths is None:
            lengths = [len(X)]
        obs_logprob = self._get_log_likelihood(X)
        transmat = self._get_sticky_transmat()
        _, state_sequence = super()._decode(obs_logprob, transmat)
        return state_sequence
   
    def sample(self, n_samples=1, random_state=None):
        random_state = random_state or self.random_state

        # Initialize the samples and state sequence arrays
        X = np.empty((n_samples, self.n_features))
        state_sequence = np.empty(n_samples, dtype=int)

        # Sample the initial state
        state = random_state.choice(self.n_components, p=self.startprob_)

        # Sample the first observation from the initial state
        X[0] = self._generate_sample_from_state(state, random_state=random_state)
        state_sequence[0] = state

        # Sample the remaining observations and state sequence
        for t in range(1, n_samples):
            # Sample the next state based on the sticky transition matrix
            transmat = self._get_sticky_transmat()
            state = random_state.choice(self.n_components, p=transmat[state])
            state_sequence[t] = state

            # Sample the next observation from the new state
            X[t] = self._generate_sample_from_state(state, random_state=random_state)

        return X, state_sequence
