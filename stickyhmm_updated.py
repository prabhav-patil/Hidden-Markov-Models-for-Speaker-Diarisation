import numpy as np
from hmmlearn import hmm

class StickyHMM:
    def __init__(self, n_components, startprob_prior, transmat_prior, \
                 observation_prior, kappa=10):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.observation_prior = observation_prior
        self.kappa = kappa
        
    def fit(self, X):
        self.model = hmm.GaussianHMM(n_components=self.n_components, \
                                     startprob_prior=self.startprob_prior, \
                                     transmat_prior=self.transmat_prior, \
                                     observation_prior=self.observation_prior, \
                                     covariance_type='diag', \
                                     params='mct', \
                                     init_params='mc')
        
        self.model.fit(X)
        
        A = self.model.transmat_
        B = self.model.means_
        C = self.model.covars_
        pi = self.model.startprob_
        
        self.sticky_A = (1 - self.kappa) * A + self.kappa * np.eye(self.n_components)
        self.sticky_B = B
        self.sticky_C = C
        self.sticky_pi = pi
        
    def sample(self, n_samples):
        return self.model.sample(n_samples=n_samples)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X):
        return self.model.score(X)
    
    def logprob(self, X):
        return self.model.score(X)
    
    def generate(self, n_samples):
        samples, _ = self.model.sample(n_samples=n_samples)
        return samples
    
    def generate_viterbi(self, n_samples):
        states = self.model.predict(X)
        samples = np.zeros((n_samples, self.model.n_features))
        
        for t in range(n_samples):
            samples[t] = self.model._generate_sample_from_state(states[t])
        
        return samples
