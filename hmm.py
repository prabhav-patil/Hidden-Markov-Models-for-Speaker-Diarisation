from hmmlearn import hmm
import numpy as np

class HMM:
    def __init__(self, num_states, num_obs, A=None, B=None, pi=None):
        self.num_states = num_states
        self.num_obs = num_obs
        
        # Initialize transition matrix A, emission matrix B, and initial state distribution pi
        if A is not None:
            self.A = np.array(A)
        else:
            self.A = np.random.rand(num_states, num_states)
            self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        if B is not None:
            self.B = np.array(B)
        else:
            self.B = np.random.rand(num_states, num_obs)
            self.B /= np.sum(self.B, axis=1, keepdims=True)
        
        if pi is not None:
            self.pi = np.array(pi)
        else:
            self.pi = np.random.rand(num_states)
            self.pi /= np.sum(self.pi)
        
        self.model = hmm.MultinomialHMM(n_components=num_states)
        self.model.startprob_ = self.pi
        self.model.transmat_ = self.A
        self.model.emissionprob_ = self.B
    
    def fit(self, obs):
        obs = np.array(obs)
        lengths = [len(ob) for ob in obs]
        self.model.fit(obs, lengths=lengths)
    
    def predict(self, obs):
        obs = np.array(obs)
        return self.model.predict(obs)
    
    def score(self, obs):
        obs = np.array(obs)
        lengths = [len(ob) for ob in obs]
        return self.model.score(obs, lengths=lengths)
