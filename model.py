from sklearn.mixture import GaussianMixture
import numpy as np

class Model:
  FONTS_NO = 9
  def __init__(self):
    self.models = [GaussianMixture(n_components=10) for _ in range(Model.FONTS_NO)]

  def fit(self, X, y):
    self.models[y].fit(X)

  def predict(self, X):
    label = None
    global_likelihood = float("-inf")
    for idx, model in enumerate(self.models):
      local_likelihoods = model.score_samples(X)
      temp_global_likelihood = sum(local_likelihoods)

      if temp_global_likelihood > global_likelihood:
        label = idx
        global_likelihood = temp_global_likelihood
    return label

