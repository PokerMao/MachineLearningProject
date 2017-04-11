import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class BaselineRecommender(BaseEstimator, RegressorMixin):
	"""
	Baseline model that predicts user ratings to be the average
	of rating for that object in the training data. If an object
	does not exist in the training data, it predicts the rating
	to be the overall average rating in the training data.
	"""
	def __init__(self, id_col='movieId', rating_col='rating'):
		"""
		@param id_col: column name of ID
		@param rating_col: column name of rating
		"""
		self.rating_count = pd.DataFrame()
		self.all_avg = 0.
		self.id_col = id_col
		self.rating_col = rating_col
		self.fitted = False

	def fit(self, X):
		"""
		"Fits" the model to the training data by aggregating the movie rating averages 
		and storing the aggregated dataframe in an object variable.
		@param X: a pandas DataFrame object containing both the object id's and ratings
		"""
		self.rating_count = X[[self.id_col,self.rating_col]].groupby(by=self.id_col).mean()
		self.all_avg = float(X[self.rating_col].mean())
		self.fitted = True
		return self

	def predict(self, X):
		"""
		Predicts user ratings for objects using baseline method by predicting the
		averages as stored in the class variable. Assumes same structure of input
		data as the fit method.
		@param X: a pandas DataFrame object containing both the object id's and ratings
		@returns a numpy-array containing the predicted ratings in the same order as input
		"""
		if not self.fitted:
			raise RuntimeError("You must train model before predicting data.")

		preds = [float(self.rating_count.ix[record[1][self.id_col]]) if record[1][self.id_col] in self.rating_count.index else self.all_avg for record in X.iterrows()]
		return np.array(preds)

	def score(self, X):
		"""
		Returns the score of performance on the input test data X. Assumes the 
		same structure as fit and predict. The metric used is RMSE.
		@param X: a pandas DataFrame object containing both the object id's and ratings
		@returns a float, the RMSE
		"""
		preds = self.predict(X)
		return np.sqrt(mean_squared_error(X[self.rating_col].as_matrix(), preds))
		
