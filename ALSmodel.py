import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class ALSRecommender(BaseEstimator, RegressorMixin):
	"""
	Baseline model that predicts user ratings to be the average
	of rating for that object in the training data. If an object
	does not exist in the training data, it predicts the rating
	to be the overall average rating in the training data.
	"""
	def __init__(self, user_col='userId', item_col='movieId', rating_col='rating', lambda_reg=1, n_factors=100, max_iter=100):
		"""
		@param id_col: column name of ID
		@param rating_col: column name of rating
		"""
		self.user_col = user_col
		self.item_col = item_col
		self.rating_col = rating_col
		self.lambda_reg = lambda_reg
		self.n_factors = n_factors
		self.max_iter = max_iter

		self.prepared = False
		self.fitted = False

	def prepare(self, train_data, test_data):
		"""
		Prepares the model for training by getting metadata and constructing the user-item matrix.
		Both the train data and test data are required to avoid error cases. The ratings information
		in the test data is not used. Only the user and item id's are used to construct the matrix.
		@param train_data: a pandas DataFrame of training data containing both the object id's and ratings 
		@param test_data: a pandas DataFrame of test data containing the object and user id's. Ratings are not need in this data.
		"""
		# get list of users and movies
		self.users = list(set(train_data[self.user_col]) | set(test_data[self.user_col]))
		self.items = list(set(train_data[self.item_col]) | set(test_data[self.item_col]))
		self.num_users = len(self.users)
		self.num_items = len(self.items)
		self.user2idx = {user: idx for (idx, user) in enumerate(self.users)}
		self.item2idx = {item: idx for (idx, item) in enumerate(self.items)}

		# construct matrix
		self.R_mat = np.zeros((self.num_users, self.num_items))
		for row in train_data.iterrows():
		    self.R_mat[self.user2idx[row[1][self.user_col]], self.item2idx[row[1][self.item_col]]] = row[1][self.rating_col]

		self.prepared = True
		return self

	def fit(self, num_iter=None):
		"""
		"Fits" the model to the training data by aggregating the movie rating averages 
		and storing the aggregated dataframe in an object variable.
		"""
		if not self.prepared:
			raise RuntimeError("You must prepare model before training.")

		self.user_vec = np.random.rand(self.num_users, self.n_factors) 
		self.item_vec = np.random.rand(self.n_factors, self.num_items)

		if not num_iter:
			num_iter = self.max_iter

		for i in range(num_iter):
		    self.user_vec = np.linalg.solve(np.dot(self.item_vec, self.item_vec.T) + self.lambda_reg * np.eye(self.n_factors), 
		                        np.dot(self.item_vec, self.R_mat.T)).T
		    self.item_vec = np.linalg.solve(np.dot(self.user_vec.T, self.user_vec) + self.lambda_reg * np.eye(self.n_factors),
		                        np.dot(self.user_vec.T, self.R_mat))
		    if i % 10 == 0:
		        print('{}th iteration'.format(i))
		
		self.fitted = True
		return self

	def predict_one(self, userId, itemId):
		"""
		Predicts user ratings for objects using baseline method by predicting the
		averages as stored in the class variable. Assumes same structure of input
		data as the fit method.
		@param X: a pandas DataFrame object containing both the object id's and ratings
		@returns a numpy-array containing the predicted ratings in the same order as input
		"""
		if not self.fitted:
			raise RuntimeError("You must train model before predicting data.")

		userIdx = self.user2idx[userId]
		itemIdx = self.item2idx[itemId]

		return self.user_vec[userIdx,:].dot(self.item_vec[:,itemIdx])

	def predict(self, test_data):
		if not self.fitted:
			raise RuntimeError("You must train model before predicting data.")

		pred = []
		for row in test_data.iterrows():
			pred.append(self.predict_one(row[1][self.user_col], row[1][self.item_col]))
		return pred

	def score(self, X):
		"""
		Returns the score of performance on the input test data X. Assumes the 
		same structure as fit and predict. The metric used is RMSE.
		@param X: a pandas DataFrame object containing both the object id's and ratings
		@returns a float, the RMSE
		"""
		preds = self.predict(X)
		return np.sqrt(mean_squared_error(X[self.rating_col].as_matrix(), preds))
		
