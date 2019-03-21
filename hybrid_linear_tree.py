import math
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator

class HybridRegressionTree(BaseEstimator):
    def __init__(self, min_node_size=10, mode='Ridge', min_split_improvement=0):
        self.min_node_size = min_node_size
        self.mode = mode
        self.min_split_improvement = min_split_improvement

    def fit(self, hrt_X, y):
        X = hrt_X.values
        self.root = Node.create_node(self, X, hrt_X, y)
        return self
        
    def hrt_predictions(self, X, y):
 
        if self.mode == 'Lasso':
            reg = Lasso()
            reg.fit(X, y)
            hrt = Model(reg)
        else:
            reg = Ridge()
            reg.fit(X, y)
            hrt = Model(reg)
            
        predictions = hrt.predict(X)
        if np.isnan(predictions).any():
            print('There are nan predictions')
            
        return predictions, hrt
     

    def predict(self, hrt_X):
        X = hrt_X.values 
        
        data = []
       
        for i in range(len(X)):
            data.append(self.predict_one(X[i, :], hrt_X.iloc[[i]]))
#             data.append(self.predict_one(X[i, :], X.iloc[[i]]))
        return np.array(data)

    def predict_one(self, X, hrt_X):
        return self.root.predict_one(X, hrt_X)

    # 2-d array of [node_id, prediction]
    def predict_full(self, X, hrt_X):
        data = []
        
        for i in range(len(X)):
            data.append(self.predict_full_one(X[i, :], hrt_X.iloc[[i]]))
#             data.append(self.predict_full_one(X[i, :], X.iloc[[i]]))
        return np.array(data)

    def predict_full_one(self, X, hrt_X):
        return self.root.predict_full_one(X, hrt_X)

    def node_count(self):
        return self.root.node_count()



class Node:
    def __init__(self, feature_idx, pivot_value, hrt):
        self.feature_idx = feature_idx
        self.pivot_value = pivot_value
        self.hrt = hrt
        self.row_count = 0
        self.left = None
        self.right = None
        
    @staticmethod
    def create_node(tree, X, hrt_X, y):

        (feature_idx, pivot_value, hrt, residuals) = Node.find_best_split(tree, X, hrt_X, y)
        node = Node(feature_idx, pivot_value, hrt)
        node.row_count = len(X)

        if feature_idx is not None:
            left_X, left_hrt_X, left_residuals, right_X, right_hrt_X, right_residuals = Node.split_on_pivot(
              X, hrt_X, residuals, feature_idx, pivot_value)
            node.left = Node.create_node(tree, left_X, left_hrt_X, left_residuals)
            node.right = Node.create_node(tree, right_X, right_hrt_X, right_residuals)
        return node

    def node_count(self):
        if self.feature_idx is not None:
            return 1 + self.left.node_count() + self.right.node_count()
        else:
            return 1

    def predict_one(self, X, hrt_X):
    
        local_value = self.hrt.predict(hrt_X)[0]
#         local_value = self.hrt.predict(X)[0]
        if self.feature_idx is not None:
            child_value = 0
            if X[self.feature_idx] < self.pivot_value:
                child_value = self.left.predict_one(X, hrt_X)
            else:
                child_value = self.right.predict_one(X, hrt_X)

            return child_value + local_value
        else:
            return local_value

    def predict_full_one(self, X, hrt_X):

        local_value = self.hrt.predict(hrt_X)[0]
#         local_value = self.hrt.predict(X)[0]
        if self.feature_idx is not None:
            result = None
            if X[self.feature_idx] < self.pivot_value:
                result = self.left.predict_full_one(X, hrt_X, 'L')
            else:
                result = self.right.predict_full_one(X, hrt_X, 'R')
            result[1] += local_value
            return result
        else:
            return np.array([local_value + self.hrt.predict(hrt_X)[0]])  # convert to 2-d array, then back

    @staticmethod
    def split_on_pivot(X, hrt_X, y, feature_idx, pivot_value):
        sorting_indices = X[:, feature_idx].argsort()  
        sorted_X = X[sorting_indices]
        pivot_idx = np.argmax(sorted_X[:, feature_idx] >= pivot_value)
        sorted_hrt_X = hrt_X.iloc[sorting_indices, :] 
        sorted_y = y[sorting_indices]

        return (sorted_X[:pivot_idx, :],
                sorted_hrt_X.iloc[:pivot_idx, :],
                sorted_y[:pivot_idx],
                sorted_X[pivot_idx:, :],
                sorted_hrt_X.iloc[pivot_idx:, :],
                sorted_y[pivot_idx:])

    @staticmethod
    def find_best_split(tree, X, hrt_X, y):
        
        predictions, hrt = tree.hrt_predictions(hrt_X, y)
        residuals = y - predictions
        n, m = X.shape
        sse = (residuals**2).sum()
        best_sse = sse
        best_feature = None
        best_feature_pivot = None

        for feature_idx in range(m):
            sorting_indices = X[:, feature_idx].argsort()  
            sorted_X = X[sorting_indices]
            sorted_resid = residuals[sorting_indices]
            sum_left = 0
            sum_right = residuals.sum()
            sum_squared_left = 0
            sum_squared_right = sse
            count_left = 0
            count_right = n
            pivot_idx = 0

            while count_right >= tree.min_node_size:

                row_y = sorted_resid[pivot_idx]
                sum_left += row_y
                sum_right -= row_y
                sum_squared_left += row_y*row_y
                sum_squared_right -= row_y*row_y
                count_left += 1
                count_right -= 1
                pivot_idx += 1

                if count_left >= tree.min_node_size and count_right >= tree.min_node_size:

                    rmse_left = math.sqrt((count_left * sum_squared_left) - (sum_left * sum_left)) / count_left
                    sse_left = rmse_left * rmse_left * count_left
                    rmse_right = math.sqrt((count_right * sum_squared_right) - (sum_right * sum_right)) / count_right
                    sse_right = rmse_right * rmse_right * count_right
                    split_sse = sse_left + sse_right

                    if (split_sse < best_sse and sse - split_sse > tree.min_split_improvement and
                          # only if the value is different than the last value
                          (count_left <= 1 or sorted_X[pivot_idx, feature_idx] != sorted_X[pivot_idx - 1, feature_idx])):
                        best_sse = split_sse
                        best_feature = feature_idx
                        best_feature_pivot = sorted_X[pivot_idx, feature_idx]

        return (best_feature, best_feature_pivot, hrt, residuals)
    
class Model:
    def __init__(self, reg):
        self.reg = reg
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_
        
    def predict(self, X):
        return self.reg.predict(X)
