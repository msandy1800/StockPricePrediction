from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

def build_model():
    # instantiating knn model
    #using gridsearch to find the best parameter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    return model
