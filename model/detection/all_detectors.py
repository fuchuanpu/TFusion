import joblib
import numpy as np

from abc import ABC, abstractmethod
class BaseDetector(ABC):
    def __init__(self, ):
        super().__init__()
        self.model = None
        self.model_name = None

    def load(self, path:str) -> None:
        self.model = joblib.load(f'{path}_{self.model_name}.pkl')
        
    def save(self, path:str) -> None:
        joblib.dump(self.model, f'{path}_{self.model_name}.pkl')
        
    @abstractmethod
    def train(self, data, label) -> None:
        pass

    @abstractmethod
    def test(self, data, label) -> None:
        pass


class DetectorFactory:
    @staticmethod
    def create(model_name:str) -> BaseDetector:
        if model_name == 'kmeans':
            return KMeansDetector()
        elif model_name == 'iforest':
            return IForestDetector()
        elif model_name == 'ocsvm':
            return OneClassSVMDetector()
        elif model_name == 'rforest':
            return RandomForestDetector()
        elif model_name == 'knn':
            return KNNDetector()
        elif model_name == 'bayes':
            return BayesDetector()
        elif model_name == 'logistic':
            return LogisticDetector()
        else:
            raise ValueError(f'Invalid model name: {model_name}')


from sklearn.cluster import KMeans
class KMeansDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'kmeans'
        self.model = KMeans(n_clusters=10)

    def train(self, data, label) -> None:
        self.model.fit(data[label == 0])

    def test(self, data) -> None:
        center = self.model.cluster_centers_
        loss_res = [min([np.linalg.norm(ve - c) for c in center]) for ve in data]
        return loss_res


from sklearn.ensemble import IsolationForest
class IForestDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'iforest'
        self.model = IsolationForest(random_state=42)

    def train(self, data, label) -> None:
        self.model.fit(data[label == 0])

    def test(self, data) -> None:
        loss_res = self.model.score_samples(data)
        return -loss_res 


from sklearn.svm import OneClassSVM
class OneClassSVMDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'ocsvm'
        self.model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)

    def train(self, data, label) -> None:
        self.model.fit(data[label == 0])

    def test(self, data) -> None:
        loss_res = self.model.decision_function(data)
        return -loss_res


from sklearn.ensemble import RandomForestClassifier
class RandomForestDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'rforest'
        self.model = RandomForestClassifier(n_estimators=20, max_depth=50)

    def train(self, data, label) -> None:
        self.model.fit(data, label)

    def test(self, data) -> None:
        loss_res = self.model.predict(data)
        return loss_res


from sklearn.neighbors import KNeighborsClassifier
class KNNDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'knn'
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, data, label) -> None:
        self.model.fit(data, label)

    def test(self, data) -> None:
        loss_res = self.model.predict(data)
        return loss_res
    

from sklearn.naive_bayes  import GaussianNB
class BayesDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'bayes'
        self.model = GaussianNB()

    def train(self, data, label) -> None:
        self.model.fit(data, label)

    def test(self, data) -> None:
        loss_res = self.model.predict(data)
        return loss_res


from sklearn.linear_model import LogisticRegression
class LogisticDetector(BaseDetector):
    def __init__(self, ):
        super().__init__()
        self.model_name = 'logistic'
        self.model = LogisticRegression()

    def train(self, data, label) -> None:
        self.model.fit(data, label)

    def test(self, data) -> None:
        loss_res = self.model.predict(data)
        return loss_res
