import torch
import onlinehd
import numpy as np
import sklearn

class FHRR_Model:
    def __init__(self, classes, features, dim, pw='uniform'):
        self.classes = classes
        self.features = features
        self.dim = dim
        self.model = np.zeros([classes, dim], dtype = 'complex_')
        self.bundled_model = np.zeros([classes, dim], dtype = 'complex_')
        self.X = []
        self.Y = []
        self.pw = pw
        self.feature_hyperv = np.empty([features, dim])
        self.init_hyperv()
        
    def init_hyperv(self):
        for k in range(self.features):
            if self.pw == 'uniform':
                angles = np.random.uniform(-np.pi, np.pi, self.dim)
            elif self.pw == 'rbf':
                angles = np.random.normal(0, 1, self.dim)
            else:
                angles = np.random.uniform(-np.pi, np.pi, self.dim)
                self.feature_hyperv[k] = angles

    def encode(self, x):
        # Encode MNIST image into FHRR HD
        encoded = np.zeros(self.dim, dtype = 'complex_')
        x = x.numpy()
        for k in range(self.features):
            theta = self.feature_hyperv[k]
            exp = np.exp(1j* theta * x[k], dtype = 'complex_')
            encoded += exp  
        return self.create_complex_num(np.angle(encoded))

    def encode_set(self, X, Y):
        self.Y = Y
        for x in X:
            self.X.append(self.encode(x))
        self.X = np.array(self.X, dtype = 'complex_')
        print("X train shape", self.X.shape)

    def test_encode(self, X):
        X_encoded = []
        for x in X:
            X_encoded.append(self.encode(x))
        X_encoded = np.array(X_encoded, dtype = 'complex_')
        return X_encoded
    
    def set_train_set(self, X, Y):
        self.X = X
        self.Y = Y

    def create_complex_num(self, angle):
        return np.exp(1j*angle)
    
    def get_angle(self, x):
        return np.angle(x)
    
    def get_module(self, x):
        return abs(x)
    
    def compute_real_similarity(self, a, b):
        sim = np.vdot(a, b)
        return np.real(sim)
    
    def get_encoded_set(self):
        return self.X
    
    """def train(self):
        classes = np.unique(self.Y)
        for c in classes:
            inds = np.where(self.Y == c)[0]
            for i in inds:
                self.model[int(c)] += self.X[i]
            self.model[int(c)] = self.create_complex_num(np.angle(self.model[int(c)]))

    def retrain(self, epochs):
        for e in range(epochs):
            for i in range(len(self.Y)):
                vals = [self.compute_real_similarity(self.X[i], class_hyper) for class_hyper in self.model]
                class_index = np.argmax(vals)
                if class_index != self.Y[i]:
                    self.model[class_index] = self.create_complex_num(np.angle(self.model[class_index]) - np.angle(self.X[i]))
                    self.model[int(self.Y[i])] = self.create_complex_num(np.angle(self.model[class_index] + self.X[i]))
    """
    # """
    def train(self):
        classes = np.unique(self.Y)
        for c in classes:
            inds = np.where(self.Y == c)[0]
            for i in inds:
                self.bundled_model[int(c)] += self.X[i]
            self.model[int(c)] = self.create_complex_num(np.angle(self.bundled_model[int(c)]))
        print(self.bundled_model)

    def retrain(self, epochs, lr=0.035):
        for e in range(epochs):
            for i in range(len(self.Y)):
                vals = [self.compute_real_similarity(self.X[i], class_hyper) for class_hyper in self.model]
                class_index = np.argmax(vals)
                if class_index != self.Y[i]:
                    self.model[class_index] = self.create_complex_num(np.angle(self.bundled_model[class_index] - self.X[i]))
                    self.model[int(self.Y[i])] = self.create_complex_num(np.angle(self.bundled_model[class_index] + self.X[i]))
                    
    
    # """
    def predict(self, x):
        vals = [self.compute_real_similarity(x, class_hyper) for class_hyper in self.bundled_model]
        class_index = np.argmax(vals)
        return class_index
    
    