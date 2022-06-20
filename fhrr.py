import torch
import onlinehd
import numpy as np
import sklearn

class FHRR_Model:
    def __init__(self, classes, features, dim):
        self.classes = classes
        self.features = features
        self.dim = dim
        self.model = np.zeros([classes, dim], dtype = 'complex_')
        self.X = []
        self.Y = []
        
        self.feature_hyperv = np.empty([features, dim], dtype = 'complex_')
        self.init_hyperv()
        
    def init_hyperv(self):
        for k in range(self.features):
            #angles = np.random.uniform(-np.pi, np.pi, self.dim)
            angles = np.random.normal(0, 1, self.dim)
            for i in range(self.dim):
                self.feature_hyperv[k, i] = angles[i]
        
                
    def encode(self, x):
        # Encode MNIST image into FHRR HD
        encoded = np.zeros(self.dim)
        x = x.numpy()
        for i in range(self.features):
            encoded =+ np.exp(1j*self.feature_hyperv[i] * x[i])  
        return encoded

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
    
    def train(self):
        classes = np.unique(self.Y)
        for c in classes:
            inds = np.where(self.Y == c)[0]
            for i in inds:
                self.model[int(c)] += self.X[i]
        print("Training class hypervectors...\n", self.model)

    def retrain(self, epochs):
        for e in range(epochs):
            for i in range(len(self.Y)):
                vals = [self.compute_real_similarity(self.X[i], class_hyper) for class_hyper in self.model]
                class_index = np.argmax(vals)
                
                if class_index != self.Y[i]:
                    self.model[class_index] -= self.X[i]
                    self.model[int(self.Y[i])] += self.X[i]
                    
    def predict(self, x):
        vals = [self.compute_real_similarity(x, class_hyper) for class_hyper in self.model]
        class_index = np.argmax(vals)
        return class_index
    
    
    