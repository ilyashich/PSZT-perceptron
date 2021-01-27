import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston  # ladowanie danych
from sklearn.preprocessing import StandardScaler  # skalowanie danych
from statsmodels.tools.tools import add_constant  # dodanie kolumny jedynek do zbioru uczacego

'''
Reprezentacja sieci neuronowej
'''
class perceptron():

    def __init__(self, inp, target, iterations, rate):

        self.X = inp
        self.y = target
        self.n = inp.shape[0]
        self.p = inp.shape[1]

        self.weights = np.random.normal(loc=0.0, scale=(np.sqrt(2 / self.p)), size=self.p)
        self.l_rate = rate
        self.iterations = iterations

    def relu(self, x):
        if x >= 0:
            return x
        else:
            return 0

    def relu_derivative(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1

    def train(self):

        for epoch in range(self.iterations):
            for i, observation in enumerate(self.X):
                pa = np.sum(np.dot(self.weights, observation.T))
                y_pred = self.relu(pa)
                error = self.y[i] - y_pred
                # backprop
                self.weights += self.l_rate * error * observation * self.relu_derivative(pa)  # delta rule

    # przewidujemy wartosc
    def predict(self, x):
        return self.relu(np.sum(np.dot(self.weights, x.T)))


dataset = load_boston()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
y = dataset.target


# oddzielenie danych treningowych od testowych
def separate_data():
    A = boston_dataset[:406]
    tA = boston_dataset[406:]
    return A, tA


chas = df['CHAS'].values
df.drop(labels=['CHAS'], axis=1, inplace=True)

# skalowanie danych
scaler = StandardScaler()
scaler.fit(df.values)
X_train_stan = add_constant(scaler.transform(df.values))
X_train_stan = np.c_[X_train_stan, chas]

boston_dataset = np.column_stack((X_train_stan, y.T))
boston_dataset = list(boston_dataset)

# permutowanie wierszy danych
random.shuffle(boston_dataset)

# oddzielamy dane treningowe od testowych
Filetrain, Filetest = separate_data()
train_X = np.array([i[:14] for i in Filetrain])
train_y = np.array([i[14] for i in Filetrain])
test_X = np.array([i[:14] for i in Filetest])
test_y = np.array([i[14] for i in Filetest])

# ustawienie parametrow sieci
n_iterations = 500
l_rate = 0.05

# inicjacja sieci
p = perceptron(train_X, train_y, n_iterations, l_rate)

# obliczenie czasu treningu
start = time.time()
p.train()
end = time.time()
elapsed = end - start

print("Weights after training")
print(p.weights)


# obliczenie procentu trafien
residuals = []
y_predicted = []
count = 0
for i, el in enumerate(test_X):
    y_pred = p.predict(el)
    if 0.8 * test_y[i] <= y_pred <= 1.2 * test_y[i]:
        count = count + 1
    residuals.append(test_y[i] - y_pred)
    y_predicted.append(y_pred)

print("Percent of matches: ", count)
print("Time: ", elapsed)

# wykres bledow predykcji
plt.scatter(y_predicted, residuals)
plt.title("Residual plot of test set, n_iterations = " + str(n_iterations) + " " + "l_rate = " + str(l_rate))
plt.xlabel('Y predicted')
plt.ylabel('Residuals')
plt.show()
