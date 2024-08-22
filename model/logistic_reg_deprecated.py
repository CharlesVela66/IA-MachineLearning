import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Cargar el data frame limpio
df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

__errors__ = []

def h(params, sample):
    acum = 0
    for i in range(len(params)):
        acum += params[i] * sample[i]
    acum = -acum
    acum = 1 / (1 + np.exp(acum))
    return acum

def GD(params, samples, lr, y):
    n = len(samples)
    new_params = params[:-1]
    bias = params[-1]
    # Creamos lista del tamaño de de new_params
    temp = list(new_params)
    n = len(samples)
    # Hacemos la funcion gradiente descendiente por cada parametro que tenemos
    for j in range(len(new_params)):
        # Variable para llevar la sumatoria
        acum=0
        # Hacemos la sumatoria por cada x que tenemos
        for i in range(n):
            # Llamamos la funcion h con el parametro de samples[i] para sacar nuetra hipotesis y para ello necesitamos los valores de x1 y x2
            hyp = h(new_params, samples[i]) + bias
            #Error en prediccion. Comparamos la prediccion con el valor real y multiplicamos por la x correspondiente
            # Agarramos i j para fijar la columna con el valor de j, porque los valores de una columna representan los valores de la misma sample
            acum += (hyp-y[i]) * samples[i][j]
        # Agregamos a la lista el valor final de la funcion gradiente descendiente del parametro(theta) correspondiente
        temp[j] = params[j] -((lr/n) * acum)
        #print(temp[j])
    # Agregamos la funcion gradiente descendiente del bias a nuestra lista
    temp.append(GD_B(new_params, samples, lr, y, bias))
    return temp

def GD_B(params,samples, lr, y, bias):
    n = len(samples)
    acum = 0
    for i in range(n):
        hyp = h(params, samples[i]) + bias
        acum += (hyp-y[i])
    res = bias - ((lr/n) * acum)
    #print(res)
    return res

def cross_entropy(params, samples, y):
    bias = params[-1]
    new_params = params[:-1]
    predictions = np.array([h(new_params, sample) + bias for sample in samples])

    # Agregamos el epsilon para evitar el log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    errors = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    loss = np.mean(errors)
    __errors__.append(loss)
    return loss


def normalize(samples):
    samples = np.array(samples)
    for i in range(samples.shape[1]):
        avg = np.mean(samples[:, i])
        max_val = np.max(samples[:, i])
        samples[:, i] = (samples[:, i] - avg) / max_val
    return samples

def create_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

X = df_cleaned.drop('income_int', axis=1).values
y = df_cleaned['income_int'].values

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_ratio = 0.8
train_size = int(train_ratio * X.shape[0])

x_train = X[:train_size]
x_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

X_train = normalize(x_train)
X_test = normalize(x_test)

params = np.zeros(X_train.shape[1] + 1)

lr = 0.001

epochs = 0

batch_size = 500

while True:
    old_params = np.copy(params)
    for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
        params = GD(params, x_train, lr, y_train)
        error = cross_entropy(params, x_train, y_train)
        print(error)
    epochs += 1
    if np.array_equal(old_params, params) or error <= 0.4 or epochs >= 100:
        print("samples: ", x_train)
        print("final params: ", params)
        print("Final error: ", error)
        break

preds = np.array([h(params[:-1], sample) + params[-1] for sample in x_train])
print(f"# of epochs: {epochs}")

# plt.plot(__errors__)
# plt.title('Logistic Regression Error per Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Value of Error')
# plt.show()

plt.scatter(range(len(preds)), preds, color='blue', alpha=0.5, label='Predictions')
plt.scatter(range(len(y_train)), y_train, color='red', alpha=0.5, label='Actual Values')

# Add title and labels
plt.title('Scatter Plot of Predictions and Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.show()