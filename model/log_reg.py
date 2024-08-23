import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el data frame limpio
df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

# Arreglo para guardar los errores
__errors__ = []

# Funcion sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función que nos ayuda a calcular las predicciones usando los parametros del modelo
def predict(params, X):
    return sigmoid(np.dot(X, params[:-1]) + params[-1])

# Funcion para calcular la perdida de entropia cruzada
def cross_entropy(params, X, y):
    predictions = predict(params, X)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    errors = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    return np.mean(errors)

# Funcion que optimiza los parametros del algoritmo mediante la actualizacion de los parametros
def gradient_descent(params, X, y, lr, lambda_reg=0.01):
    predictions = predict(params, X)
    errors = predictions - y
    gradient = np.dot(X.T, errors) / len(y)
    params[:-1] -= lr * (gradient + lambda_reg * params[:-1])
    params[-1] -= lr * np.mean(errors)
    return params

# Funcion que normaliza los datos de entrada (samples)
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Preparar los datos para la regresión logística
x = df_cleaned.drop('income_int', axis=1).values
y = df_cleaned['income_int'].values

# Hacer una shuffle de los datos y asignarlos a x y y
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# División manual de train-test
train_ratio = 0.8
train_size = int(train_ratio * x.shape[0])

# Asignar el tamaño de train y test para x y y
x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Normalizar muestras
x_train = normalize(x_train)
x_test = normalize(x_test)

# Inicializar parámetros
params = np.zeros(x_train.shape[1] + 1)

# Tasa de aprendizaje
lr = 0.03

# Número de iteraciones
epochs = 10000

# Entrenamiento
for epoch in range(epochs):
    params = gradient_descent(params, x_train, y_train, lr)
    error = cross_entropy(params, x_train, y_train)
    __errors__.append(error)
    print(f'Epoch {epoch}, Error: {error}')

# Predicciones
preds_train = predict(params, x_train) >= 0.5
preds_test = predict(params, x_test) >= 0.5

# Evaluación de las predicciones
accuracy_train = np.mean(preds_train == y_train)
print(f'Accuracy: {accuracy_train}')

accuracy_test = np.mean(preds_test == y_test)
print(f'Accuracy: {accuracy_test}')

plt.plot(__errors__)
plt.title('Logistic Regression Error per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Value of Error')
plt.show()

# plt.scatter(range(len(preds_train)), preds_train, color='blue', alpha=0.1, label='Predictions from train')
# plt.scatter(range(len(y_train)), y_train, color='red', alpha=0.1, label='Actual Values')

# # Add title and labels
# plt.title('Scatter Plot of Predictions and Actual Values')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()

# # Show plot
# plt.show()