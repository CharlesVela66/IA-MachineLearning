import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Funcion para calcular la perdida de entropia cruzada con pesos
def weighted_cross_entropy(params, X, y, class_weights):
    predictions = predict(params, X)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    errors = -y * np.log(predictions) * class_weights[1] - (1 - y) * np.log(1 - predictions) * class_weights[0]
    return np.mean(errors)

# Funcion que optimiza los parametros del algoritmo mediante la actualizacion de los parametros
def gradient_descent(params, X, y, lr, class_weights, lambda_reg=1e-15):
    predictions = predict(params, X)
    errors = predictions - y
    weighted_errors = errors * (y * class_weights[1] + (1 - y) * class_weights[0])
    gradient = np.dot(X.T, weighted_errors) / len(y)
    params[:-1] -= lr * (gradient + lambda_reg * params[:-1])
    params[-1] -= lr * np.mean(weighted_errors)
    return params

# Funcion que normaliza los datos de entrada (samples)
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def confusion_matrix(pred, target):
    matrix = [[0,0],[0,0]]
    for i in range(len(pred)):
        if (pred[i] == 0):
            if (target[i] == 0):
                matrix[0][0] += 1
            else: 
                matrix[0][1] += 1
        else :
            if (target[i] == 1):
                matrix[1][1] +=1
            else:
                matrix[1][0] +=1
    return matrix

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

# Calcular los pesos de las clases
class_weights = np.array([1.0, 1.3])

# Inicializar parámetros
params = np.zeros(x_train.shape[1] + 1)

# Tasa de aprendizaje
lr = 0.03

# Número de iteraciones
epochs = 3000

# Entrenamiento
for epoch in range(epochs):
    params = gradient_descent(params, x_train, y_train, lr, class_weights)
    error = weighted_cross_entropy(params, x_train, y_train, class_weights)
    __errors__.append(error)
    print(f'Epoch {epoch + 1}, Error: {error}')

# Predicciones
preds_train = predict(params, x_train) >= 0.38
preds_test = predict(params, x_test) >= 0.38

# Evaluación de las predicciones
accuracy_train = np.mean(preds_train == y_train)
print(f'Accuracy train: {accuracy_train}')

accuracy_test = np.mean(preds_test == y_test)
print(f'Accuracy test: {accuracy_test}')

# Matrices de confusion
cm_train = confusion_matrix(preds_train, y_train)
print(cm_train)

cm_test = confusion_matrix(preds_test, y_test)
print(cm_test)

plt.plot(__errors__)
plt.title('Logistic Regression Error per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Value of Error')
plt.ylim(0,0.8)
plt.show()


def scatter_plot(pred, actual, num_points, title):
    plt.figure(figsize=(14, 6))
    plt.scatter(range(num_points), pred[:num_points], color='blue', alpha=0.5, label='Predictions')
    plt.scatter(range(num_points), actual[:num_points], color='red', alpha=0.5, label='Actual Values')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

scatter_plot(preds_train, y_train, 500, "Scatter Plot of Predictions vs Actual Vales(Train Data)")
scatter_plot(preds_train, y_train, 500, "Scatter Plot of Predictions vs Actual Vales(Test Data)")

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_pred, y_true)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', yticklabels=['Predicted 0', 'Predicted 1'], xticklabels=['Actual 0', 'Actual 1'])
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_train, preds_train, title='Confusion Matrix - Train Data')
plot_confusion_matrix(y_test, preds_test, title='Confusion Matrix - Test Data')