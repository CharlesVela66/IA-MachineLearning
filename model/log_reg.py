import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el data frame limpio
df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

# Arreglo para guardar los errores y los thresholds para poder clasificar mejor las clases
__errors__ = []
__thresholds__ = []

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
def gradient_descent(params, X, y, lr, lambda_reg=0.001):
    predictions = predict(params, X)
    errors = predictions - y
    gradient = np.dot(X.T, errors) / len(y)
    params[:-1] -= lr * (gradient + lambda_reg * params[:-1]) #El valor de lambda_reg controla la fuerza de la regularización. Al aumentar su valor, se penaliza más fuertemente los coeficientes grandes.
    params[-1] -= lr * np.mean(errors)
    return params

# Funcion que normaliza los datos de entrada (samples)
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Funcion para obtener cuantas instancias fueron predecidas con la clase correcta
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

def roc_curve(y_true, y_preds):
    sorted_indices = np.argsort(y_preds)
    y_true_sorted = y_true[sorted_indices]
    y_preds_sorted = y_preds[sorted_indices]

    tpr_values = []
    fpr_values = []
    num_positives = np.sum(y_true)
    num_negatives = len(y_true) - num_positives

    # Calculate TPR and FPR for different thresholds
    for threshold in np.linspace(1, 0, 100):  # From 1 to 0, descending
        tp = np.sum((y_preds_sorted >= threshold) & (y_true_sorted == 1))
        fp = np.sum((y_preds_sorted >= threshold) & (y_true_sorted == 0))
        tpr = tp / num_positives
        fpr = fp / num_negatives
        tpr_values.append(tpr)
        fpr_values.append(fpr)
        __thresholds__.append(threshold)

    return fpr_values, tpr_values

def auc_roc(fpr_values, tpr_values):
    auc = 0.0
    
    for i in range(1, len(fpr_values)):
        width = fpr_values[i] - fpr_values[i - 1]
        height_avg = (tpr_values[i] + tpr_values[i - 1]) / 2
        auc += width * height_avg
    
    return auc

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

# Division manual de train-validation
val_ratio = 0.8
val_size = int(val_ratio * x_train.shape[0])

x_val = x_train[val_size:]
x_train = x_train[:val_size]
y_val = y_train[val_size:]
y_train = y_train[:val_size]

# Inicializar parámetros
params = np.zeros(x_train.shape[1] + 1)

# Tasa de aprendizaje
lr = 0.03

# Número de iteraciones
epochs = 3000

# Entrenamiento
for epoch in range(epochs):
    params = gradient_descent(params, x_train, y_train, lr)
    error = cross_entropy(params, x_train, y_train)
    __errors__.append(error)
    print(f'Epoch {epoch + 1}, Error: {error}')

# Predicciones
preds_train = predict(params, x_train)
preds_val = predict(params, x_val)
preds_test = predict(params, x_test)

plt.plot(__errors__)
plt.title('Logistic Regression Error per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Value of Error')
plt.ylim(0,0.8)
plt.show()


def scatter_plot(pred, actual, num_points, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), pred[:num_points], color='blue', alpha=0.3, label='Predictions')
    plt.scatter(range(num_points), actual[:num_points], color='red', alpha=0.3, label='Actual Values')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


fpr_values_train, tpr_values_train = roc_curve(y_train, preds_train)
fpr_values_val, tpr_values_val = roc_curve(y_val, preds_val)
fpr_values_test, tpr_values_test = roc_curve(y_test, preds_test)

plt.figure(figsize=(8, 6))

# Curva ROC para el conjunto de prueba (test)
plt.plot(fpr_values_test, tpr_values_test, color='blue', label=f'Test ROC Curve (AUC = {auc_roc(fpr_values_test, tpr_values_test):.2f})')

# Curva ROC para el conjunto de entrenamiento (train)
plt.plot(fpr_values_train, tpr_values_train, color='green', label=f'Train ROC Curve (AUC = {auc_roc(fpr_values_train, tpr_values_train):.2f})')

# Curva ROC para el conjunto de validación (validation)
plt.plot(fpr_values_val, tpr_values_val, color='orange', label=f'Validation ROC Curve (AUC = {auc_roc(fpr_values_val, tpr_values_val):.2f})')

# Línea de referencia para un clasificador aleatorio
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier (AUC = 0.5)')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Desired FPR and TPR
desired_fpr = 0.22
desired_tpr = 0.80

# Find the index of the closest FPR
index = np.argmin(np.abs(np.array(fpr_values_test) - desired_fpr))

# Get the threshold corresponding to this index
threshold = __thresholds__[index]

print("Threshold corresponding to FPR {:.2f} and TPR {:.2f}: {:.4f}".format(desired_fpr, desired_tpr, threshold))

# Predicciones
preds_train = predict(params, x_train) >= threshold
preds_val = predict(params, x_val) >= threshold
preds_test = predict(params, x_test) >= threshold

# Evaluación de las predicciones
accuracy_train = np.mean(preds_train == y_train)
print(f'Accuracy train: {accuracy_train}')

accuracy_val = np.mean(preds_val == y_val)
print(f'Accuracy validation: {accuracy_val}')

accuracy_test = np.mean(preds_test == y_test)
print(f'Accuracy test: {accuracy_test}')

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_pred, y_true)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', yticklabels=['Predicted 0', 'Predicted 1'], xticklabels=['Actual 0', 'Actual 1'])
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title(title)
    plt.show()

scatter_plot(preds_train, y_train, 100, "Scatter Plot of Predictions vs Actual Vales(Train Data)")
scatter_plot(preds_val, y_val, 100, "Scatter Plot of Predictions vs Actual Vales(Validation Data)")
scatter_plot(preds_train, y_train, 100, "Scatter Plot of Predictions vs Actual Vales(Test Data)")

plot_confusion_matrix(y_train, preds_train, title='Confusion Matrix - Train Data')
plot_confusion_matrix(y_val, preds_val, title='Confusion Matrix - Validation Data')
plot_confusion_matrix(y_test, preds_test, title='Confusion Matrix - Test Data')

