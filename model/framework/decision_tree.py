import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

# Preparar los datos para el árbol de decisión
x = df_cleaned.drop('income_int', axis=1).values
y = df_cleaned['income_int'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Escalar los datos (opcional, pero puede ayudar)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# Crear el modelo de árbol de decisión
tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Entrenar el modelo
tree_classifier.fit(x_train_std, y_train)

# Realizar predicciones en los conjuntos de prueba
y_pred_train = tree_classifier.predict(x_train_std)
y_pred_test = tree_classifier.predict(x_test_std)

# Evaluar el modelo
print("Training accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))


cm_train = confusion_matrix(y_train, y_pred_train, normalize='true')
cm_test = confusion_matrix(y_test, y_pred_test, normalize='true')

cm_train_display = ConfusionMatrixDisplay(cm_train, display_labels=['0', '1'])
print(cm_train)
fig, ax = plt.subplots(figsize=(8, 8))
cm_train_display.plot(ax=ax, cmap=plt.cm.Blues) 

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Real')

plt.show()

cm_test_display = ConfusionMatrixDisplay(cm_test, display_labels=['0', '1'])
print(cm_test)
fig, ax = plt.subplots(figsize=(8, 8))
cm_test_display.plot(ax=ax, cmap=plt.cm.Blues)

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Real')

plt.show()