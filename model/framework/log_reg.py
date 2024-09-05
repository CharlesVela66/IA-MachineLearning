import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

# Preparar los datos para la regresión logística
x = df_cleaned.drop('income_int', axis=1).values
y = df_cleaned['income_int'].values

# Hacer una shuffle de los datos y asignarlos a x y y
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# Creación del modelo de regresión logística con regularización L2
log_reg = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=1000, class_weight={0: 1, 1: 1.5})

# Entrenamiento del modelo
log_reg.fit(x_train_std, y_train)

# Evaluación del modelo
print("Training accuracy:", log_reg.score(x_train_std, y_train))
print("Test accuracy:", log_reg.score(x_test_std, y_test))


y_pred_train = log_reg.predict(x_train_std)
y_pred_test = log_reg.predict(x_test_std)

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