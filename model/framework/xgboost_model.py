import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Cargar los datos
df_cleaned = pd.read_csv("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset/cleaned_data.csv")

# Preparar los datos para XGBoost
x = df_cleaned.drop('income_int', axis=1).values
y = df_cleaned['income_int'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Dividir los datos en conjuntos de entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Crear el modelo XGBoost
xgboost_model = xgb.XGBClassifier(n_estimators=100,
                                  max_depth=4,
                                  learning_rate=0.1,
                                  objective='binary:logistic',
                                  scale_pos_weight=1.9,
                                  use_label_encoder=False,  # Evitar warnings
                                  eval_metric='logloss')

# Ajustar los datos de entrenamiento en el modelo XGBoost
xgboost_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)

# Predicciones
y_pred_train = xgboost_model.predict(x_train)
y_pred_val = xgboost_model.predict(x_val)
y_pred_test = xgboost_model.predict(x_test)

# Evaluación del modelo
print("Train Accuracy: ", xgboost_model.score(x_train, y_train))
print("Validation Accuracy: ", xgboost_model.score(x_val, y_val))
print("Test Accuracy: ", xgboost_model.score(x_test, y_test))

# Matriz de confusión para el conjunto de entrenamiento
cm_train = confusion_matrix(y_train, y_pred_train, normalize='true')
cm_train_display = ConfusionMatrixDisplay(cm_train, display_labels=['0', '1'])
fig, ax = plt.subplots(figsize=(8, 8))
cm_train_display.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix (Train)')
plt.show()

# Matriz de confusión para el conjunto de validación
cm_val = confusion_matrix(y_val, y_pred_val, normalize='true')
cm_val_display = ConfusionMatrixDisplay(cm_val, display_labels=['0', '1'])
fig, ax = plt.subplots(figsize=(8, 8))
cm_val_display.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix (Validation)')
plt.show()

# Matriz de confusión para el conjunto de prueba
cm_test = confusion_matrix(y_test, y_pred_test, normalize='true')
cm_test_display = ConfusionMatrixDisplay(cm_test, display_labels=['0', '1'])
fig, ax = plt.subplots(figsize=(8, 8))
cm_test_display.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix (Test)')
plt.show()

# CURVA ROC y AUC-ROC
# Obtener las probabilidades de predicción
y_prob_train = xgboost_model.predict_proba(x_train)[:, 1]
y_prob_val = xgboost_model.predict_proba(x_val)[:, 1]
y_prob_test = xgboost_model.predict_proba(x_test)[:, 1]

# Calcular la curva ROC y el AUC-ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
fpr_val, tpr_val, _ = roc_curve(y_val, y_prob_val)
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

auc_train = roc_auc_score(y_train, y_prob_train)
auc_val = roc_auc_score(y_val, y_prob_val)
auc_test = roc_auc_score(y_test, y_prob_test)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_train:.2f})')
plt.plot(fpr_val, tpr_val, label=f'Validation (AUC = {auc_val:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.show()
