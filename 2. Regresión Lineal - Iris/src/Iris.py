import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import learning_curve

# --- 1. Cargar y Preparar Datos ---
try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo 'Iris.csv'.")
    exit()

species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species_Num'] = df['Species'].map(species_map)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species_Num']

# --- 2. Entrenar Modelo y Predecir ---
model = LinearRegression()
model.fit(X, y)

y_pred_float = model.predict(X)
y_pred_class = np.clip(np.round(y_pred_float), 0, 2).astype(int)
df['Predicted_Float'] = y_pred_float

# --- 3. Métricas de Rendimiento ---
accuracy = accuracy_score(y, y_pred_class)
print("--- MÉTRICAS DEL MODELO ---")
print(f"Precisión (Accuracy): {accuracy:.2%}")

mse = mean_squared_error(y, y_pred_float)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred_float)

print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print("\n--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(y, y_pred_class, target_names=target_names))

# --- 4. Generación de Gráficas ---

# Gráfica 1: Matriz de Confusión (Antes Gráfica 2)
cm = confusion_matrix(y, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión', fontsize=16)
plt.ylabel('Etiqueta Real (Lo que realmente es)')
plt.xlabel('Etiqueta Predicha (Lo que el modelo dijo que es)')
plt.savefig('Matriz de Confusión.png')
plt.show()

# Gráfica 2: Distribución de Predicciones (Antes Gráfica 3)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Predicted_Float', hue='Species', fill=True)
plt.title('Distribución de las Predicciones por Especie Real', fontsize=16)
plt.xlabel('Salida Continua del Modelo (Antes de Redondear)')
plt.ylabel('Densidad')
plt.axvline(x=0.5, color='grey', linestyle='--', label='Límite Setosa/Versicolor')
plt.axvline(x=1.5, color='grey', linestyle='--', label='Límite Versicolor/Virginica')
plt.legend()
plt.savefig('Distribución de Predicciones.png')
plt.show()


# Gráfica 3: Curva de Aprendizaje con Información Integrada (Antes Gráfica 6)
cv_folds = 5
n_validation_samples = round(X.shape[0] * (1/cv_folds))

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=LinearRegression(),
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv_folds,
    scoring='neg_mean_squared_error'
)
train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Error de Entrenamiento')
ax.plot(train_sizes, validation_scores_mean, 'o-', color='g', label=f'Error de Validación (sobre {n_validation_samples} muestras)')

# Bucle para añadir texto a cada punto de la curva de entrenamiento
for i in range(len(train_sizes)):
    ax.text(train_sizes[i],
            train_scores_mean[i],
            f" {train_sizes[i]}",
            color='black',
            fontsize=11,
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

ax.set_ylabel('Error Cuadrático Medio (MSE)')
ax.set_xlabel('Cantidad de Muestras de Entrenamiento')
ax.set_title('Curva de Aprendizaje del Modelo', fontsize=16)
ax.legend(loc='lower left')
ax.grid(True)

# --- MODIFICACIÓN AQUÍ: Añadir texto con información a la gráfica ---
# Preparamos el texto que vamos a mostrar
species_counts = df['Species'].value_counts()
info_text = (
    f"Precisión Total: {accuracy:.2%}\n\n"
    f"Conteo de Muestras:\n"
    f"Iris-setosa: {species_counts['Iris-setosa']}\n"
    f"Iris-versicolor: {species_counts['Iris-versicolor']}\n"
    f"Iris-virginica: {species_counts['Iris-virginica']}"
)

# Colocamos el texto en la esquina superior derecha de la gráfica
ax.text(0.95, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

plt.savefig('Curva de Aprendizaje.png')
plt.show()