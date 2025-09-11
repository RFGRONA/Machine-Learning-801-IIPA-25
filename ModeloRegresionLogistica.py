import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import os

# === 1. Cargar dataset ===
folder = r"C:\Users\ingLo\OneDrive\Desktop\MACHINE LEARNING"
file_path = os.path.join(folder, "dataset_correos.csv")
df = pd.read_csv(
    file_path,
    sep=";",
    quotechar='"',
    engine="python",
    on_bad_lines="skip"
)

# === 2. Preprocesamiento ===
df = df.dropna(subset=["etiqueta"])
df = df[df["etiqueta"].isin(["HAM", "SPAM"])]
df["etiqueta"] = df["etiqueta"].map({"HAM": 0, "SPAM": 1})

# Nombres cortos de las características originales
features = [
    "frecuencia_palabras_spam",
    "reputacion_dominio_remitente",
    "cantidad_destinatarios_visibles",
    "presencia_y_numero_urls",
    "relacion_texto_html",
    "autenticidad_remitente",
    "personalizacion_saludo",
    "adjuntos_peligrosos",
    "uso_simbolos_excesivos",
    "scripts_o_formularios"
]

# === 3. Preparación de Datos para el Modelo ===
X_for_model = df[features]
y = df["etiqueta"]
X = pd.get_dummies(X_for_model, drop_first=True)

# === 4. Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 5. Escalar datos ===
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# === 6. Entrenar modelo (Logística) ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# === 7. Predicciones ===
y_scores = model.predict_proba(X_test_s)[:, 1]

# === 8. Buscar umbral óptimo ===
best_thr, best_f1 = 0.5, -1
for thr in np.arange(0, 1.01, 0.01):
    y_pred = (y_scores >= thr).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

# === 9. Evaluación final ===
y_pred_final = (y_scores >= best_thr).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, y_pred_final)

# === 10. GRÁFICA 1: Matriz de Confusión y Distribución ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# --- Matriz de confusión ---
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title("Matriz de Confusión")
axes[0].set_xlabel("Predicción (0=HAM, 1=SPAM)")
axes[0].set_ylabel("Real (0=HAM, 1=SPAM)")
axes[0].text(
    0.5, -0.25,
    f"TN={tn}: HAM bien clasificados\n"
    f"FP={fp}: HAM clasificados como SPAM\n"
    f"FN={fn}: SPAM clasificados como HAM\n"
    f"TP={tp}: SPAM bien clasificados",
    ha="center", va="top", fontsize=9, transform=axes[0].transAxes
)
# --- Distribución de probabilidades ---
counts_ham, bins_ham, _ = axes[1].hist(y_scores[y_test == 0], bins=30, alpha=0.6, label="HAM (0)")
counts_spam, bins_spam, _ = axes[1].hist(y_scores[y_test == 1], bins=30, alpha=0.6, label="SPAM (1)")
axes[1].axvline(best_thr, color="red", linestyle="--", label=f"Umbral óptimo {best_thr:.2f}")
axes[1].set_title(
    f"Distribución de Probabilidades\n"
    f"F1-Score={(best_f1 * 100):.1f}%, Umbral={best_thr:.2f}"
)
axes[1].set_xlabel("Probabilidad de SPAM")
axes[1].set_ylabel("Cantidad")
axes[1].legend()
total = len(y_test)
for counts, bins in [(counts_ham, bins_ham), (counts_spam, bins_spam)]:
    for c, b in zip(counts, bins[:-1]):
        if c > 0:
            pct = 100 * c / total
            if pct > 1:
                axes[1].text(b + (bins[1]-bins[0])/2, c, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig('DistribucionProbabilidades.png')
plt.show()

# === 11. GRÁFICA 2: Curva de Aprendizaje ===
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model, X=X_train_s, y=y_train, cv=5, scoring='f1', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)
plt.figure(figsize=(10, 6))
n_train = len(X_train)
n_val = len(X_test)
title_text = (
    f'Curva de Aprendizaje del Modelo Logístico\n'
    f'({n_train} registros de entrenamiento, {n_val} de validación)'
)
plt.title(title_text)
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("F1-Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                 validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntuación de Entrenamiento")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Puntuación de Validación Cruzada")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('CurvaDeAprendizaje.png')
plt.show()

# === 12. GRÁFICA 3: Análisis de Correlación ===
correlation_df = df[features + ['etiqueta']].copy()
for col in correlation_df.columns:
    if correlation_df[col].dtype == 'object':
        correlation_df[col] = correlation_df[col].astype('category').cat.codes
corr_matrix = correlation_df.corr()

# <<< INICIO DE LA CORRECCIÓN DEFINITIVA
# 1. Creamos la figura y los ejes de forma explícita
fig, ax = plt.subplots(figsize=(12, 10))

# 2. Le pasamos el eje 'ax' a seaborn para que dibuje sobre él
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
ax.set_title('Matriz de Correlación de las 10 Características Principales', fontsize=16)

# 3. Usamos fig.tight_layout() que a veces es más efectivo
fig.tight_layout()

# 4. Guardamos la figura usando bbox_inches='tight' para un ajuste perfecto en el archivo
plt.savefig('MatrizDeCorrelacion_Simple.png', bbox_inches='tight')
# <<< FIN DE LA CORRECCIÓN DEFINITIVA

plt.show()


# === 13. GRÁFICA 4: Utilidad de las Características ===
coefficients = model.coef_[0]
feature_names_processed = X.columns
importance_df = pd.DataFrame({'Processed_Feature': feature_names_processed, 'Coefficient': coefficients})
importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
descriptive_features_map = {
    "frecuencia_palabras_spam": "Frecuencia de Palabras Clave de SPAM",
    "reputacion_dominio_remitente": "Reputación del Dominio del Remitente",
    "cantidad_destinatarios_visibles": "Cantidad de Destinatarios Visibles",
    "presencia_y_numero_urls": "Presencia y Número de Enlaces (URLs)",
    "relacion_texto_html": "Relación Texto/HTML",
    "autenticidad_remitente": "Autenticidad del Remitente (SPF, DKIM)",
    "personalizacion_saludo": "Personalización del Saludo",
    "adjuntos_peligrosos": "Adjuntos Peligrosos",
    "uso_simbolos_excesivos": "Uso Excesivo de Símbolos",
    "scripts_o_formularios": "Scripts o Formularios Embebidos"
}
aggregated_importance = {}
for original_feature_short, original_feature_long in descriptive_features_map.items():
    related_features = importance_df[importance_df['Processed_Feature'].str.startswith(original_feature_short)]
    total_impact = related_features['Abs_Coefficient'].sum()
    aggregated_importance[original_feature_long] = total_impact
final_importance_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Feature', 'Total_Impact'])
final_importance_df = final_importance_df.sort_values(by='Total_Impact', ascending=False)
total_impact_sum = final_importance_df['Total_Impact'].sum()
final_importance_df['Percentage'] = (final_importance_df['Total_Impact'] / total_impact_sum) * 100
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Total_Impact', y='Feature', data=final_importance_df, palette='viridis', hue='Feature', legend=False)
plt.title('Utilidad de Cada Característica en la Predicción', fontsize=16)
plt.xlabel('Puntuación de Impacto Total (Más es más útil)', fontsize=12)
plt.ylabel('Característica', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, (index, row) in enumerate(final_importance_df.iterrows()):
    ax.text(row['Total_Impact'] + 0.03, i, f"{row['Percentage']:.1f}%", color='black', ha="left", va="center", fontweight='bold')
plt.xlim(right=ax.get_xlim()[1] * 1.15)
plt.tight_layout()
plt.savefig('ImportanciaCaracteristicas.png')
plt.show()

# === 14. Resumen en consola ===
print("\n--- Resumen de Evaluación ---")
print("Umbral óptimo:", best_thr)
print(f"Mejor F1-Score: {(best_f1 * 100):.1f}%")
print("Matriz de confusión:\n", cm)