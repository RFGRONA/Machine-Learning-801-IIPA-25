import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Se añade plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score

print("--- Iniciando el Clasificador de SPAM ---")

try:
    # --- FASE 1: PREPARACIÓN DE DATOS ---
    print("Cargando el dataset 'dataset_correos.csv'...")
    df = pd.read_csv('dataset_correos.csv', delimiter=';', on_bad_lines='skip')
    print("Dataset cargado exitosamente.")
    print(f"Etiquetas únicas encontradas: {df['etiqueta'].unique()}")

    features_to_drop = ['remitente', 'destinatarios', 'asunto', 'cuerpo', 'fecha_hora']
    df_processed = df.drop(columns=features_to_drop).dropna(subset=['etiqueta'])

    X = df_processed.drop('etiqueta', axis=1)
    y = df_processed['etiqueta']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    print("La preparación de datos ha finalizado.\n")

    # --- FASE 2: BUCLE DE ENTRENAMIENTO Y PREDICCIÓN ---
    n_executions = 50
    accuracy_results = []
    f1_results = []

    print(f"Iniciando {n_executions} ejecuciones del modelo...")
    for i in range(n_executions):
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.3, random_state=i, stratify=y_encoded
        )

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_results.append(accuracy_score(y_test, y_pred))
        f1_results.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    
    print(f"Las {n_executions} ejecuciones han finalizado.\n")

    # --- FASE 3: MEDICIÓN DE LA CALIDAD (EXACTITUD Y F1) ---
    mean_accuracy = np.mean(accuracy_results)
    std_accuracy = np.std(accuracy_results) 
    mean_f1 = np.mean(f1_results)

    print("--- Resultados de las Métricas de Desempeño ---")
    print(f"Exactitud (Accuracy) Promedio: {mean_accuracy:.4f}")
    print(f"Puntuación F1 (F1-Score) Promedio: {mean_f1:.4f}")

    # --- FASE 4: CÁLCULO DEL Z-SCORE ---
    z_scores_accuracy = [(score - mean_accuracy) / std_accuracy for score in accuracy_results]
    print(f"\nSe calcularon los Z-Scores para las {n_executions} ejecuciones.")

    # --- FASE 5: GRAFICAR LOS TRES RESULTADOS DE MÉTRICAS ---
    print("\nGenerando gráficos de resultados en ventanas distintas...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Gráfico 1: Exactitud (Accuracy)
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_executions), accuracy_results, marker='o', linestyle='-', color='dodgerblue', label='Exactitud en cada ejecución')
    plt.axhline(y=mean_accuracy, color='red', linestyle='--', label=f'Promedio = {mean_accuracy:.3f}')
    plt.title('Resultados de Exactitud (Accuracy) en 50 Ejecuciones', fontsize=15)
    plt.xlabel('Número de Ejecución', fontsize=12)
    plt.ylabel('Exactitud', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('Exactitud.png')

    # Gráfico 2: F1-Score
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_executions), f1_results, marker='o', linestyle='-', color='forestgreen', label='F1-Score en cada ejecución')
    plt.axhline(y=mean_f1, color='red', linestyle='--', label=f'Promedio = {mean_f1:.3f}')
    plt.title('Resultados de F1-Score en 50 Ejecuciones', fontsize=15)
    plt.xlabel('Número de Ejecución', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('F1-Score.png')
    
    # Gráfico 3: Z-Score de la Exactitud
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_executions), z_scores_accuracy, marker='.', linestyle='--', color='purple', label='Z-Score de la Exactitud')
    plt.axhline(y=0.0, color='black', linestyle='-') # El promedio de los Z-Scores es siempre 0
    plt.title('Z-Score de la Exactitud en 50 Ejecuciones', fontsize=15)
    plt.xlabel('Número de Ejecución', fontsize=12)
    plt.ylabel('Z-Score (Desviaciones Estándar)', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig('Z-Score.png')
    print("Gráficos de métricas mostrados.")

    # --- FASE 6: GENERAR GRÁFICO DEL ÁRBOL DE DECISIÓN ---
    print("\nEntrenando un último modelo para visualizar el árbol de decisión...")
    
    # Entrenar un modelo final con una división de datos representativa
    X_train_final, _, y_train_final, _ = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    # Se limita la profundidad a 4 niveles para que la gráfica sea legible.
    # Puedes aumentar este número si quieres ver un árbol más complejo y detallado.
    final_model = DecisionTreeClassifier(max_depth=4) 
    final_model.fit(X_train_final, y_train_final)

    # Obtener los nombres de las características para que el árbol sea interpretable
    feature_names = preprocessor.get_feature_names_out()
    class_names = label_encoder.classes_

    # Crear la visualización del árbol
    plt.figure(figsize=(30, 20)) # Tamaño grande para que los nodos no se solapen
    plot_tree(final_model, 
              feature_names=feature_names, 
              class_names=class_names, 
              filled=True, 
              rounded=True,
              fontsize=10)
    
    # Guardar la figura en un archivo PNG
    plt.savefig('arbol_de_decision.png')
    print("¡Éxito! Se ha guardado la gráfica del árbol como 'arbol_de_decision.png'")
    print("El programa ha finalizado exitosamente.")

except Exception as e:
    print(f"\nOcurrió un error inesperado durante la ejecución: {e}")