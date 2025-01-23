import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV

data = pd.read_csv("H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/Modelos_00/TabTransformer_02/GridSearch_07/GridSearch_7_Model_49/training_history.csv")

# Crear el gráfico
plt.figure(figsize=(10, 6))  # Tamaño del gráfico
plt.plot(data['val_Ganancia_Kp_loss'], label='val_Ganancia_Kp_loss', color='blue', linewidth=2)
plt.plot(data['Ganancia_Kp_loss'], label='ent_Ganancia_Kp_loss', color='red', linewidth=2)

# Personalizar el gráfico
plt.title('Evolución de pérdida de Ganancia_Kp durante el entrenamiento', fontsize=16)
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Error MSE', fontsize=14)

# Hacer el eje y más claro
plt.grid(True,axis='y', linestyle='--', alpha=0.7)  # Cuadrícula en el eje y, estilo punteado y transparencia
plt.gca().set_facecolor('#f7f7f7')  # Color de fondo del gráfico

# Añadir leyenda
plt.legend(fontsize=12)

# Mostrar el gráfico
plt.tight_layout()  # Ajustar el layout para que no se corten etiquetas
plt.show()