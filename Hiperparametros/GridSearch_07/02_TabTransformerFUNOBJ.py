import pandas as pd
import tensorflow as tf
### GPU:
import tensorflow as tf

print("Dispositivos disponibles:")
print(tf.config.list_physical_devices('GPU'))

if tf.test.is_gpu_available():
    print("TensorFlow está utilizando la GPU.")
else:
    print("TensorFlow no está utilizando la GPU.")
### FIN GPU.
from tensorflow.keras import Model, Input
from itertools import product
# Calbacks:
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from TabSynth import gen_cat_vocab, save_loss_plots, TabTransformer
import os
from tensorflow.keras.optimizers import Adam
from SeqMetrics import RegressionMetrics

# Learning rate de optimizador Adam:
lr = 0.005# 0.001

dataset = pd.read_csv('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/data/dataAP5/datasetAP5.csv')
dataset = dataset.drop(columns=['Overshoot','Tiempo_asentamiento','Valor_estacionario'], axis=1)
# Cargar el conjunto de prueba total
TestX = pd.read_csv('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/data/dataAP5/test_set_all_AP5.csv')
# Cargar el conjunto de datos de entrenamiento:
TrainX = pd.read_csv('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/data/dataAP5/train_set_AP5.csv')
# Cargar el conjunto de validación:
ValX = pd.read_csv('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/data/dataAP5/val_set_AP5.csv')
#Calbacks para monitorear el learning rate. (optimizador Adam)
lr_scheduler = ReduceLROnPlateau(monitor='val_Ganancia_Kp_loss',
                                 factor=0.2,
                                 patience=5,
                                 verbose=0)
# Callbacks Early Stopping:
# early_stopping = EarlyStopping(monitor='val_Ganancia_Kp_loss', #'val_loss
#                                patience=10,
#                                restore_best_weights=True)
CAT_FEATURES = dataset.columns.to_list()[0:2]
LABEL = dataset.columns.to_list()[2:4]
NUMERIC_FEATURES = dataset.columns.to_list()[4:434]
# print(f'Numeric features: {NUMERIC_FEATURES}\nCategoric features: {CAT_FEATURES}\nObjetivo: {LABEL}')
# Tipo de variable:
dataset[NUMERIC_FEATURES] = dataset[NUMERIC_FEATURES].astype(float)
dataset[CAT_FEATURES] = dataset[CAT_FEATURES].astype(str)
dataset[LABEL] = dataset[LABEL].astype(float)
# Vocabulario:
vocabulary = gen_cat_vocab(dataset)
# Características y labels:
train_x_all = [TrainX[NUMERIC_FEATURES].values, TrainX[CAT_FEATURES].values]
train_y_all = TrainX[LABEL]
val_x_all = [ValX[NUMERIC_FEATURES].values, ValX[CAT_FEATURES].values]
val_y_all = ValX[LABEL]
test_x_all = [TestX[NUMERIC_FEATURES].values, TestX[CAT_FEATURES].values]
test_y_all = TestX[LABEL]
# Entradas para el modelo:
input_num = Input(shape=(len(NUMERIC_FEATURES),), name='Input_num')
input_cat = Input(shape=(2,), dtype=tf.string, name='Input_cat')

# Definimos los valores para el GridSearch
param_grid = {
    'hidden_units': [2, 4, 8, 16, 32], # Dimension de la capa Multi-Head Attention.  8, 16
    'num_heads': [2, 4, 8], # Head de la Multi-Head Attention.
    'depth': [2, 4, 8], # Cantidad de Bloques Transformer.
    'dropout':[0.3], # Dropout de la capa de atención.
    'l2_lambda': [0.000001, 0.00001, 0.0001] #L2 de la salida de la capa MLP, ORIGINAL: 0.001 , 0.01, 0.1 --- 0.0001, 0.001
}

# Crear combinaciones de los hiperparámetros
combinations = list(product(*param_grid.values()))
# Almacenar resultados
results = []
for i, (hidden_units, num_heads, depth, dropout, l2_lambda) in enumerate(combinations):# hidden_units, num_heads, depth, dropout,
    print(f"Combinación {i + 1}/{len(combinations)}:hidden_units={hidden_units},num_heads={num_heads},depth={depth},dropout={dropout},l2_lambda={l2_lambda}")#  num_heads={num_heads}, depth={depth}, dropout={dropout}, 

    tab_transformer = TabTransformer(
        num_numeric_features=len(NUMERIC_FEATURES),
        cat_vocabulary=vocabulary,
        hidden_units=hidden_units,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout, # Dropout de la capa de Atención. 0.3
        num_dense_lyrs=2,
        l2_lambda=l2_lambda, # Controla el kernel_regularized de la MLP .
        num_outputs=2,
        final_mlp_units= [42, 84]
    )

    outputs, imp = tab_transformer([input_num, input_cat])

    model = Model(inputs=[input_num, input_cat], outputs=outputs, name=f'Model_{i + 1}')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={'Ganancia_Kp': 'mse', 'Periodo_Ti': 'mse'},
        metrics={'Ganancia_Kp': ['mse','mae'], 'Periodo_Ti': ['mse','mae']}
    )
    
    save_dir = f"Modelos_00/TabTransformer_02/GridSearch_07/GridSearch_7_Model_{i + 1}"
    os.makedirs(save_dir, exist_ok=True)

    plot_model(model, to_file=os.path.join(save_dir, "model_structure.png"), show_shapes=True, dpi=300)
    
    # Vamos a monitorear val_Ganancia_Kp_loss
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_dir, "best_weights.h5"), 
        monitor='val_Ganancia_Kp_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    early_stopping = EarlyStopping(
        monitor='val_Ganancia_Kp_loss',
        patience=15,
        restore_best_weights=True
    )    
    
    history = model.fit(
        x=train_x_all,
        y={'Ganancia_Kp':train_y_all['Kp'],
           'Periodo_Ti': train_y_all['Ti']},
        validation_data=(val_x_all,
                         {'Ganancia_Kp': val_y_all['Kp'],
                          'Periodo_Ti': val_y_all['Ti']}),        
        callbacks=[early_stopping, checkpoint_callback, lr_scheduler],
        batch_size=32,
        epochs=500,
        verbose=0 
    )
    # print(f'LR: {lr_scheduler}')
    prediction = model.predict(test_x_all)
    true = test_y_all
    
    # Evaluación del modelo para cada salida:
    r2_kp = RegressionMetrics(true['Kp'], prediction['Ganancia_Kp'])
    r2_ti = RegressionMetrics(true['Ti'], prediction['Periodo_Ti'])
    kp_r2 = r2_kp.r2()
    ti_r2 = r2_ti.r2()
    print(f'R² para Kp: {kp_r2}\nR² para Ti: {ti_r2}')
    # Guardamos el "history"
    hist = pd.DataFrame(history.history)
    hist.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    # Guardamos las figuras    
    save_loss_plots(history, save_dir)    
    # Guardar resultados, en este caso guarda el 'val_loss' y toma el menor de ellos en todos los epochs.
    val_loss = min(history.history['val_loss'])
    val_loss_kp = min(history.history['val_Ganancia_Kp_loss'])
    val_loss_ti = min(history.history['val_Periodo_Ti_loss'])
    loss = min(history.history['loss'])
    loss_kp = min(history.history['Ganancia_Kp_loss'])
    loss_ti = min(history.history['Periodo_Ti_loss'])

    results.append({
        'combination': (hidden_units, hidden_units, num_heads, depth, dropout, l2_lambda), # 
        'val_loss': val_loss,
        'val_kp_loss': val_loss_kp,
        'iteracion': i + 1
    })
    
    print(f"Validación final:\n\tval_loss: {val_loss}\n\tval_loss_Kp: {val_loss_kp}\n\tval_loss_ti: {val_loss_ti}\nPérdida total final:\n\tloss: {loss}\n\tloss_kp: {loss_kp}\n\tloss_ti: {loss_ti}")
    print('#'*100)

# Ordenar resultados por menor val_loss
results = sorted(results, key=lambda x: x['val_kp_loss']) #original 'val_loss'

# Imprimir 5 mejores combinaciones
print("Mejores combinaciones de hiperparámetros:")
for r in results[:5]:
    print(f"Iteración: {i},Parámetros: {r['combination']}, val_kp_loss: {r['val_kp_loss']}")