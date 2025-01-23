'''
Dashboard 20 con set de datos AP5.
    - Período de muestreo T = 0.07043

Contenido:
    - MLP
    - Tabular-Transformer
    - Método Asignación de Polos

Código funciona para:
    - Obtener la señal de la planta con:
        - Valores de red neuronal
        - Valores de AP
Funcionamiento:
    1. Iniciar Simulación (Botón Iniciar).
    2. Cambiar valores de K y tau de la Planta con los slider.
    3. Cambiamos la referencia.
    4. Presionar "Mantener PI-AP" o "Actualizar PI-AP"
    6. Botón "Iniciar"
'''

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import TabSynth as TSS
from TabSynth import gen_cat_vocab, TabTransformer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import control as ctr
import os

################
dataset = pd.read_csv('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/data/dataAP5/datasetAP5.csv')
dataset = dataset.drop(columns=['Overshoot','Tiempo_asentamiento','Valor_estacionario'], axis=1)
vocabulary = gen_cat_vocab(dataset)
print(f'Vocabulario: {vocabulary}')
def build_model(hidden_units, num_heads, depth, dropout, l2_lambda):
    input_num = tf.keras.Input(shape=(430,), name='Input_num')
    input_cat = tf.keras.Input(shape=(2,), dtype=tf.string, name='Input_cat')

    tab_transformer = TabTransformer(
        num_numeric_features=430,
        cat_vocabulary=vocabulary,
        hidden_units=hidden_units,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout,
        num_dense_lyrs=2,
        l2_lambda=l2_lambda,
        num_outputs=2,
        final_mlp_units=[42, 84]
    )

    outputs, imp = tab_transformer([input_num, input_cat])
    model = tf.keras.Model(inputs=[input_num, input_cat], outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss={'Ganancia_Kp': 'mse',
              'Periodo_Ti': 'mse'},
        metrics={'Ganancia_Kp': ['mse', 'mae'],
                 'Periodo_Ti': ['mse', 'mae']}
    )
    
    return model

TAB_model = build_model(4,8,4,0.3,1e-06)
# Ruta del archivo de los pesos:
save_dir = f"H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/Modelos_00/TabTransformer_02/GridSearch_07/GridSearch_7_Model_49"#Modelos_00/TabTransformer_02/GridSearch_07/GridSearch_7_Model_49
weights_path = os.path.join(save_dir, "best_weights.h5")
# Cargar los pesos
TAB_model.load_weights(weights_path)
################

# '''MODELO MLP:'''
from tensorflow.keras.models import load_model
MLP_model = load_model('H:/Mi unidad/00_Magister/01_Tesis/Control_PID_Code_2/MLP_TensorFlow/MLP_01/Modelo_MLP.h5')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


T = 0.07043  # Tiempo de muestreo ajustado
ref_actual = 1.0  # Referencia actual
# Para almacenar las salidas de cada señal
output_planta = []
output_planta_AP = []
output_planta_TAB = []
output_error = []
output_error_TAB = []
output_control = []
output_control_TAB = []
resultados = []
x_acumulado = []  # Para acumular el eje x
ref = []

Kp_ant = []
Ti_ant = []
controlador = None
controlador_AP = None
controlador_TAB = None
planta_AP = None
planta = None
planta_TAB = None
simulacion_activa = False  # Controla si la simulación está activa
n_detener = 0
n_controlador_AP = 0
n_controlador_PI = 0
# Valores para categorias:
cat = ['BAJADA', 'LINEA RECTA', 'SUBIDA', 'LENTA', 'MEDIA', 'RAPIDA', 'CD0', 'CD2']
def valor_cat(cat, output):
    # Creamos lista de ceros, del mismo tamaño que las categorias:
    indicadores = [0.0]*len(cat)
    for i, categoria in enumerate(cat):
        if categoria in output:
            indicadores[i] = 1.0
    return indicadores
# Valores (de inicialización, iteración 0) del set de entrenamiento:
K = 5
tau_p = 1.0
Kp_MLP = 0.305556
Ti_MLP = 1.302778
Kp_TAB = 0.305556
Ti_TAB = 1.302778

multiplicador = 1

trace1 = go.Scatter(
        x=x_acumulado, # Eje temporal 
        y= output_planta, # Señal de la planta 
        mode='lines',#lines+markers
        name='PI-NN'
        )
trace2 = go.Scatter(
        x=x_acumulado, # Eje temporal 
        y= output_planta_AP, # Señal de la planta AP
        mode='lines',#lines+markers
        name='PI-AP'
        )
trace4 = go.Scatter(
            x=x_acumulado, # Eje temporal 
            y= ref, # Señal de la planta AP
            mode='lines+markers',#lines+markers
            name='Ref'
        )
trace3 = go.Scatter(
            x=x_acumulado, # Eje temporal 
            y= output_planta_TAB, # Señal de la planta AP
            mode='lines',#lines+markers
            name='PI-TAB'
        )

# Inicializar el controlador y la planta
# Primera iteración:
def inicializar_sistema():
    global controlador, planta, K, tau_p, Kp_MLP, Ti_MLP, Kp_TAB, Ti_TAB, union_,controlador_TAB, planta_TAB, controlador_AP, planta_AP
    K = K
    tau_p = tau_p
    
    output_cat_init_MLP = asignacion_categoria(Kp_MLP, Ti_MLP, K, tau_p)
    output_cat_init_TAB = asignacion_categoria(Kp_TAB, Ti_TAB, K, tau_p)
    asignacion_de_polos()
    
    print(f'Inicializando MLP con:\n\tKp: {Kp_MLP}, {type(Kp_MLP)}\n\tTi: {Ti_MLP}, {type(Ti_MLP)}\n\tK: {K}, {type(K)}\n\ttau_p: {tau_p}, {type(tau_p)},\n\tCategoria_1: {output_cat_init_MLP[0]}\n\tCategoria_2: {output_cat_init_MLP[1]}')
    print(f'Inicializando TAB con:\n\tKp: {Kp_TAB}, {type(Kp_TAB)}\n\tTi: {Ti_TAB}, {type(Ti_TAB)}\n\tK: {K}, {type(K)}\n\ttau_p: {tau_p}, {type(tau_p)},\n\tCategoria_1: {output_cat_init_TAB[0]}\n\tCategoria_2: {output_cat_init_TAB[1]}')
    controlador = TSS.PIControlador(Kp_MLP, Ti_MLP, T)
    planta = TSS.PlantaPrimerOrden(K, tau_p, T)
    controlador_TAB = TSS.PIControlador(Kp_TAB, Ti_TAB, T)
    planta_TAB = TSS.PlantaPrimerOrden(K, tau_p, T)
    controlador_AP = TSS.PIControlador(0.305556,1.302778,T)# Por esto queda fijo Kp=0.12, Ti=0.6397
    planta_AP = TSS.PlantaPrimerOrden(K,tau_p,T)


def asignacion_de_polos():
    global controlador_AP, planta_AP, Kp_ant, Ti_ant, Kp_AP, Ti_AP
    metodoDirecto = TSS.AsignacionPolos(K, tau_p, T)
    param = metodoDirecto.parametros_controladorAP()
    Kp_AP = param['Kp']
    Ti_AP = param['Ti']
    Kp_ant.append(Kp_AP)
    Ti_ant.append(Ti_AP)
    output_cat_init_AP = asignacion_categoria(Kp_AP, Ti_AP, K, tau_p)
    print('*'*100)
    print(f'Inicializando AP con:\n\tKp: {Kp_AP}, {type(Kp_AP)}\n\tTi: {Ti_AP}, {type(Ti_AP)}\n\tK: {K}, {type(K)}\n\ttau_p: {tau_p}, {type(tau_p)},\n\tCategoria_1: {output_cat_init_AP[0]}\n\tCategoria_2: {output_cat_init_AP[1]}')    
    print(f'Asignación de Polos Actual\n\tKp: {Kp_AP}\n\tTi: {Ti_AP}\n\tPara Planta:\n\tGanancia K: {K}, Tau_p: {tau_p} ')
    
    # print('-'*50)
    print(f'\tKp total ant: {Kp_ant}\n\tTi total ant: {Ti_ant}')
    # print(f'Asignación de Polos Anterior\nKp_ant: {Kp_ant[-2]}\nTi_ant: {Ti_ant[-2]}')
    T_asen = param['Tiempo_asentamiento']
    ess = param['Valor_estado_estacionario']
    sobrepaso = param['Sobrepaso']
    # controlador_AP = TSS.PIControlador(Kp_AP,Ti_AP,T)
    # planta_AP = TSS.PlantaPrimerOrden(K,tau_p,T)

#Función NO mantener PI anterior, utiliza los valores de Kp y Ti por AP con respecto a la planta dada.
'''Las funciones NO_mantener_PI y mantener_PI son solo para el controlador y la planta por asignación de polos'''
def NO_mantener_PI():
    global controlador_AP, planta_AP
    print(f'PI\n\tKp_AP: {Kp_AP}\n\tTi_AP: {Ti_AP}\nPara Planta:\n\tGanancia K: {K}, Tau_p: {tau_p}')
    # controlador_AP = TSS.PIControlador(Kp_ant[-2], Ti_ant[-2], T)
    # planta_AP = TSS.PlantaPrimerOrden(K,tau_p, T)
    controlador_AP.K_p = Kp_AP
    print(f'\tValor Kp a ingresar: {controlador_AP.K_p}')
    controlador_AP.T_i = Ti_AP
    print(f'\tValor Ti a ingresar: {controlador_AP.T_i}')
    planta_AP.K = K
    planta_AP.tau_p = tau_p

#Función mantiene el Kp y Ti de la planta anterior.
def mantener_PI():
    global controlador_AP, planta_AP
    # print(f'Asignación de Polos Actual\n\tKp: {Kp_AP}\n\tTi: {Ti_AP}\nPara Planta:\n\tGanancia K: {K}, Tau_p: {tau_p} ')
    print(f'Kp total ant: {Kp_ant}\nTi total ant: {Ti_ant}')
    print(f'Mantener PI\n\tKp_ant: {Kp_ant[-1]}\n\tTi_ant: {Ti_ant[-1]}\nPara Planta:\n\tGanancia K: {K}, Tau_p: {tau_p}')
    # controlador_AP = TSS.PIControlador(Kp_ant[-2], Ti_ant[-2], T)
    # planta_AP = TSS.PlantaPrimerOrden(K,tau_p, T)
    controlador_AP.K_p = Kp_ant[-1]
    print(f'\tValor Kp a ingresar: {controlador_AP.K_p}')
    controlador_AP.T_i = Ti_ant[-1]
    print(f'\tValor Ti a ingresar: {controlador_AP.T_i}')
    planta_AP.K = K
    planta_AP.tau_p = tau_p

def asignacion_categoria(Kp, Ti, K, tau_p):
    global cat_1, cat_2
    num_cd = [Kp*(1+T/(2*Ti)), -Kp*(1-T/(2*Ti))]
    den_cd = [1, -1]
    Cd = ctr.TransferFunction(num_cd,den_cd, T)
    e = np.exp(-T/tau_p)
    num_gd = K*(1 -e)
    den_gd = [1, -e]
    Gd = ctr.TransferFunction(num_gd,den_gd, T)
    #Lazo cerrado:
    H = ctr.feedback(Cd*Gd,1)
    info = ctr.step_info(H)
    Valor_ess = info['SteadyStateValue']# Valor estado estable
    Tiempo_estable = info['SettlingTime']# Tiempo de asentamiento
    Overshoot = info['Overshoot']
    categoria = TSS.Categoria(K, tau_p, K_nominal=10, tau_nominal=10)
    cat_1 = categoria.categoria_pendiente()
    cat_2 = categoria.categoria_(Tiempo_estable)
    return [cat_1, cat_2]

'''La función actualizar_parametros es para la actualización de los valores del controlador de la red neuronal'''
def actualizar_parametros(Kp_MLP, Ti_MLP,Kp_TAB, Ti_TAB, K, tau_p, ref=ref_actual):
    global controlador, planta,controlador_TAB, planta_TAB
    # print(f'Valores de la actualización de parámetros NN:\n\tKp: {Kp}, {type(Kp)}\n\tTi: {Ti}, {type(Ti)}\n\tK: {K}, {type(K)}\n\ttau_p: {tau_p}, {type(tau_p)}\n\tReferencia actual: {ref}')
    print('-'*50)
    # Valores que ingresan al controlador de la red MLP:
    controlador.K_p = Kp_MLP
    controlador.T_i = Ti_MLP
    output_cat_init_MLP = asignacion_categoria(Kp_MLP, Ti_MLP, K, tau_p)
    print(f'Valores del controlador MLP:\n\tKp MLP: {controlador.K_p}\n\tTi MLP: {controlador.T_i}')
    print(f'\tCategoría 1: {output_cat_init_MLP[0]}\n\tCategoría 2: {output_cat_init_MLP[1]}')
    #Probamos con valores de la planta fija MLP:
    planta.K = K
    planta.tau_p = tau_p
    # Valores que ingresar al controlador de la red TAB:
    controlador_TAB.K_p = Kp_TAB
    controlador_TAB.T_i = Ti_TAB
    output_cat_init_TAB = asignacion_categoria(Kp_TAB, Ti_TAB, K, tau_p)
    print(f'Valores del controlador TAB:\n\tKp: {controlador_TAB.K_p}\n\tTi: {controlador_TAB.T_i}')
    print(f'\tCategoría 1: {output_cat_init_TAB[0]}\n\tCategoría 2: {output_cat_init_TAB[1]}')
    # Planta de la red TAB:
    planta_TAB.K = K
    planta_TAB.tau_p = tau_p
    print(f'Valores de la Planta MLP y TAB:\n\tK: {planta.K}\n\ttau_p: {planta.tau_p}')

# Simular el sistema
def simular_sistema(ref_actual, K, tau_p):
    global output_control, output_planta, output_error, x_acumulado, controlador, planta, union_, controlador_AP, planta_AP, output_planta_AP, ref, ind, planta,controlador_TAB, planta_TAB

    # Si la simulación no ha comenzado, inicializar el sistema
    if controlador is None and planta is None and controlador_AP is None and planta_AP is None:
        inicializar_sistema()

    # Tomar el último valor de la salida anterior para continuar
    if (len(output_planta) == 0) or (len(output_planta_AP) == 0) or (len(output_planta_TAB)==0):
        v_med = 0
        v_med_AP = 0
        v_med_TAB = 0
    else:
        v_med = output_planta[-1]
        v_med_AP = output_planta_AP[-1]
        v_med_TAB = output_planta_TAB[-1]
    # Datos para la asignacion de Polos:
    u_AP, e_AP =  controlador_AP.actualizar_PI(ref_actual, v_med_AP)
    v_nuevo_AP = planta_AP.actualizar_Planta(u_AP)
    output_planta_AP.append(v_nuevo_AP)
    # Simular un nuevo paso con la referencia actual para MLP:
    u, e = controlador.actualizar_PI(ref_actual, v_med)
    v_nuevo = planta.actualizar_Planta(u) # Salida de la planta v(t)
    # Simular un nuevo paso con la referencia actual para TAB:
    u_TAB, e_TAB = controlador_TAB.actualizar_PI(ref_actual, v_med_TAB)
    v_nuevo_TAB = planta_TAB.actualizar_Planta(u_TAB) # Salida de la planta v(t)
    ##################################
    # Salida del controlador MLP
    output_control.append(u)
    # Salida del error
    output_error.append(e)
    # Salida de la planta
    output_planta.append(v_nuevo)
    ##################################
    # Salida del controlador TAB:
    output_control_TAB.append(u_TAB)
    # Salida del error
    output_error_TAB.append(e_TAB)
    # Salida de la planta
    output_planta_TAB.append(v_nuevo_TAB)


    # Se divide en 10 el eje temporal por el nuevo T = 0.7043
    x_acumulado.append(len(x_acumulado) * T)  # Eje x ajustado con el tiempo de muestreo
    output_cat_MLP = asignacion_categoria(controlador.K_p, controlador.T_i, planta.K, planta.tau_p)
    output_cat_TAB = asignacion_categoria(controlador_TAB.K_p, controlador_TAB.T_i, planta.K, planta.tau_p)
    # Esta función es para la red MLP:
    ind = valor_cat(cat, output_cat_MLP)

    ref.append(ref_actual)
    # Kp_m, tau_p, K, control, planta, error
    union_ = [np.array([[Kp_MLP, Ti_MLP, tau_p, K] + output_control + output_planta + output_error], dtype=float)]

    return output_planta, output_planta_TAB, output_control, output_control_TAB, output_error, output_error_TAB, [Kp_MLP, Ti_MLP, tau_p, K], x_acumulado, union_, output_planta_AP, ref, ind, output_cat_TAB, [Kp_TAB, Ti_TAB, tau_p, K]

# Inicialización del dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simulación del Sistema con Controlador PI"),

    html.Div([
        html.Div([
            html.Label("Ingresar Referencia:"),
            dcc.Input(id='input-referencia', type='number', value=1.0),            
            html.Button('Iniciar', id='boton-iniciar', n_clicks=0, className='btn'),
            html.Button('Pausar', id='boton-detener', n_clicks=0, className='btn'),
            html.Button('Mantener PI-AP', id='boton-mantener', n_clicks=0, className='btn'),
            html.Button('Actualizar PI-AP', id='boton-referencia', n_clicks=0),
        ], style={'margin-bottom': 15}),

        # Ajuste de valores de la Planta
        html.Div([
            html.H4("Ajuste de parámetros de la planta"),
            html.Div([
                html.H5("Valor de K:"),
                dcc.Slider(
                    id='slider-k',
                    min=1,
                    max=20,  # Maximo según dataset es 6
                    step=0.25,
                    value=K,  #21.42 Valor inicial
                    marks={i: f'{i}' for i in range(1, 21, 1)},
                ),
            ], style={'margin-bottom': 15}),

            html.Div([
                html.H5("Valor de tau:"),
                dcc.Slider(
                    id='slider-tau-p',
                    min=1,
                    max=20,  # Maximo según dataset es 3
                    step=0.25,
                    value=tau_p,  # 4.47 Valor inicial
                    marks={i: f'{i}' for i in range(1, 21, 1)},
                ),
            ], style={'margin-bottom': 15}),
        ], style={'margin-bottom': 20}),

        # Sección para mostrar los parámetros
        html.H4("Parámetros del Sistema:"),
        html.Div([
            html.Div([
                html.H5("Planta:"),
                html.P("K: ", id="k-value"),
                html.P("tau: ", id="tau-value"),
            ], style={'display': 'flex', 'flex-direction': 'column', 'width': '40%'}),
            html.Div([
                html.H5("Controlador PI-MLP:"),
                html.P("Kp MLP: ", id="kp-value", dir='ltr'),
                html.P("Ti MLP: ", id="ti-value", dir='ltr'),
            ], style={'display': 'flex', 'flex-direction': 'column', 'width': '60%'}),
            html.Div([
                html.H5("Controlador PI-TAB:"),
                html.P("Kp TAB: ", id="kp-value-tab", dir='ltr'),
                html.P("Ti TAB: ", id="ti-value-tab", dir='ltr'),
            ], style={'display': 'flex', 'flex-direction': 'column', 'width': '60%'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin': 5}),

        # Gráfico de la señal en tiempo real
        dcc.Graph(id='graph-output'),

        # Intervalo para actualizar la gráfica en tiempo real
        dcc.Interval(
            id='interval-component',
            interval=70.43,  # 1000 ms = 1 segundo
            n_intervals=0,
            disabled=True  # Inicia deshabilitado
        )
    ], className='container')
])

# Callback para controlar la simulación, detener y reiniciar
@app.callback(
    [Output('interval-component', 'disabled'),
     Output('graph-output', 'figure'),
     Output('k-value', 'children'),
     Output('tau-value', 'children'),
     Output('kp-value', 'children'),
     Output('ti-value', 'children'),
     Output('kp-value-tab', 'children'),
     Output('ti-value-tab', 'children'),
     Output('boton-iniciar', 'n_clicks'),
     Output('boton-detener', 'n_clicks'),
     Output('boton-referencia', 'n_clicks'),
     Output('boton-mantener', 'n_clicks')],
    [Input('interval-component', 'n_intervals'),
     Input('boton-iniciar', 'n_clicks'),
     Input('boton-detener', 'n_clicks'),     
     Input('boton-referencia', 'n_clicks'),
     Input('boton-mantener', 'n_clicks'),
     Input('slider-k', 'value'),
     Input('slider-tau-p', 'value')],
    [State('input-referencia', 'value')]
)
def actualizar_simulacion(n_intervals, n_iniciar, n_detener_click, n_controlador_AP_click, n_controlador_PI_click, k_value, tau_p_value, referencia):
    global simulacion_activa, ref_actual, trace1, trace2, trace3,trace4, n_detener, n_controlador_AP,  K, tau_p, Kp_MLP, Ti_MLP,Ti_TAB,Kp_TAB, controlador, planta, union, output_control, output_planta, output_error, multiplicador, controlador_AP, planta_AP, n_controlador_PI

    # Actualiza los valores de K y tau_p basados en el slider
    K = k_value
    tau_p = tau_p_value
    Ti_MLP = Ti_MLP
    Kp_MLP = Kp_MLP
    Ti_TAB = Ti_TAB
    Kp_TAB = Kp_TAB
    
    # Si presiona el botón de mantener controlador PI para AP:
    if n_controlador_PI_click > n_controlador_PI:
        print('click mantener', n_controlador_PI_click)
        n_controlador_PI = 0
        mantener_PI()
        asignacion_de_polos()##
        multiplicador = multiplicador + 1
        ref_actual = referencia
        #La función actualizar parametros es para la NN:
        actualizar_parametros(Kp_MLP=Kp_MLP, Ti_MLP=Ti_MLP, Kp_TAB=Kp_TAB, Ti_TAB=Ti_TAB , K=K, tau_p=tau_p, ref=ref_actual)
        return True, {'data': [trace1, trace2, trace3,trace4], 'layout': go.Layout(title="Mantener ganancias",xaxis=dict(title="Tiempo (s)"),yaxis=dict(title="Salida"))}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI
    
    # Si se presiona el botón de referencia, actualizar ref_actual
    if n_controlador_AP_click > n_controlador_AP:
        print('click no mantener' ,n_controlador_AP_click)
        n_controlador_AP = 0
        asignacion_de_polos()
        NO_mantener_PI()        
        multiplicador = multiplicador + 1
        ref_actual = referencia        
        actualizar_parametros(Kp_MLP=Kp_MLP, Ti_MLP=Ti_MLP, Kp_TAB=Kp_TAB, Ti_TAB=Ti_TAB , K=K, tau_p=tau_p, ref=ref_actual)
        return True, {'data': [trace1, trace2, trace3,trace4], 'layout': go.Layout(title="No mantener ganancias",xaxis=dict(title="Tiempo (s)"),yaxis=dict(title="Salida"))}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI
    
    # Si se presiona el botón de detener, pausar la simulación
    if n_detener_click > n_detener:
        simulacion_activa = False
        n_detener = 0#n_detener_click #Reset del contador detener
        return True, {'data': [trace1, trace2, trace3,trace4], 'layout': go.Layout(title="Simulación Pausada",xaxis=dict(title="Tiempo (s)"),yaxis=dict(title="Salida"))}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI

    # Si se presiona el botón de iniciar, activar la simulación
    if n_iniciar > 0:
        # print(n_iniciar)
        multiplicador = multiplicador
        simulacion_activa = True

    # Si la simulación está activa, continuar simulando
    if simulacion_activa:
        #return output_planta, output_planta_TAB, output_control, output_control_TAB, output_error, output_error_TAB, [Kp, Ti, tau_p, K], x_acumulado, union_, output_planta_AP, ref, ind, output_cat_TAB
        output, out_planta_tab, control, out_control_tab, error, out_error_tab, par, x, union, out_AP, ref, ind, out_cat_tab, par_2 = simular_sistema(ref_actual, K, tau_p)
        # ----
        # Para modelos con categorias TabTransformer:
        valor_predecir_TAB = [np.array([par_2 + out_control_tab[-142:] + out_planta_tab[-142:] + out_error_tab[-142:]], dtype=float), np.array([[out_cat_tab[0], out_cat_tab[1]]], dtype=object)]
        valor_predecir_MLP = ind + par + control[-142:] + output[-142:] + error[-142:]
        
        # print(valor_predecir_MLP)
        # Crear la gráfica
        trace1 = go.Scatter(
            x=x, # Eje temporal 
            y= output, # Señal de la planta 
            mode='lines',#lines+markers
            name='PI-MLP'
        )
        trace2 = go.Scatter(
            x=x, # Eje temporal 
            y= out_AP, # Señal de la planta AP
            mode='lines',#lines+markers
            name='PI-AP'
        )
        trace4 = go.Scatter(
            x=x, # Eje temporal 
            y= ref, # Señal de la planta AP
            mode='lines',#lines+markers
            name='Ref',
            line=dict(dash='dash')
        )
        trace3 = go.Scatter(
            x=x, # Eje temporal 
            y= out_planta_tab, # Señal de la planta AP
            mode='lines',#lines+markers
            name='PI-TAB'
        )
        layout = go.Layout(
            title="Salida del Sistema con Controlador PI",
            xaxis=dict(title="Tiempo (s)", range=[max(0, x[-1] - 20), x[-1]]), # Eje x ahora en segundos
            yaxis=dict(title="Salida"),
        )
        if len(x) >= 142*multiplicador:
            print('-'*50)
            ultimos_datos_TAB = valor_predecir_TAB
            # print(f'Datos de la union para TAB: {ultimos_datos_TAB}')
            # print(f'Datos de la unión MLP: {valor_predecir_MLP}')
            ### MODELO MLP:
            valor_predecir_MLP = np.array(valor_predecir_MLP).reshape(-1,1)
            # Escalamos los datos:
            ultimos_datos_MLP = scaler.fit_transform(valor_predecir_MLP).reshape(1,-1)
            # Desactivamos la simulación:
            simulacion_activa = False
            # Datos escalados:
            # print(f'DATOS ESCALADOS: {ultimos_datos_MLP}')      
            predictions_MLP = MLP_model.predict(x=ultimos_datos_MLP)
            ### MODELO TAB:            
            predictions_TAB = TAB_model.predict(x=ultimos_datos_TAB)
            # print(f'Prediccion MLP: {predictions_MLP}')
            # print('Predicción MLP:')
            # print(f'Kp: {predictions_MLP[0][0][0]}')
            # print(f'Ti: {predictions_MLP[1][0][0]}')
            Kp_MLP = predictions_MLP[0][0][0]
            Ti_MLP = predictions_MLP[1][0][0]
            Kp_TAB = predictions_TAB['Ganancia_Kp'][0]
            Ti_TAB = predictions_TAB['Periodo_Ti'][0]
            print(f'Predicción MLP\n\tKp MLP: {Kp_MLP}--Tipo: {type(Kp_MLP)}\n\tTi MLP: {Ti_MLP}')
            print(f'Predicción TAB\n\tKp TAB: {Kp_TAB}\n\tTi TAB: {Ti_TAB}')
            # print(f'Valores NN:\n\tKp: {predictions_2[0][0]}\n\tTi: {predictions_2[0][1]}\n\tCategoria_1: {cat_1}\n\tCategoria_2: {cat_2}')
            # print(f'Valores NN:\n\tKp: {Kp}\n\tTi: {Ti}\n\tCategoria_1: {cat_1}\n\tCategoria_2: {cat_2}')
            print('-'*50)
            print('-'*50)

            return True, {'data': [trace1, trace2, trace3,trace4], 'layout': go.Layout(title="Obteniendo datos",xaxis=dict(title="Tiempo (s)", range=[max(0, x[-1] - 20), x[-1]]),yaxis=dict(title="Salida"))}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI
        
        return False, {'data': [trace1, trace2, trace3,trace4], 'layout': layout}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI

    return True, {'data': [trace1, trace2, trace3,trace4], 'layout': go.Layout(title="Inicie la Simulación",xaxis=dict(title="Tiempo (s)"),yaxis=dict(title="Salida"))}, f"K: {K}", f"tau: {tau_p}", f"Kp MLP: {Kp_MLP}", f"Ti MLP: {Ti_MLP}", f"Kp TAB: {Kp_TAB}", f"Ti TAB: {Ti_TAB}", 0, 0, n_controlador_AP, n_controlador_PI

# Ejecutar el servidor local
if __name__ == '__main__':
    app.run_server(debug=True)