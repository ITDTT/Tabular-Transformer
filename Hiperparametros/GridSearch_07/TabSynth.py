'''
Código de la red neuronal fue realizado en base al original de ai4water: https://ai4water.readthedocs.io/en/latest/
Código de la generación de datos, creada por Ignacio Carvajal.
'''
import numpy as np
import pandas as pd
import os
import math
import control as ctr
import random
from itertools import count, takewhile, product
import matplotlib.pyplot as plt
from scipy import signal

#Creamos clases para nuestro Controlador PI y la Planta desconocida de Primer Orden.
class PIControlador:
    def __init__(self, Kp,Ti, T=0.07): #Ganancia proporcional, Tiempo integral y Tiempo de muestreo
        self.K_p = Kp
        self.T = T
        self.T_i = Ti
        self.u_prev = 0 #Inicializamos en 0
        self.e_prev = 0

    def actualizar_PI(self, ref, medida):
        error = ref - medida
        #Señal de control u(k):
        u = (self.u_prev + (self.K_p + ((self.K_p*self.T) / (2*self.T_i)))*error - (self.K_p - ((self.K_p*self.T) / (2* self.T_i)))*self.e_prev)
        #Actualización de los valores previos para la siguiente iteración:
        self.u_prev = round(u,3)
        self.e_prev = round(error,3)
        # print(f'Tipo u: {type(u)}\nTipo error: {type(error)}')
        #Entregamos los valores de la señal de control y el error.
        lista = [u, error]
        return lista

class PlantaPrimerOrden:
    '''
        Planta de primer orden de la forma:
        G(s) = K/(tau_p*s + 1)
    '''
    def __init__(self, K, tau_p, T=0.07):
        self.K = K
        self.tau_p = tau_p
        # self.a = a
        self.T = T
        self.v_prev = 0

    def actualizar_Planta(self, u_prev):
        exp_term = np.exp(-self.T/self.tau_p)
        # exp_term = np.exp(-self.a*self.T) # Antiguo: VA CON K/(s+a)
        #Salida de la planta (nueva: K/(tau*s + 1))
        v = (self.K)*(1 - exp_term)*u_prev + exp_term*self.v_prev
        # Salida de la planta (antigua: K/(s+a)):
        # v = (self.K/self.a)*(1 - exp_term)*u_prev + exp_term*self.v_prev
        #Actualizamos el valor de v para la siguiente iteración
        self.v_prev = v
        return v
    
class Combinaciones:
    def __init__(self):
        None
    
    def frange(self, init, fin, paso):
        return takewhile(lambda x: x <= fin, count(init, paso))

    def combinaciones(self, shuffle=False):
        self.shuffle = shuffle
        K_m = self.frange(1, 6, 1) #13------#47 datos-->self.frange(0.25, 11.75, 0.25)... self.frange(0.25, 23.75, 0.25)
        tau_m = self.frange(1, 3, 0.5) #14-----#47 datos-->self.frange(0.25, 6, 0.125)...self.frange(0.25, 12, 0.125)
        
        Kp_m = self.frange(0.1, 3.8, 0.205) #0.20555555555555557
        Ti_m = self.frange(0.5, 1.35, 0.097)#0.047222222222222276
        combinacion = list(product(K_m, tau_m, Kp_m, Ti_m))
        if self.shuffle == True:
            random.shuffle(combinacion)
        return combinacion

class Categoria:
    '''
    Valores nominales:
        K_nominal = 5.5
        tau_nominal = 3.0
    CONDICIONES DESCONOOCIDAS:
        CD0: Si K y tau_p < a valores nominales.
        CD1: None
        CD2: Si K y tau_p > a valores nominales.
    '''
    def __init__(self, K, tau_p, K_nominal=3.0,tau_nominal=1.5):
        self.K = K
        self.K_nominal = K_nominal
        self.tau_p = tau_p
        self.tau_nominal = tau_nominal

    def categoria_pendiente(self): #Categoria de la pendiente
        if self.K < self.K_nominal:
            if self.tau_p > self.tau_nominal or self.tau_p == self.tau_nominal:
                return "SUBIDA"#"MOVIMIENTO"#
            else:
                return "CD0" #CONDICION DESCONOCIDA 0 -- "MOVIMIENTO"#
        elif self.K == self.K_nominal:
            if self.tau_p > self.tau_nominal:
                return "SUBIDA"#"MOVIMIENTO"#
            elif self.tau_p < self.tau_nominal:
                return  "BAJADA"# "MOVIMIENTO"#
            elif self.tau_p == self.tau_nominal:
                return "LINEA RECTA"# "MOVIMIENTO" #
            else:
                return "CD1" #CONDICION DESCONOCIDA 1 -- "MOVIMIENTO"#
        elif self.K > self.K_nominal:
            if self.tau_p < self.tau_nominal or self.tau_p == self.tau_nominal:
                return "BAJADA"#"MOVIMIENTO"#
            else:
                return "CD2" #CONDICION DESCONOCIDA 2 -- "MOVIMIENTO"#
        else:
            return "CD3" #CONDICION DESCONOCIDA 3 -- "MOVIMIENTO"#
    
    def categoria_(self, PT):
        self.PT = PT
        if self.PT <= 4.0:
            return "RAPIDA" #"AVANCE"#
        elif self.PT > 4.0 and self.PT <= 5.0:
            return "MEDIA" #"AVANCE"#
        elif self.PT > 5.0:
            return "LENTA"# "AVANCE"#


class AsignacionPolos:
    def __init__(self, K, tau_p, T):
        self.K = K
        self.tau_p = tau_p
        self.T = T
                
    def parametros_controladorAP(self):
        '''Retorna los parámetros del controlado PI:
            - Ganancia proporcional Kp.
            - Tiempo Integrativo. 
            return: diccionario.       
        '''
        FA = 0.8260850546     
        Omega_n = 4/(FA*5) # Al 2% de establecimiento, Tss = 5 
        # Parámetros Kp y Ti
        Kp = (2*FA*Omega_n*self.tau_p - 1) / self.K
        Ti = (self.K*Kp) /(Omega_n**2 * self.tau_p)
        num_cd = [Kp*(1+self.T/(2*Ti)), -Kp*(1-self.T/(2*Ti))]
        den_cd = [1, -1]
        Cd = ctr.TransferFunction(num_cd,den_cd, self.T)
        e = np.exp(-self.T/self.tau_p)
        num_gd = self.K*(1 -e)
        den_gd = [1, -e]
        Gd = ctr.TransferFunction(num_gd,den_gd, self.T)
        #Lazo cerrado:
        H = ctr.feedback(Cd*Gd,1)
        info = ctr.step_info(H)
        Valor_ess = info['SteadyStateValue']# Valor estado estable
        Tiempo_estable = info['SettlingTime']# Tiempo de asentamiento
        Overshoot = info['Overshoot']
        return {'Kp': Kp, 'Ti': Ti, 'Tiempo_asentamiento': Tiempo_estable, 'Valor_estado_estacionario': Valor_ess, 'Sobrepaso': Overshoot}


class PRBS:
    '''Con 142 muestras y un T=0.07 --> 10 segundos de gráfica.'''
    def __init__(self, nstep=142):
        '''
        Constructor de la clase.
        Args:
            nstep: Número de muestras.
        '''
        self.nstep = nstep

    def Ref_PRBS(self):
        a_range = [0,2]
        a = np.random.rand(self.nstep) * (a_range[1]-a_range[0]) + a_range[0]
        b_range = [2, 10]
        b = np.random.rand(self.nstep) *(b_range[1]-b_range[0]) + b_range[0]
        b = np.round(b)
        b = b.astype(int)

        b[0] = 0

        for i in range(1,np.size(b)):
            b[i] = b[i-1]+b[i]

        i=0
        random_signal = np.zeros(self.nstep)
        while b[i]<np.size(random_signal):
            k = b[i]
            random_signal[k:] = a[i]
            i=i+1

        # PRBS
        a = np.zeros(self.nstep)
        j = 0
        while j < self.nstep:
            a[j] = 1
            a[j+1] = 0
            j = j+2

        i=0
        prbs = np.zeros(self.nstep)
        while b[i]<np.size(prbs):
            k = b[i]
            prbs[k:] = a[i]
            i=i+1
        return prbs

class Grafico:
    def __init__(self, referencia,tiempo, señal_u, output_planta, Kp, Ti, K, tau_p, tipo):
        self.referencia = referencia
        self.tiempo = tiempo
        self.señal_u = señal_u
        self.output_planta = output_planta
        self.Kp = Kp
        self.Ti = Ti
        self.K = K
        self.tau_p = tau_p
        self.tipo = tipo
        #Si "tipo es True":
        if self.tipo == 1 or self.tipo == 2:
            if self.tipo == 1:
                self.GrafRef()
            if self.tipo == 2:
                self.GrafRefAux()
        else:
            self.GrafRefOne()

    def GrafRefAux(self):
        time = np.linspace(0, self.tiempo, len(self.referencia))
        señal1 = self.referencia
        señal2 = self.señal_u
        señal3 = self.output_planta
        #Creación de la gráfica
        plt.figure(figsize=(10,5))
        plt.plot(time, señal1, label='Referencia')
        plt.plot(time, señal2, label='Señal de Control (u)')
        plt.plot(time, señal3, label='Salida de la Planta (v)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def GrafRef(self):
        time = np.linspace(0, self.tiempo, len(self.referencia))
        señal1 = self.referencia
        # señal2 = self.señal_u
        señal3 = self.output_planta
        #Creación de la gráfica
        plt.figure(figsize=(10,5))
        plt.plot(time, señal1, label='Referencia')
        # plt.plot(time, señal2, label='Señal de Control (u)')
        plt.plot(time, señal3, label='Salida de la Planta (v)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def GrafRefOne(self):
        # Crear un vector de tiempo
        time = np.linspace(0, self.tiempo, len(self.referencia))
        etiquetas_leyenda = [f"Kp: {self.Kp}", f"Ti: {self.Ti}", f"K: {self.K}", f"tau_p: {self.tau_p}"]
        plt.figure(figsize=(12, 8))
        # Gráfico de la referencia
        plt.subplot(3, 1, 1)
        plt.plot(time, self.referencia, label='Referencia (r)')
        plt.title('Referencia (r)')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        plt.grid(True)

        # Gráfico de la señal de control
        plt.subplot(3, 1, 2)
        # plt.plot(time, self.señal_u, label='Señal de control (u)', color='orange')# ORIGINAL
        plt.plot(time, self.señal_u, label=[etiquetas_leyenda[0], etiquetas_leyenda[1]], color='orange')
        plt.title('Señal de control (u)')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        plt.grid(True)

        # Gráfico de la salida de la planta
        plt.subplot(3, 1, 3)
        # plt.plot(time, self.output_planta, label='Salida de la planta (v)', color='green')
        plt.plot(time, self.output_planta, label=[etiquetas_leyenda[2], etiquetas_leyenda[3]], color='green')
        plt.title('Salida de la planta (v)')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

class Grafico_2:
    def __init__(self, tiempo, referencia ,señal1, señal2,señal3,señal4,señal5,señal6, tipo):
        self.referencia = referencia
        self.tiempo = tiempo
        self.señal1 = señal1
        self.señal2 = señal2
        self.señal3 = señal3
        self.señal4 = señal4
        self.señal5 = señal5
        self.señal6 = señal6
        self.tipo = tipo
        #Si "tipo es True":
        if self.tipo == 1:
            if self.tipo == 1:
                self.GrafSeñales()
        else:
            None

    def GrafSeñales(self):
        time = np.linspace(0, self.tiempo, len(self.referencia))
        ref = self.referencia
        señal1 = self.señal1
        señal2 = self.señal2
        señal3 = self.señal3
        señal4 = self.señal4
        señal5 = self.señal5
        señal6 = self.señal6
        #Creación de la gráfica
        plt.figure(figsize=(10,5))
        plt.plot(time, ref, label='Referencia')
        plt.plot(time, señal1, label='Señal 1')
        plt.plot(time, señal2, label='Señal 2')
        plt.plot(time, señal3, label='Señal 3')
        plt.plot(time, señal4, label='Señal 4')
        plt.plot(time, señal5, label='Señal 5')
        plt.plot(time, señal6, label='Señal 6')
        plt.legend()
        plt.grid(True)
        plt.show()

class Grafico_1:
    def __init__(self, referencia,tiempo,output_planta):
        self.referencia = referencia
        self.tiempo = tiempo
        self.output_planta = output_planta

    def GrafRef(self):
        time = np.linspace(0, self.tiempo, len(self.referencia))
        señal1 = self.referencia
        # señal2 = self.señal_u
        señal3 = self.output_planta
        #Creación de la gráfica
        plt.figure(figsize=(10,5))
        plt.plot(time, señal1, label='Referencia')
        # plt.plot(time, señal2, label='Señal de Control (u)')
        plt.plot(time, señal3, label='Salida de la Planta (v)')
        plt.legend()
        plt.grid(True)
        plt.show()

class Metricas:
    '''Métricas:
        Argumento: Tau de la planta.
        Retorna: 
            Tiempo de Asentamiento (Ts)
            Tiempo de Levantamiento (Tr)
    '''
    def __init__(self, K, tau_p, Kp, Ti, T, metric):
        self.K = K
        self.tau_p = tau_p
        self.Kp = Kp
        self.Ti = Ti
        self.T = T
        self.metric = str(metric)

    def metrica(self):
        #####################
        # Las funciones de transferencia están en tiempo Discreto:
        num_cd = [self.Kp*(1+self.T/(2*self.Ti)), -self.Kp*(1-self.T/(2*self.Ti))]
        den_cd = [1, -1]
        Cd = ctr.TransferFunction(num_cd,den_cd, self.T)
        e = np.exp(-self.T/self.tau_p)
        num_gd = self.K*(1 -e)
        den_gd = [1, -e]
        Gd = ctr.TransferFunction(num_gd,den_gd, self.T)
        #Lazo cerrado:
        H = ctr.feedback(Cd*Gd,1)
        #print(f'Funciones de transferencias {_}\nPlanta: {Gd}\nControlador: {Cd}\nLazo carrado: {H}')
        info = ctr.step_info(H)
        PT = info[self.metric] #PeakTime
        PT = np.round(PT,5)
        #####################
        #Tiempo de Asentamiento: Al 2%.
        #self.Ts = 4/(self.chi*self.omega_n)
        # Tr = 2.2*self.tau_p
        return PT#self.Ts


'''----------------------------------------------'''
'''-------------Tabular Transformer--------------'''
'''----------------------------------------------'''
import tensorflow as tf
from keras.layers import StringLookup
from tensorflow.keras.regularizers import l2
layers = tf.keras.layers
Dense = tf.keras.layers.Dense
Layer = tf.keras.layers.Layer
activations = tf.keras.activations
K = tf.keras.backend
constraints = tf.keras.constraints
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
from typing import Union, List

def gen_cat_vocab(
        data,
        cat_columns:list = None,
)->dict:

    if data.ndim != 2:
        raise TypeError('Expected a 2-dimensional dataframe or array')

    assert isinstance(data, pd.DataFrame)

    vocab = {}
    if cat_columns:
        for feature in cat_columns:
            vocab[feature] = sorted(list(data[feature].unique()))
    else:
        for col in data.columns:
            if data.loc[:, col].dtype == 'O':
                vocab[col] = sorted(list(data.loc[:, col].unique()))

    return vocab

class CatEmbeddings(tf.keras.layers.Layer):
    def __init__(
            self,
            vocabulary:dict,
            embed_dim:int = 32,
            lookup_kws:dict = None,
            name='Embedding_Categorial'
    ):
        super(CatEmbeddings, self).__init__(name=name)

        self.vocabulary = vocabulary
        self.embed_dim = embed_dim
        self.lookup_kws = lookup_kws
        self.lookups = {}
        self.embedding_lyrs = {}
        self.feature_names = []

        _lookup_kws = dict(mask_token=None,
                num_oov_indices=0,
                output_mode="int")
        
        if lookup_kws is not None:
            _lookup_kws.update(lookup_kws)

        for feature_name, vocab in vocabulary.items():

            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocab,
                **_lookup_kws
            )

            self.lookups[feature_name] = lookup

            embedding = tf.keras.layers.Embedding(
                input_dim=len(vocab), output_dim=embed_dim
            )

            self.embedding_lyrs[feature_name] = embedding

            self.feature_names.append(feature_name)

    def get_config(self)->dict:
        config = {
            "lookup_kws": self.lookup_kws,
            "embed_dim": self.embed_dim,
            "vocabulary": self.vocabulary
        }
        return config
    
    def call(self, inputs):
        encoded_features = []
        for idx, feat_name in enumerate(self.feature_names):
            feat_input = inputs[:, idx]
            lookup = self.lookups[feat_name]
            encoded_feature = lookup(feat_input)

            embedding = self.embedding_lyrs[feat_name]
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_features.append(encoded_categorical_feature)

        cat_embeddings = tf.stack(encoded_features, axis=1)

        return cat_embeddings
    
class NumericalEmbeddings(tf.keras.layers.Layer):

    def __init__(
            self,
            num_features,
            emb_dim
    ):

        self.num_features = num_features
        self.emb_dim = emb_dim
        super(NumericalEmbeddings, self).__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        # features, n_bins, emb_dim
        self.linear_w = tf.Variable(
            initial_value=w_init(
                shape=(self.num_features, 1, self.emb_dim), dtype='float32'
            ), trainable=True, name="NumEmbeddingWeights")

        # features, n_bins, emb_dim
        self.linear_b = tf.Variable(
            w_init(
                shape=(self.num_features, 1), dtype='float32'
            ), trainable=True, name="NumEmbeddingBias")
        return

    def get_config(self)->dict:
        config = {
            "num_features": self.num_features,
            "emb_dim": self.emb_dim
        }
        return config

    def call(self, X):
        embs = tf.einsum('f n e, b f -> bfe', self.linear_w, X)
        embs = tf.nn.relu(embs + self.linear_b)
        return embs
    
class Transformer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_heads: int = 4,
            embed_dim: int = 64,
            dropout=0.1,
            post_norm: bool = True,
            prenorm_mlp: bool = False,
            num_dense_lyrs: int = 1,
            seed: int = 313,
            name='TransformerxN',
            *args,
            **kwargs
    ):
        super(Transformer, self).__init__(name=name,*args, **kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.post_norm = post_norm
        self.prenorm_mlp = prenorm_mlp
        self.seed = seed

        assert num_dense_lyrs <= 2
        self.num_dense_lyrs = num_dense_lyrs

        # Bloque 1: Multi-Head Attention:
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout
        )
        # Bloque 2: Dropout_1 después de la Multi-Head Attention:
        self.dropout_1 = tf.keras.layers.Dropout(0.4, seed=self.seed)
        # Bloque 3: Add_1 para salida de la capa de atención y entrada de datos (inputs_cat)
        self.Add_1 = tf.keras.layers.Add()
        # Bloque 4: Normalización 1
        self.Norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Bloque 5: Feed-Forward Network
        self.ffn = self._make_mlp(name='FFN')
        # Bloque 6: Dropout_2
        self.dropout_2 = tf.keras.layers.Dropout(0.4, seed=self.seed)
        # Bloque 7: Add_2 para la salida de la capa FNN y salida de Norm_1
        self.Add_2 = tf.keras.layers.Add()
        # Bloque 8: Normalización 2 para la salida de Add_2
        self.Norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)        

    def _make_mlp(self,
                  name=None):
        lyrs = []

        if self.prenorm_mlp:
            lyrs += [tf.keras.layers.LayerNormalization(epsilon=1e-6)]

        lyrs += [
            tf.keras.layers.Dense(self.embed_dim, activation=tf.keras.activations.gelu,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.4, seed=self.seed),
        ]

        if self.num_dense_lyrs > 1:
            lyrs += [tf.keras.layers.Dense(self.embed_dim,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.01))]

        return tf.keras.Sequential(lyrs)

    def get_config(self) -> dict:
        config = {
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dropout": self.dropout,
            "post_norm": self.post_norm,
            "prenorm_mlp": self.prenorm_mlp,
            "seed": self.seed,
            "num_dense_lyrs": self.num_dense_lyrs
        }
        return config

    def __call__(self, inputs, training=None, *args, **kwargs):
        # La salida de "Column Embedding", entra--> MultiHeadAttention.        
        attention_output, att_weights = self.att(
            inputs, inputs, return_attention_scores=True
        )
        # Capa de dropout_1 después de multi-head attention
        attention_output = self.dropout_1(attention_output, training=training)
        # Capa de Add_1, salida de la capa de Attention y la de la Normalization 1
        attention_output = self.Add_1([inputs, attention_output])
        # Caoa de Norm_1:
        attention_output = self.Norm_1(attention_output)
        # Capa Feed forward:
        feedforward_output = self.ffn(attention_output)
        # Aplicar dropout después de la capa feedforward
        feedforward_output = self.dropout_2(feedforward_output, training=training)
        # Capa de Add 2:
        outputs = self.Add_2([feedforward_output, attention_output])
        # Normalizar después de todo. es la Normalization 2 (dejar en True):
        if self.post_norm:
            return self.Norm_2(outputs), att_weights

        return outputs, att_weights

class TransformerBlocks(tf.keras.layers.Layer):
    def __init__(
            self,
            num_blocks:int,
            num_heads:int,
            embed_dim:int,
            name:str = "TransformerBlocks",
            **kwargs
    ):
        super(TransformerBlocks, self).__init__(name=name)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.blocks = []
        for n in range(num_blocks):
            self.blocks.append(Transformer(num_heads, embed_dim, **kwargs))

    def get_config(self)->dict:
        config = {
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim
        }
        return config
    
    def __call__(self, inputs, *args, **kwargs):

        attn_weights_list = []
        for transformer in self.blocks:
            inputs, attn_weights = transformer(inputs)
            attn_weights_list.append(tf.reduce_sum(attn_weights[:, :, 0, :]))

        importances = tf.reduce_sum(tf.stack(attn_weights_list), axis=0) / (
                self.num_blocks * self.num_heads)

        return inputs, importances
    
class TabTransformer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_numeric_features: int,
            cat_vocabulary: dict,
            hidden_units=32,
            lookup_kws:dict=None,
            num_heads: int = 4,
            depth: int = 4,
            dropout: float = 0.4,
            num_dense_lyrs: int = 2,
            prenorm_mlp: bool = True,
            post_norm: bool = True,
            final_mlp_units = 16,
            num_outputs: int = 2,
            final_mpl_activation:str = "linear",
            seed: int = 313,
            l2_lambda: float = 0.001,
            name:str='TabTransformer',
            *args, **kwargs
    ):
        super(TabTransformer, self).__init__(name=name,*args, **kwargs)

        self.cat_vocabulary = cat_vocabulary
        self.num_numeric_inputs = num_numeric_features
        self.hidden_units = hidden_units
        self.lookup_kws = lookup_kws
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.final_mlp_units = final_mlp_units
        self.num_outputs = num_outputs
        self.final_mpl_activation = final_mpl_activation
        self.seed = seed
        self.l2_lambda = l2_lambda

        # Bloque 00: Column Embedding:
        self.cat_embs = CatEmbeddings(
            vocabulary=cat_vocabulary,
            embed_dim=hidden_units,
            lookup_kws=lookup_kws
        )
        # Bloque 00: Layer Normalization:
        self.lyr_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Bloque 01: Transformer xN (depth)
        self.transformers = TransformerBlocks(
            embed_dim=hidden_units,
            num_heads=num_heads,
            num_blocks=depth,
            num_dense_lyrs=num_dense_lyrs,
            post_norm=post_norm,
            prenorm_mlp=prenorm_mlp,
            dropout=dropout, # Dropout de la capa de Attention.
            seed=seed
        )
        # Aplanamos las salidas de la úlima normalización
        self.flatten = tf.keras.layers.Flatten()
        # Concadenamos la salida aplanada + layer normalization (numerica)
        self.concat = tf.keras.layers.Concatenate()
        # Bloque MLP:
        self.mlp = self.create_mlp(
            activation=self.final_mpl_activation,
            normalization_layer=tf.keras.layers.BatchNormalization(),
            l2_lambda = l2_lambda,
            name="MLP",
            num_outputs=num_outputs
        )

    # Multi-Layer Perceptron
    def create_mlp(
            self,
            activation,
            normalization_layer,
            l2_lambda,
            name=None,
            num_outputs = 2
    ):
        hidden_units = [self.final_mlp_units] if isinstance(self.final_mlp_units, int) else self.final_mlp_units

        mlp_layers = []
        for units in hidden_units:
            mlp_layers.append(normalization_layer)
            mlp_layers.append(tf.keras.layers.Dense(units,
                                                    activation=activation,
                                                    kernel_regularizer=l2(l2_lambda)
                                                    ))
            mlp_layers.append(tf.keras.layers.Dropout(self.dropout, seed=self.seed))

        mlp_layers.append(tf.keras.layers.Dense(num_outputs,
                                                activation=activation,
                                                kernel_regularizer = l2(l2_lambda)
                                                ))
        return tf.keras.Sequential(mlp_layers, name=name)

    def __call__(self, inputs:list , *args, **kwargs):
        """
        inputs :
            list of 2. The first tensor is numerical inputs and second
            tensor is categorical inputs
        """
        num_inputs = inputs[0]
        cat_inputs = inputs[1]
        # Column Embedding:
        cat_embs = self.cat_embs(cat_inputs)
        # Bloque Transformer:
        transformer_outputs, imp = self.transformers(cat_embs)
        # Aplanamos "outputs == transfoemr_outputs":
        flat_transformer_outputs = self.flatten(transformer_outputs)
        # Normalización de las entradas numéricas:
        num_embs = self.lyr_norm(num_inputs)
        # Concatenamos la salida aplanada y la normalización de los valores numéricos:
        x = self.concat([num_embs, flat_transformer_outputs])
        # Ingresamos la salida concadenada al MLP:
        outputs = self.mlp(x)
        # Obtenemos las dos salidas necesarias:
        # Kp y Ti
        gananacia_kp = tf.keras.layers.Lambda(lambda x: x[:, 0], name='Ganancia_Kp')(outputs)
        periodo_ti = tf.keras.layers.Lambda(lambda x: x[:, 1], name='Periodo_Ti')(outputs)

        return {'Ganancia_Kp': gananacia_kp, 'Periodo_Ti': periodo_ti}, imp
    
# Función para graficar y guardar cada gráfico por separado
def save_loss_plots(history, save_dir):
    # Pérdida total
    plt.figure()
    plt.plot(history.history['loss'], label='Loss (Train)', color='red')
    plt.plot(history.history['val_loss'], label='Loss (Validation)', color='orange', linestyle='--')
    plt.title('Total Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "total_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Pérdida de Kp
    plt.figure()
    plt.plot(history.history['Ganancia_Kp_loss'], label='Kp Loss (Train)', color='blue')
    plt.plot(history.history['val_Ganancia_Kp_loss'], label='Kp Loss (Validation)', color='cyan', linestyle='--')
    plt.title('Kp Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "kp_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Pérdida de Ti
    plt.figure()
    plt.plot(history.history['Periodo_Ti_loss'], label='Ti Loss (Train)', color='green')
    plt.plot(history.history['val_Periodo_Ti_loss'], label='Ti Loss (Validation)', color='lime', linestyle='--')
    plt.title('Ti Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "ti_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()