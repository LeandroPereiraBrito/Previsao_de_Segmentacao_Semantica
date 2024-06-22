# Manipulação de arquivos
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import random
# controle de data e tempo
import time
from datetime import datetime
# Manipulação de imagens
import cv2
import imageio
from IPython.display import Image
from PIL import Image as pil
# Ciencia de dados
import numpy as np
import pandas as pd
# Apresentação
import collections
# Bloquear avisos de cluster
import warnings
warnings.filterwarnings("ignore")

# Funções sklearn

# funções de cluster

# Superte vector machine

# Salvar os modelos
dir_Raiz = os.path.dirname(os.path.realpath(__file__)).replace(chr(92), "/")
dir_Raiz  = dir_Raiz.replace("funcoes/routers","")
# Gerar a lista de diretórios e arquivos
dir_list = {'2': f'{dir_Raiz}dataSets_Lab/Imagens',
            '2.1.1': f'{dir_Raiz}dataSets_Lab/Imagens/analise/',
            '2.1.2': f'{dir_Raiz}dataSets_Lab/Imagens/treino/',
            '2.1.3': f'{dir_Raiz}dataSets_Lab/Imagens/validacao/',
            '2.2': f'{dir_Raiz}dataSets_Lab/Modelos/',
            '2.3': f'{dir_Raiz}dataSets_Lab/Logs/',
            '2.4': f'{dir_Raiz}dataSets_Lab/Saidas/',
            '2.5': f'{dir_Raiz}dataSets_Lab/Saidas/imagens/'
            }

model_name = {'1': 'espectro', '2': 'menoDesmat',
              '3': 'maioDesmat', '4': 'ltms.pkl', '5': 'svm'}
saidas_name = {'1': 'analise.csv', '2': 'load_image.pkl',
               '3': 'imagens.pkl', '4': 'dataSet_complemetar.csv',
               '5': 'dataset_desmatamento.pkl',
               '6': 'previsoes.pkl'}
logs_name = {'1': 'ana_classificador.csv'}
