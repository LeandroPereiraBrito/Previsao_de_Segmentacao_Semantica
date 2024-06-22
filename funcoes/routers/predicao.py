from joblib import load
import warnings
try:
    import config as cf
except:
    import funcoes.routers.config as cf


# ==============================================================
#  Função que carrega o modelo
# ==============================================================
def modelo(path):
    warnings.filterwarnings("ignore")
    model = None
    with open(path, 'rb') as pkl:
        model = load(pkl)
    return model

# ==============================================================
#    Fução que mapeia o nome correto dos modelos existentes
# ==============================================================
def mapa(reg):
    municipio = {1:"Paracatu_",
                 2:"Lucas_do_Rio_Verde_",
                 3:"Parauapebas_",
                 4:"São_Paulo_"
    }
    return municipio[reg]

# ==============================================================
# Fução para prever os desmatamentos
# ==============================================================
def desmatamento(reg):
    lista = ["des_NO", "des_NE", "des_SO", "des_SE"]
    saida = []

    for metrica in lista:
        warnings.filterwarnings("ignore")
        modelo_path = f"{cf.dir_list['2.2']}{mapa(reg)}{metrica}.pkl"
        arima = modelo(modelo_path)  # Carrega o modelo ARIMA treinado
        previsoes = arima.predict(n_periods=4)
        arr = [int(round(x,2)) for x in previsoes]
        saida.append(arr)
    correto = []
    for i in range(4):
        correto.append([int(reg),int(i+1),saida[0][i], saida[1][i], saida[2][i], saida[3][i]])
    return correto


# =======================================================================================
#  Fução que prever os numero de preservação para a proxima janela
# =======================================================================================
def preservacao(reg):
    lista = ["mat_NO", "mat_NE", "mat_SO", "mat_SE"]
    saida = []

    for metrica in lista:
        warnings.filterwarnings("ignore")
        modelo_path = f"{cf.dir_list['2.2']}{mapa(reg)}{metrica}.pkl"
        arima = modelo(modelo_path)  # Carrega o modelo ARIMA treinado
        previsoes = arima.predict(n_periods=4)
        arr = [int(round(x,2)) for x in previsoes]
        saida.append(arr)
        correto = []
    for i in range(4):
        correto.append([saida[0][i],saida[1][i],saida[2][i],saida[3][i]])
    return correto


# ==============================================================
#  Fução que preve a populaçãa para a proxima janela
# ==============================================================
def populacao(reg):
    modelo_path = f"{cf.dir_list['2.2']}{mapa(reg)}populacao.pkl"
    warnings.filterwarnings("ignore")
    arima = modelo(modelo_path)
    previsoes = arima.predict(n_periods=4)
    saida = [int(round(x,2)) for x in previsoes]
    return saida