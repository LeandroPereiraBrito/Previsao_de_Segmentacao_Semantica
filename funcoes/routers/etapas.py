import numpy as np

import funcoes.routers.config as cf
import funcoes.routers.funcoes as fc
import funcoes.routers.predicao as prev
import pandas as pd
import os
# =======================================================================================================================
#                    Incio da Fase 1 classificar pixels das imagens  de analise.
#
# Objetico gerar um classificador de pixels para as imgens.
# processo 1: motar dataset com as imagens do diretório
# processo 2: Montar um classificar por uma pequena amostra da população para gerar as classes
# processo 3: Aplicar classificador em  data set e gerar modelo mais performaticm em SVM
# =======================================================================================================================


def etapa1():
    lista = []
    # Pega o caminho das imagens
    files_path = fc.getFiles(cf.dir_list['2.1.1'])
    # fragmentar o arquivo para extrair informações
    for item in files_path:
        # Sepra caminho do nome da imagem
        path, info = item
        # Pega Ano, regisão, latitude e longitude
        reg, ano = info.split('.')[0].split("_")
        # Acrescentar intem na lista
        lista.append([ano, reg, path])
    # Gerar um datasets para facilitar o acesso de informações
    df_lista = cf.pd.DataFrame(lista, columns=['ano', 'reg', 'path'])
    # Gravar dataSets da leitura das imagens

    if cf.os.path.exists(cf.dir_list['2.4']+cf.saidas_name['3']):
        df_imagens = pd.read_pickle(cf.dir_list['2.4']+cf.saidas_name['3'])
    else:
        df_imagens = fc.generato_ClassificadorPX()
        df_imagens.to_pickle(cf.dir_list['2.4']+cf.saidas_name['3'])
    #  Gera a lista de imagens candidatas para o novo datasets
    lista_CNN = [[df_lista.loc[(df_lista.reg == reg) & (df_lista.ano == (df_lista.ano[df_lista.reg == reg].max())), 'path'].values[0],
                  df_lista.loc[(df_lista.reg == reg) & (df_lista.ano == (
                      df_lista.ano[df_lista.reg == reg].min())), 'path'].values[0]
                  ]for reg in df_lista.reg.drop_duplicates()]
    # Monta datases das imagens
    if os.path.exists(cf.dir_list['2.4']+cf.model_name['4']):
        df_imagens = cf.load(cf.dir_list['2.4']+cf.model_name['4'])
    else:
        df_imagens = fc.generate_dsCNN(lista_CNN)
    # Gerar modelo
    model = fc.modelsvm(df_imagens)
    return [df_lista, model]
# =====================================================( Fim da etapa 1 )===============================================
# =======================================================================================================================
#                    Incio da Fase 2 construir dataset já classificado e apresentar imagens  apartir do dataset.
#
# Objetico:
# processo 1:
# processo 2:
# processo 3:
# =======================================================================================================================


def etapa2():
    path_ds = cf.dir_list['2.4'] + f'PWimagens.pkl'
    if os.path.exists(path_ds) == False:
        dados, modelo = etapa1()

        # Carregar o data set com as imagens já classificadas
        ds_carregado = fc.analiseDataSet(dados.path, modelo)

        # Carregar o melhor modelo
        if os.path.exists(cf.dir_list['2.2']+cf.model_name['1']+'.joblib') == False:
            fc.espect_v2()

        ds_img = []
        for index, row in ds_carregado.iterrows():
            file  = row['file']
            info = file.replace(".jpg","").split("_")
            ano = info[1]
            reg = info[0]

            

            # verificar se a imagens ja existe
            if os.path.exists(cf.dir_list['2.5']+file) == False:
                # Carregar as imagem (original, teste, com)
                df = ds_carregado[ds_carregado.file == file]
                img_ml = fc.rebuild(df)
                original = fc.openImageCinza(cf.dir_list['2.1.1']+file)
                ap = cf.cv2.hconcat((original, img_ml))
                #Remecionar para o power BI
                larg, alt = ap.shape[1], ap.shape[0]
                pro = float(alt / larg)
                largura = 200
                altura = int(largura * pro)
                img = cf.cv2.resize(ap, (largura, altura), interpolation=cf.cv2.INTER_AREA)
                # gravar as imagens
                fc.gravar(cf.dir_list['2.5']+file, img)
            
            # Adicionar o nome da imagens no dataset
            ds_img.append([ano, reg, cf.dir_list['2.5']+file])

        ds_img = cf.pd.DataFrame(ds_img, columns=['ano', 'reg', 'path'])
        if os.path.exists(path_ds):
            os.remove(path_ds)
        ds_img.drop_duplicates()
        ds_img.to_pickle(path_ds)

    return path_ds
# =====================================================( Fim da fase 2 )===============================================

# =======================================================================================================================
#                    Incio da Fase 3 construir dataset já classificado e apresentar imagens  apartir do dataset.
#
# Objetico:
# processo 1:
# processo 2:
# processo 3:
# =======================================================================================================================
# Gerar o data set


def etapa3():
    if os.path.exists(cf.dir_list['2.4']+cf.saidas_name['5']) == False:
        dados = etapa1()
        # Carregar o data set com as imagens já classificadas
        ds_carregado = fc.analiseDataSet(dados[0].path, dados[1])
        df = fc.dataSet_Gerador(ds_carregado)

        # Ordenar o dataset  por ano e regiao
        df.sort_values(by=['ano', 'reg'], inplace=True)

        # Colocar a coluna data como indice
        df.set_index('data', inplace=True)

        # apresentar as 5 primeiras linhas
        df.head(5)

        if os.path.exists(cf.dir_list['2.4']+cf.saidas_name['1']):
            os.remove(cf.dir_list['2.4']+cf.saidas_name['1'])

        # Salvar o DataFrame em um arquivo CSV separado por ponto e vírgula
        df.to_csv(cf.dir_list['2.4']+cf.saidas_name['1'], sep=';', index=False)
        df_complement = pd.read_csv(
            cf.dir_list['2.4']+cf.saidas_name['4'], delimiter=';')
        df['populacao'] = df.apply(lambda row: fc.ts(
            df_complement, row['reg'], row['ano'], 2), axis=1)
        df['prefeito'] = df.apply(lambda row: fc.ts(
            df_complement, row['reg'], row['ano'], 1), axis=1)
        df['partido'] = df.apply(lambda row: fc.ts(
            df_complement, row['reg'], row['ano'], 0), axis=1)
        df.to_csv(cf.dir_list['2.4']+cf.saidas_name['5'], sep=';', index=False)

    return cf.dir_list['2.4']+cf.saidas_name['5']
# =====================================================( Fim da fase 3 )===============================================
# =======================================================================================================================
#                    Incio da Fase 4 Realizar as predições dos municipios.
#
# Objetico:
# processo 1:
# processo 2:
# processo 3:
# =======================================================================================================================
def prepar(arr,i):
    saida = []
    lag = 1
    for c in arr:
        saida.append([c[0], c[1], c[2], c[3]])
        lag += 1
    return saida

def etapa4():
    final = []
    if os.path.exists(cf.dir_list['2.4']+cf.saidas_name['6']) == False:
        for i in range(4):
            reg = i +  1
            desm = prev.desmatamento(reg)
            mata = prev.preservacao(reg)
            popu = prev.populacao(reg)

            for y in range(len(popu)):
                concat = np.concatenate((desm[y],mata[y],popu[y]), axis=None)
                final.append(concat)
        colunas = ["reg", "lag", "des_NO", "des_NE", "des_SO", "des_SE",
                   "mat_NO", "mat_NE", "mat_SO", "mat_SE",
                   "populacao"]
        df = pd.DataFrame(final, columns=[colunas])
        df.to_pickle(cf.dir_list['2.4']+cf.saidas_name['6'])
    return cf.dir_list['2.4']+cf.saidas_name['6']
