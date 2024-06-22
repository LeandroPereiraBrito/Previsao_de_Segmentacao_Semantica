import funcoes.routers.config as cf
import os

# ========================================================
#          Função para apresentar a imagem
# ========================================================


def apresentar(img):
    cf.cv2.imshow('Teste', img)
    cf.cv2.waitKey(0)


# ========================================================
#          Função para graver as imagens
# ========================================================
def gravar(filename, img):
    cf.cv2.imwrite(filename, img)


# ========================================================
#               Função para abrir a imagem
# ========================================================
def openImageCinza(path):
    img = cf.cv2.imread(path)
    img = cf.cv2.cvtColor(img, cf.cv2.COLOR_BGR2GRAY)
    return cf.cv2.resize(img, (255, 255), interpolation=cf.cv2.INTER_AREA)


def openImageColorida(path):
    imgc = cf.cv2.imread(path)
    return cf.cv2.resize(imgc, (255, 255), interpolation=cf.cv2.INTER_AREA)

# ========================================================
#        Função para gerar a lista com os arquivos
# ========================================================


def getFiles(dir):
    listFiles = []
    for currentFolder, subFolder, files in os.walk(dir):
        for file in files:
            arquivo = dir+file
            if os.path.exists(arquivo) and arquivo not in listFiles:
                listFiles.append([arquivo, file])
    return listFiles




#===============================================================
#        Função que gerar os modelos de claficação
#==============================================================
def model_generator(df,x_col, y_col, path):
    nom_model = ""
    vd_score = 0
    espc_model = None
    data_atual = cf.datetime.now()
    data_atual = data_atual.strftime("%d-%m-%Y %H:%M")
    saidas = [f'Treinamento realizado em {data_atual}']

    # Dividir o conjunto de dados em features (entradas) e target (saída)
    X = df[x_col]
    y = df[y_col]

    # Dividir os dados em  treinamento e teste
    X_train, X_test, y_train, y_test = cf.train_test_split(X, y, test_size=0.2, random_state=42)

    parametros = [int(x * 3) for x in range(7)]

    for lup in  parametros:
        # Inicializar e treinar os modelos de clustering
        kmeans = cf.KMeans(n_clusters=lup, random_state=42)
        dbscan = cf.DBSCAN(eps=(lup/10), min_samples=lup)
        gmm = cf.GaussianMixture(n_components=lup, random_state=42)
        rfc = cf.RandomForestClassifier(n_estimators=int(lup*10), random_state=42)

        # Lista de modelos
        modelos = [('K-Means', kmeans),
                   ('DBSCAN', dbscan),
                   ('RandomForest', rfc),
                   ('Gaussian Mixture', gmm)]

        # Avaliar os modelos
        for nome, modelo in modelos:
            start = cf.time.time()

            try:
                # Ajustar o modelo
                if nome != 'RandomForest':
                    labels = modelo.fit_predict(X)
                    score = cf.silhouette_score(X, labels)
                    mensagem =  f'{nome}: Índice de Silhueta = {score} tempo {(cf.time.time() - start)} com cluster {lup}'
                else:
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                    score = cf.accuracy_score(y_test, y_pred)
                    mensagem = f'{nome}: accuracy_score = {score} tempo {(cf.time.time() - start)} com estimators {int(lup * 10)}'

                # Verificar se o modelo possui a melhor performance
                if vd_score < score:
                    vd_score = score
                    espc_model = modelo
                    nom_model = f'{nome} com {lup} cluster e acurácia de {score} '
                    if nome == 'RandomForest':
                        nom_model = f'{nome} com {int(lup * 10)} estimators e acurácia de {score} '

            except:
                if nome != 'RandomForest':
                    mensagem = f'{nome}: não aplicavel ao dataset com {lup} cluster'
                else:
                    mensagem = f'{nome}: não aplicavel ao dataset com {int(lup * 10)} estimators'

            print(mensagem)
            saidas.append(mensagem)

    mensagem = f'===(Treinamento foi realizado com 4 modelos e 28 parâmetros o mais performatico foi {nom_model}) ======'
    print(mensagem)
    saidas.append(mensagem)

    # Gravar saidas
    saidas = cf.pd.DataFrame(saidas)
    saidas.to_csv(cf.dir_list['2.3']+path+'.csv', sep=';', index=False)
    cf.dump(espc_model, cf.dir_list['2.2']+path+'.joblib')



# ===============================================================
#        Função que gerar os modelos de claficação
# ==============================================================
def generate_dsCNN(lista_CNN):

    # lêr o modelo existente
    path_modelo =  cf.dir_list['2.2']+cf.model_name['1']+'.joblib'
    if os.path.exists(path_modelo):
        modelo =  cf.load(path_modelo)
    else:
        rf = generato_ClassificadorPX()
        modelo = model_generator(rf,['r', 'g', 'b'], 'saida', cf.model_name['1'])

    matriz = []
    # Montar novo data set
    for imagens in lista_CNN:
        for paths in imagens:
            print(f'Leitura da imagen: {paths}')


            #Classifcação das imagens
            for linha in openImageColorida(paths):
                newLinha = []
                for coluna in linha:
                    predito  = modelo.predict([coluna])[0]
                    newLinha.append(predito)
                matriz.append([linha,newLinha])

    df_ltms = cf.pd.DataFrame(matriz, columns=['features','target'])
    df_ltms.to_pickle(cf.dir_list['2.4']+cf.model_name['4'])
    return


# ***************(Função com algoritomo de ML para classificação )***************


def generato_ClassificadorPX():
    # =================( Variaveis de contexto )=======================

    # ========== ( Imagens que serão adotadas com padrão )========================
    imagens = ['1_2010.jpg', '2_2017.jpg',
               '2_2004.jpg', '2_2013.jpg', '2_2007.jpg']

    # =========== ( crop das imagens com a classe que as cores são )===============
    frag = {'2_2007.jpg': [
        [30, 60, 0, 50, 255],
        [40, 75, 80, 180, 0],
        [150, 200, 1200, 1300, 255],
        [30, 200, 1020, 1060, 0],
        [550, 700, 1040, 1300, 255]
    ],
        '2_2013.jpg': [
            [520, 545, 350, 890, 255],
            [650, 690, 230, 430, 0]
        ],
        '2_2004.jpg': [
            [200, 300, 0, 300, 255],
            [790, 850, 190, 270, 0]
        ],
        '2_2017.jpg': [
            [0, 330, 75, 145, 0],
            [288, 750, 585, 670, 255]
        ],
        '1_2010.jpg': [
            [50, 90, 30, 150, 255],
            [220, 250, 760, 930, 255],
            [545, 635, 760, 800, 255],
            [680, 0, 760, 800, 255],
            [0, 130, 1230, 1300, 0],
            [450, 490, 1300, 0, 0]
        ]
    }

    matriz = []
    for imagen in (imagens):
        posicoes = frag[imagen]
        image = cf.cv2.imread(cf.dir_list['2.1.1']+imagen)
        for posicao in posicoes:
            y1, y2, x1, x2, classe = posicao

            # Se for 0 assume a quantidade de linhas da imagem
            if y2 == 0:
                y2 = image.shape[0]

            # Se for 0 assume a quantidade de colunas da imagem
            if x2 == 0:
                x2 = image.shape[1]

            # crop da imagem
            crop = image[y1:y2, x1:x2]

            # Atribuir canal de cores na dataset
            for linha in crop:
                for coluna in linha:
                    matriz.append([coluna[0], coluna[1], coluna[2], classe])

    df = cf.pd.DataFrame(matriz, columns=["r", "g", "b", "saida"])

    return df




def modelsvm(df_imagens):
    # inicialização de variaveis
    path = cf.dir_list['2.2']+cf.model_name['5']
    svc = None
    mensagem = None

    # Lêr arquivos existente modelo e metricas
    if os.path.exists(path+'.joblib'):
        dsM = cf.pd.read_pickle(path+'.csv')
        mensagem = [row['Resultado'] for index, row in dsM.iterrows()]
        svc = cf.load(path+'.joblib')
    else:

        df = []

        # Recuperar imagens de exemplo
        for index, row in df_imagens.iterrows():
            feature = [x for x in row['features']]
            for y in range(len(feature)):
                r, g, b = feature[y]
                df.append([r, g, b, row['target'][y]])

        # Converter em dataframe
        df = cf.pd.DataFrame(df, columns=['R', 'G', 'B', 'C'])

        # remover duplicador
        df.drop_duplicates()

        # Separar target e features
        x = df[['R', 'G', 'B']]
        y = df['C']

        # Separar em Train e teste
        X_train, X_test, y_train, y_test = cf.train_test_split(
            x, y, test_size=0.2, random_state=42)

        # Carregar modelo
        svc = cf.SVC()

        # Ajustar modelo aos dados
        svc.fit(X_train, y_train)

        # Realizar previsão de teste
        y_pred = svc.predict(X_test)

        # Guarda a imagem dataset
        mensagem = ['Model accuracy : {0:0.3f}'. format(
            cf.accuracy_score(y_test, y_pred))]
        rs = cf.pd.DataFrame(mensagem, columns=['Resultado'])

        rs.to_pickle(path+'.csv')
        cf.dump(svc, path+'.joblib')

    return svc


# ===============================================================================
#                      Função para classificar as cores
# ===============================================================================

# ***************(Função com algoritomo de ML para classificação )***************
def espect_v2():
    # =================( Variaveis de contexto )=======================
    img = cf.cv2.imread(cf.dir_list['2.1.2']+'1_2010.jpg')
    nom_model = ""
    vd_score = 0
    espc_model = None
    data_atual = cf.datetime.now()
    data_atual = data_atual.strftime("%d-%m-%Y %H:%M")
    saidas = [f'Treinamento realizado em {data_atual}']

    # Crop da imagem
    fx = img[220:400]

    # Gravar trecho recortado
    gravar(cf.dir_list['2.1.3']+'crop_0.jpg', fx)

    figs = [
        [fx[:, :100], 0],
        [fx[55:90, 1100:1300], 0],
        [fx[55:90, 800:1050], 255],
        [fx[60:70, 1320:1340], 255]
    ]

    # Montagem da estrutura de dados
    cr = 1
    matriz = []
    for cor in figs:
        tobe = cor[1]
        asis = cor[0]
        gravar(cf.dir_list['2.1.3']+f'crop_{cr}.jpg', asis)
        for i in asis:
            for j in i:
                matriz.append([j[0], j[1], j[2], tobe])
        cr += 1
    df = cf.pd.DataFrame(matriz, columns=["r", "g", "b", "saida"])
    model_generator(df, ['r', 'g', 'b'], 'saida', cf.model_name['1'])

# ========================================================
# Função para reconstruir a imagem a partir do dataset
# ========================================================
def rebuild(ds_img):
    espectro = ds_img.drop(['file', 'posicao'], axis=1)
    espectro = cf.np.asarray([x[1]['linha'] for x in espectro.iterrows()])
    img_es = cf.cv2.cvtColor(espectro.astype(
        cf.np.uint8), cf.cv2.COLOR_GRAY2BGR)
    return cf.cv2.cvtColor(img_es, cf.cv2.COLOR_BGR2GRAY)



# ========================================================
# Função para apresentar a imagens com seu spectro
# ========================================================
def analiseDataSet(list, model_svm):
    # Gerar o caminho do dataset existente
    path = cf.dir_list['2.4']+cf.saidas_name['2']

    # Carregar o modelo de classificação de cores
    modelo = cf.load(cf.dir_list['2.2']+cf.model_name['1']+'.joblib')

    # Gerar o nome das colunas
    columns = ['file', 'posicao', 'linha']

    for item in list:
        try:
            # Recuperar o nome do arquivo
            info_path = item.split("/")
            nome = f'{info_path[len(info_path)-1]}'
            classificar = False
            #print(f'Arquivo {nome} emprocessamento')
            # ler a imagem
            try:
                img = openImageColorida(item)
            except:
                print("Fala na opção de leitura 1")
                img = cf.pil.open(item)
                img = cf.np.asanyarray(img)
                img = cf.cv2.resize(img, (255, 255), interpolation=cf.cv2.INTER_AREA)

            # Carregar o datasets gravado

            existente = None
            if os.path.exists(path):
                existente = cf.pd.read_pickle(path)
                if (not existente['file'].isin([nome]).any()):
                    classificar = True
            else:
                classificar = True

            # verificar se a imagem já foi carregada no dataset
            if classificar:
                matriz = []
                cont = 1
                # Classificar os pixels
                for i in img:
                    t = []
                    for j in i:
                        t.append([j[0], j[1], j[2]])
                    resultado = model_svm.predict(t)
                    cont += 1
                    matriz.append([nome, cont, resultado])

                # Transforma a matriz em pandas dataframe
                df_pronto = cf.pd.DataFrame(matriz, columns=columns)
                try:

                    # Salvar o dataSet
                    if existente is not None:
                        df_pronto = cf.pd.concat([existente, df_pronto])
                    df_pronto.to_pickle(path)

                except BaseException as e:
                    x=0

        except BaseException as e:
            x= 0
    return cf.pd.read_pickle(path)




def dataSet_Gerador(ds_carregado):
    municipio = {'1': 'Paracatu', '2': 'Lucas do Rio Verde',
                 '3': 'Parauapebas', '4': 'São Paulo'}
    df = ds_carregado.copy()  # Evita modificar o DataFrame original

    corte = int(255/2)

    matriz = []
    for i in df['file'].drop_duplicates():
        df_analise = df[df['file'] == i]

        # informações de data e região
        info = i.split('.')
        reg = info[0][:1]
        ano = info[0][2:]

        nor = df_analise[df_analise['posicao'] < int(255/2)]['linha']
        sul = df_analise[df_analise['posicao'] >= int(255/2)]['linha']

        des_NO = [cf.collections.Counter(x[:corte])[255] for x in nor]
        des_NE = [cf.collections.Counter(x[corte:])[255] for x in nor]
        des_SO = [cf.collections.Counter(x[:corte])[255] for x in sul]
        des_SE = [cf.collections.Counter(x[corte:])[255] for x in sul]

        mat_NO = [cf.collections.Counter(x[:corte])[0] for x in nor]
        mat_NE = [cf.collections.Counter(x[corte:])[0] for x in nor]
        mat_SO = [cf.collections.Counter(x[:corte])[0] for x in sul]
        mat_SE = [cf.collections.Counter(x[corte:])[0] for x in sul]

        des_NO = cf.np.sum(des_NO)
        des_NE = cf.np.sum(des_NE)
        des_SO = cf.np.sum(des_SO)
        des_SE = cf.np.sum(des_SE)

        mat_NO = cf.np.sum(mat_NO)
        mat_NE = cf.np.sum(mat_NE)
        mat_SO = cf.np.sum(mat_SO)
        mat_SE = cf.np.sum(mat_SE)

        matriz.append([f'{ano}-1-1', ano, municipio[reg], reg, des_NO,
                       des_NE, des_SO, des_SE, mat_NO, mat_NE, mat_SO, mat_SE])

    columns = ['data', 'ano', 'municipio', 'reg', 'des_NO', 'des_NE',
               'des_SO', 'des_SE', 'mat_NO', 'mat_NE', 'mat_SO', 'mat_SE']

    return cf.pd.DataFrame(matriz, columns=columns)


# Fução para realizar o merge dos datasets
def ts(df_complement,reg, ano, t):
    dicio = {
        '1': ['partido_Paracatu', 'prefeito - Paracatu', 'popul_paracatu'],
        '2': ['partido_Lucas', 'prefeito - Lucas', 'popul_lucas'],
        '3': ['partido_Parauapebas', 'Prefeito-Para', 'popul_Parauapeba'],
        '4': ['partido_São Paulo', 'Prefeito_SP', 'popul_sp']
    }
    filtro = (df_complement['ano'] == ano)
    if filtro.any():
        return df_complement.loc[filtro, dicio[reg][t]].values[0]
    else:
        return 0
