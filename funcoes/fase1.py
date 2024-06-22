from fastapi import APIRouter
from funcoes.routers import etapas
from pydantic import BaseModel
import funcoes.routers.config as cf


router = APIRouter(prefix="/fase1")


@router.get("/", response_model=str, status_code=201)
def fase1():
    ret = etapas.etapa2()
    path = cf.dir_Raiz.replace("vs/funcoes/routers", "")
    return ret.replace("../", path)


@router.get("/fase2")
def fase2():
    ret = etapas.etapa3()
    path = cf.dir_Raiz.replace("vs/funcoes/routers", "")
    return ret.replace("../", path)
 

@router.get("/previsoes")
def previsoes():
    ret = etapas.etapa4()
    path = cf.dir_Raiz.replace("vs/funcoes/routers", "")
    return ret.replace("../", path)


@router.get("/avaliacao")
def previsoes():
    path = cf.dir_list['2.4']+'serieTemporalDesmatamento.csv'
    path = path.replace("vs/funcoes/routers", "")
    return path

# Exemplo post metodo
# @router.post("/", response_model=str, status_code=201)
# def fase1(dados: fase1_paramiters):
#     ret = etapas.etapa2(dados.reg, dados.ano)
#     path = cf.dir_Raiz.replace("vs/funcoes/routers", "")
#     return ret.replace("../", path)
