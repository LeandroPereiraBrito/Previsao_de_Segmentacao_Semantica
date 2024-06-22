from fastapi import APIRouter
from funcoes.routers import etapas

router = APIRouter(prefix="/fase2")
@router.get("/fase2")
def fase2():
    return etapas.etapa2()


