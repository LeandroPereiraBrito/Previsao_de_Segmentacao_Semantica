from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import funcoes.fase1

app = FastAPI()

# Serve the images directory
app.mount("/dataSets_Lab/Saidas/imagens", StaticFiles(directory="/app/dataSets_Lab/Saidas/imagens"), name="images")

# Serve the csv_files directory
app.mount("/dataSets_Lab/Saidas", StaticFiles(directory="/app/dataSets_Lab/Saidas"), name="csv_files")

# Redirect the root URL to /docs
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.include_router(funcoes.fase1.router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
