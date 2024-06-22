## Features

O objetivo deste projeto é desenvolver um modelo de análise de desmatamento utilizando imagens de satélite. Através da aplicação de técnicas avançadas de processamento de imagens e análise de dados, buscamos:
- Identificar áreas vegetadas e desmatadas em imagens de satélite;
- Prever possíveis áreas de desmatamento com base em padrões históricos;
- Analisar a evolução do desmatamento ao longo do tempo em uma área específica;
- Fornecer insights acionáveis para apoiar políticas e ações de conservação ambiental.

Este projeto visa contribuir para o entendimento e monitoramento do desmatamento no Brasil, fornecendo informações precisas e atualizadas para orientar decisões e intervenções na área ambiental e complementar estudos ambientais.

Esta API server os endpoints com os modelos já treinados para o nosso PowerBi consumir e servir a visualização das previsões.

[Notebook](https://colab.research.google.com/drive/14hfeKOkWzy4OVn1iybjHr88vBzwkQncb#scrollTo=bAKcjYyE2_xk) onde foi feita exploração de dados e o treinamento dos dados.


## Requirements

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Python 3.9 or higher (if running outside of Docker)

## Getting Started

To get started with the FastAPI application, follow these steps:

1. Build the Docker image:

    ```bash
    docker build -t grupo_1 .
    ```

2. Run the Docker container:

    ```bash
    docker run -v /host/data:/app/dataSets_Lab/Saidas -p 5000:5000 grupo_1
    ```

3. Access the FastAPI application in your web browser at `http://0.0.0.0:5000`.
