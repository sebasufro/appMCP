# Agente Orquestador MCP - Proyecto Práctico 3

Este repositorio contiene la implementación de un **Agente Orquestador MCP** con analítica en MongoDB. El sistema identifica estudiantes vía reconocimiento facial (v1) y responde preguntas de normativa (v2/RAG).

## Arquitectura

El sistema se compone de 3 microservicios orquestados:

1.  **Orchestrator API (Port 8000)**: `api/app.py` (FastAPI). Gestiona la lógica y logs en MongoDB.
2.  **ChatBot (Port 8001)**: `ChatBot/app.py` (Flask).
3.  **Reconocimiento Facial (Port 8002)**: `Reconocimiento-Facial/api/app.py` (Flask).

Además, incluye:
-   **MCP Server Protocol**: `mcp_server/server.py` para integración con Claude Desktop.
-   **n8n Workflow**: `n8n/pp3_workflow.json`.
-   **MongoDB**: Código de base de datos en `db/`.

## Requisitos previos

-   **Python 3.10+**
-   **MongoDB** corriendo en `localhost:27017`.

## Instalación

1.  **Clonar y configurar entorno virtual**:
    ```bash
    git clone <repo>
    cd appMCP
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

2.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Variables de Entorno**:
    Crea un archivo `.env` en la raíz (usa `.env.example` como base):
    ```ini
    FACE_REC_URL=http://localhost:8002/verify
    CHATBOT_URL=http://localhost:8001/api/ask
    MONGO_URI=mongodb://localhost:27017
    DB_NAME=ufro_master
    API_TOKEN=super-secret-token
    # Api Keys para ChatBot
    OPENAI_API_KEY=sk-...
    ```

## Ejecución

### Opción A: Scripts de Inicio Rápido (Recomendado)

**Windows (PowerShell):**
```powershell
.\scripts\start_services.ps1
```
Esto abrirá 3 terminales con cada servicio corriendo.

**Linux/Ubuntu:**
```bash
chmod +x scripts/start_services.sh
./scripts/start_services.sh
```
Los servicios correrán en segundo plano (logs en `logs/`).

### Opción B: Ejecución Manual
Debes abrir 3 terminales:
1.  `python Reconocimiento-Facial/api/app.py`
2.  `python ChatBot/app.py`
3.  `python MCP-Server/main.py`

## Uso

### API REST
-   **Endpoint**: `POST /process`
-   **Headers**: `Authorization: Bearer super-secret-token`
-   **Body**: `multipart/form-data` con `image` (file) y `question` (text).

### MCP Server (Claude Desktop)
Puedes usar el archivo `mcp_server/server.py` como un servidor MCP estándar.
```bash
python mcp_server/server.py
```
Herramientas expuestas: `identify_person`, `ask_normativa`.

### n8n
1.  Importa `n8n/pp3_workflow.json` en tu n8n.
2.  Asegúrate de que los servicios (8001/8002) estén corriendo.

## Testing
Para verificar la integración y seguridad:
```bash
pytest tests/
```