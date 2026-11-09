# FraudAI Agent -- Guia de Deployment

**Proyecto:** FraudAI Agent (Harvey AI)
**Version:** 0.1.0
**Fecha:** 2026-04-06
**Autor:** ARCA (@docs-writer) / Adrian Infantes
**Fase:** F6 -- Deployment

---

## 1. Prerequisitos

### 1.1 Hardware minimo

| Componente | Minimo | Recomendado |
|---|---|---|
| **RAM** | 16 GB | 32 GB+ |
| **VRAM (GPU)** | 8 GB (NVIDIA) | 8 GB+ (Lovelace/Ada) |
| **Disco** | 20 GB libres | 50 GB+ (modelos + datos BOE) |
| **CPU** | 4 cores | 8 cores |

La GPU NVIDIA es necesaria para Ollama (modelo local Llama 3.1 8B que ejecuta Donna, el router). Sin GPU, Ollama caera en CPU con latencia inaceptable (>10s routing).

### 1.2 Software

| Dependencia | Version minima | Verificacion |
|---|---|---|
| **Docker** | 24.0+ | `docker --version` |
| **Docker Compose** | v2.20+ | `docker compose version` |
| **NVIDIA Container Toolkit** | 1.14+ | `nvidia-ctk --version` |
| **NVIDIA GPU Driver** | 535+ | `nvidia-smi` |
| **Python** | 3.11+ | `python --version` (solo para desarrollo local) |
| **uv** | 0.4+ | `uv --version` (solo para desarrollo local) |
| **curl** | Cualquiera | Para verificaciones de salud |

### 1.3 Cuentas y API keys

| Servicio | Obligatorio | Obtencion |
|---|---|---|
| **Anthropic API Key** | SI (para 5 agentes: Harvey, Louis, Jessica, Mike, Rachel) | https://console.anthropic.com/ |

> **Nota:** Donna (router) usa modelo local via Ollama y NO requiere API key externa.

---

## 2. Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/infantesromeroadrian/FraudAI-Agent.git
cd FraudAI-Agent

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key de Anthropic y ajustes necesarios

# 3. Arrancar servicios de infraestructura
./scripts/setup-dev.sh

# 4. Instalar dependencias Python (desarrollo local)
uv sync --extra dev

# 5. Arrancar la aplicacion
uv run uvicorn fraudai.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

# 6. Verificar
curl http://localhost:8000/api/v1/health

# 7. Abrir documentacion interactiva
# Swagger UI:  http://localhost:8000/docs
# ReDoc:       http://localhost:8000/redoc
```

---

## 3. Configuracion

### 3.1 Variables de entorno

Todas las variables se configuran en el archivo `.env` en la raiz del proyecto. La aplicacion las lee via `pydantic-settings`.

| Variable | Descripcion | Valor por defecto | Obligatorio |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | API key de Anthropic para Claude Sonnet 4 (agentes Harvey, Louis, Jessica, Mike, Rachel) | `""` (vacio) | SI |
| `OLLAMA_HOST` | URL del servidor Ollama (modelo local para Donna) | `http://localhost:11434` | NO |
| `OLLAMA_MODEL` | Modelo de Ollama para routing (Donna) | `llama3.1:8b-instruct-q4_K_M` | NO |
| `OLLAMA_KEEP_ALIVE` | Tiempo que Ollama mantiene el modelo en VRAM antes de descargarlo | `5m` | NO |
| `QDRANT_HOST` | Host del servidor Qdrant (vector DB para RAG) | `localhost` | NO |
| `QDRANT_PORT` | Puerto REST de Qdrant | `6333` | NO |
| `QDRANT_GRPC_PORT` | Puerto gRPC de Qdrant | `6334` | NO |
| `LOG_LEVEL` | Nivel de logging (DEBUG, INFO, WARNING, ERROR) | `INFO` | NO |
| `ENVIRONMENT` | Entorno de ejecucion. `development` habilita CORS permisivo; `production` lo restringe | `development` | NO |

### 3.2 Ejemplo de `.env` para desarrollo

```env
# LLM
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXX
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_KEEP_ALIVE=5m

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# App
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 3.3 Ejemplo de `.env` para produccion

```env
ANTHROPIC_API_KEY=sk-ant-api03-PRODUCCION-KEY
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_KEEP_ALIVE=5m

QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

LOG_LEVEL=WARNING
ENVIRONMENT=production
```

> En produccion, `OLLAMA_HOST` y `QDRANT_HOST` apuntan a los nombres de servicio de Docker Compose en lugar de `localhost`.

---

## 4. Arquitectura de servicios

### 4.1 Componentes

```
+-------------------+     +------------------+     +------------------+
|   FraudAI App     |     |     Qdrant       |     |     Ollama       |
|   (FastAPI)       |<--->|  (Vector DB)     |     |  (LLM Local)    |
|   Puerto: 8000    |     |  REST: 6333      |     |  Puerto: 11434  |
|                   |     |  gRPC: 6334      |     |  GPU: NVIDIA    |
+-------------------+     +------------------+     +------------------+
        |                         |                        |
        |    +--------------------+                        |
        |    |                                             |
        v    v                                             |
+-------------------+     +------------------+             |
|  Anthropic API    |     | Sandbox Docker   |             |
|  (Claude Sonnet)  |     | (Codigo efimero) |             |
|  Externo          |     | Red: none        |             |
+-------------------+     | RAM: 4GB max     |             |
                          +------------------+             |
                                                           |
                          +--------------------------------+
                          | Modelos locales en VRAM:
                          | - Llama 3.1 8B Q4_K_M (~5GB) -- Donna router
                          | - BGE-M3 (~2GB) -- Embeddings RAG
                          | - BGE-reranker-v2-m3 (~0.6GB) -- Reranking
                          | (carga dinamica, NO simultaneos)
                          +--------------------------------+
```

### 4.2 Docker Compose (servicios de infraestructura)

El archivo `docker-compose.yml` levanta dos servicios:

| Servicio | Imagen | Puertos | Volumen | Health check |
|---|---|---|---|---|
| **qdrant** | `qdrant/qdrant:v1.13` | 6333 (REST), 6334 (gRPC) | `qdrant_data` | `GET /healthz` cada 10s |
| **ollama** | `ollama/ollama:latest` | 11434 | `ollama_data` | `GET /api/tags` cada 10s |

Ambos servicios tienen:
- Restart policy: `unless-stopped`
- Health checks con start period para warm-up
- Ollama reserva GPU NVIDIA automaticamente via `nvidia-container-toolkit`
- Qdrant tiene limite de memoria de 2 GB

### 4.3 Aplicacion FastAPI

La aplicacion se ejecuta con uvicorn:

- **Desarrollo:** `uv run uvicorn fraudai.api.app:create_app --factory --reload`
- **Produccion (Docker):** `docker build -t fraudai-agent:latest . && docker run -p 8000:8000 --env-file .env fraudai-agent:latest`

El Dockerfile usa multi-stage build:
1. **Builder:** Instala dependencias con `uv` sobre `python:3.11-slim`
2. **Runtime:** Imagen minima con solo paquetes necesarios, usuario no-root (`fraudai`), health check integrado

---

## 5. Deployment paso a paso

### 5.1 Desarrollo local

```bash
# Paso 1: Prerequisitos
docker --version          # >= 24.0
docker compose version    # >= 2.20
nvidia-smi                # GPU visible

# Paso 2: Clonar y configurar
git clone https://github.com/infantesromeroadrian/FraudAI-Agent.git
cd FraudAI-Agent
cp .env.example .env
# Editar .env: poner ANTHROPIC_API_KEY

# Paso 3: Levantar infraestructura
./scripts/setup-dev.sh
# Este script:
#   - Copia .env.example a .env si no existe
#   - Levanta Qdrant y Ollama via docker compose
#   - Espera a que ambos pasen health check
#   - Descarga el modelo Llama 3.1 8B en Ollama (~4.7 GB)

# Paso 4: Instalar dependencias Python
uv sync --extra dev

# Paso 5: Arrancar la API
uv run uvicorn fraudai.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

# Paso 6: Verificar todo
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

### 5.2 Produccion con Docker

```bash
# Paso 1: Configurar .env de produccion
cp .env.example .env
# Editar con valores de produccion (ver seccion 3.3)

# Paso 2: Construir imagen de la aplicacion
docker build -t fraudai-agent:latest .

# Paso 3: Levantar todos los servicios
docker compose up -d

# Paso 4: Arrancar la aplicacion
docker run -d \
  --name fraudai-app \
  --network fraudai-agent_default \
  -p 8000:8000 \
  --env-file .env \
  -e QDRANT_HOST=qdrant \
  -e OLLAMA_HOST=http://ollama:11434 \
  fraudai-agent:latest

# Paso 5: Verificar
curl http://localhost:8000/api/v1/health
```

---

## 6. Primera ejecucion

### 6.1 Ingestion del BOE (corpus RAG)

> **Nota:** El modulo de ingestion (`fraudai.ingestion`) esta planificado para la fase F4. Cuando este disponible, la primera ingestion se ejecutara asi:

```bash
# Ingestion completa del corpus BOE (normativa de fraude bancario)
python -m fraudai.ingestion --mode full

# Ingestion incremental (solo cambios desde ultima ejecucion)
python -m fraudai.ingestion --mode incremental
```

La ingestion descarga normativa relevante de la API BOE, la chunkea por articulo/seccion legal, genera embeddings con BGE-M3, y los almacena en la coleccion `boe_legislation` de Qdrant.

**Normativa clave a indexar (ver F0 Apendice):**
- Ley 10/2010 PBC/FT (BOE-A-2010-6737)
- RD 304/2014 Reglamento PBC (BOE-A-2014-4742)
- RDL 19/2018 Servicios de Pago (BOE-A-2018-16036)
- Codigo Penal arts. 248-256, 301-304 (BOE-A-1995-25444)
- LO 3/2018 LOPDGDD (BOE-A-2018-16673)

### 6.2 Descarga del modelo local

El script `setup-dev.sh` descarga automaticamente el modelo para Donna. Si necesitas hacerlo manualmente:

```bash
# Verificar que Ollama esta corriendo
curl http://localhost:11434/api/tags

# Descargar el modelo
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M

# Verificar que esta disponible
curl -s http://localhost:11434/api/tags | python -m json.tool
```

---

## 7. Verificacion

### 7.1 Health check completo

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

Respuesta esperada (todo sano):
```json
{
    "status": "healthy",
    "qdrant": true,
    "ollama": true,
    "claude_api": true,
    "corpus_version": "2026-W14"
}
```

Posibles estados:
- `healthy`: Todos los servicios operativos
- `degraded`: Al menos un servicio caido pero otros funcionan
- `unhealthy`: Todos los servicios caidos

### 7.2 Verificacion individual de servicios

```bash
# Qdrant
curl -s http://localhost:6333/healthz
# Respuesta: (sin cuerpo, HTTP 200)

# Ollama
curl -s http://localhost:11434/api/tags | python -m json.tool
# Debe listar el modelo llama3.1:8b-instruct-q4_K_M

# App (OpenAPI docs)
curl -s http://localhost:8000/docs
# Debe devolver la pagina Swagger UI

# App (API info)
curl -s http://localhost:8000/openapi.json | python -m json.tool
```

### 7.3 Test funcional rapido

```bash
# Enviar un mensaje de chat (requiere autenticacion -- ver F6_api_reference.md)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "message": "Hola, necesito analizar unas transacciones sospechosas",
    "language": "es"
  }'
```

---

## 8. Gestion de VRAM

La RTX 2000 Ada (8 GB VRAM) aloja tres modelos con carga dinamica (nunca simultaneos):

| Fase | Modelos en VRAM | Consumo estimado |
|---|---|---|
| **Routing (Donna)** | Llama 3.1 8B Q4_K_M via Ollama | ~5 GB |
| **RAG (retrieval)** | BGE-M3 + BGE-reranker-v2-m3 | ~2.6 GB |
| **Agente (Claude API)** | Solo BGE-M3 para RAG contextual | ~2 GB |

La gestion se implementa via:
- `OLLAMA_KEEP_ALIVE=5m`: Ollama descarga el modelo tras 5 min de inactividad
- Los modelos de embeddings se cargan on-demand
- CUDA context overhead permanente: ~300 MB
- Peak VRAM: ~5.3 GB (Ollama activo), margen seguro de ~2.7 GB

**Monitorizar VRAM:**
```bash
# En tiempo real
watch -n 1 nvidia-smi

# Una vez
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

---

## 9. Logs y debugging

### 9.1 Ver logs de la aplicacion

```bash
# Desarrollo (uvicorn con --reload)
# Los logs se imprimen directamente en la terminal

# Produccion (Docker)
docker logs fraudai-app --follow --tail 100
```

### 9.2 Ver logs de servicios de infraestructura

```bash
# Qdrant
docker compose logs qdrant --follow --tail 50

# Ollama
docker compose logs ollama --follow --tail 50
```

### 9.3 Niveles de log

Configurar via variable de entorno `LOG_LEVEL`:

| Nivel | Uso |
|---|---|
| `DEBUG` | Desarrollo. Muestra todas las llamadas a LLM, tool calls, estados del grafo |
| `INFO` | Default. Operaciones normales, requests, health checks |
| `WARNING` | Produccion. Solo advertencias y errores |
| `ERROR` | Solo errores criticos |

---

## 10. Parar servicios

```bash
# Parar todo (conservando datos)
docker compose down

# Parar todo Y eliminar volumenes (DESTRUCTIVO -- borra datos de Qdrant y modelos de Ollama)
docker compose down -v

# Parar solo la aplicacion
# Ctrl+C si esta en foreground, o:
docker stop fraudai-app
```
