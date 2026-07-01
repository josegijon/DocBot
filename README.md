# DocBot — Chat con tus documentos PDF

DocBot es una aplicación web full-stack que permite subir cualquier PDF y mantener
una conversación con su contenido. Las respuestas están fundamentadas
en el documento (RAG), con citas de página y memoria de conversación por sesión.

Proyecto de portfolio orientado a demostrar competencia full-stack, buenas
prácticas de ingeniería de software y atención a la accesibilidad (WCAG 2.1),
no solo a "que funcione".

> 🚀 **Demo en producción:** [DocBot](https://docbot-app.vercel.app)
> ⚠️ El backend corre en el tier gratuito de Hugging Face Spaces. La primera
> petición tras un periodo de inactividad puede tardar varios segundos
> (cold start) mientras se despierta el contenedor y se cargan los modelos
> de embeddings/reranking en memoria. Es una limitación conocida del hosting
> gratuito, no un bug. Ver [Limitaciones conocidas](#limitaciones-conocidas).

---

## Arquitectura RAG

```
INGESTA (una vez por documento)
───────────────────────────────
PDF subido por el usuario
       ↓
PyMuPDF                       → extracción de texto limpio (por página)
       ↓
Chunking semántico            → LangChain RecursiveCharacterTextSplitter
       ↓                        con overlap, respetando párrafos y secciones
sentence-transformers         → embeddings locales (all-MiniLM-L6-v2, gratuito)
       ↓
ChromaDB (cliente directo)    → vector store persistente en disco, una colección por documento

CONSULTA (cada mensaje del usuario)
────────────────────────────────────
Mensaje del usuario
       ↓
ChromaDB similarity search    → recupera top-10 chunks candidatos
       ↓
Reranking (cross-encoder)     → cross-encoder/ms-marco-MiniLM-L-6-v2
       ↓                        reordena los 10 chunks, se quedan los top-3
Prompt ensamblado             → system prompt + contexto + historial + pregunta
       ↓
Groq API (LLaMA 3.3 70B)      → genera respuesta con streaming (SSE)
       ↓
Respuesta + fuentes citadas    → texto en streaming + referencias a páginas/chunks
       ↓
Frontend React                → muestra la respuesta token a token y los fragmentos usados
```

---

## Stack técnico

| Capa | Tecnología |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | ChromaDB (PersistentClient) |
| Backend | FastAPI + Uvicorn + Python 3.13 |
| Frontend | React 19 + TypeScript + Vite |
| Estilos | Tailwind CSS v4 (config CSS-first vía `@theme`) |
| Iconos / Notificaciones | Lucide React · Sonner |

---

## Despliegue

| Componente | Plataforma | Notas |
|---|---|---|
| Frontend | Vercel | Build estático de Vite |
| Backend | Hugging Face Spaces (Docker SDK) | Tier gratuito, CPU basic, 16GB RAM |

El backend se migró de Render a Hugging Face Spaces: el tier gratuito de
Render resultó insuficiente para cargar el stack de ML (SentenceTransformer +
CrossEncoder) de forma estable. La variable `VITE_API_URL` del frontend
apunta al Space desplegado.

---

## Arranque local

### Requisitos
- Python 3.13+
- Node.js 20+
- Cuenta gratuita en [Groq](https://console.groq.com)

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env    # Rellena GROQ_API_KEY
uvicorn app.main:app --reload
```

API disponible en `http://localhost:8000`
Documentación interactiva (Swagger) en `http://localhost:8000/docs`

#### Variables de entorno (backend)

| Variable | Descripción | Ejemplo |
|---|---|---|
| GROQ_API_KEY | API key de Groq (obligatoria) | `gsk_...` |
| CHROMA_PERSIST_DIR | Ruta del vector store | `./storage/chroma` |
| UPLOAD_DIR | Ruta de uploads temporales | `./storage/uploads` |
| MAX_PDF_SIZE_MB | Tamaño máximo de PDF admitido | `50` |
| CORS_ORIGINS | Orígenes permitidos (frontend) | `http://localhost:5173` |

### Frontend

```bash
cd frontend
npm install
cp .env.example .env    # Rellena VITE_API_URL si el backend no está en localhost:8000
npm run dev
```

App disponible en `http://localhost:5173`

#### Variables de entorno (frontend)

| Variable | Descripción | Ejemplo |
|---|---|---|
| VITE_API_URL | URL base del backend. Vacío = relativo (usa el proxy de Vite en dev) | `https://josegijon-docbot.hf.space` |

---

## Tests

```bash
cd backend
pytest
```

---

## Limitaciones conocidas

Decisiones tomadas conscientemente por alcance de portfolio y documentadas:

- **`progress_store` y sesiones de chat en memoria del proceso**: no
  persisten si el servidor se reinicia, y no escalan a múltiples instancias.
  Migrar a Redis solucionaría ambos problemas, pero añade una pieza de
  infraestructura (y coste) desproporcionada para el objetivo del proyecto.
- **Cold start en Hugging Face Spaces**: el tier gratuito duerme el
  contenedor tras un periodo de inactividad. Se documenta aquí en vez de
  enmascararse con un ping periódico externo (p. ej. cron-job.org), que
  solo desplazaría el problema y añadiría una dependencia externa sin
  resolver la causa raíz.
- **Ingesta en `BackgroundTasks` de FastAPI, no en un worker dedicado**: para
  un volumen de tráfico de portfolio es suficiente; en producción real con
  carga concurrente se recomendaría Celery o ARQ con un worker separado.
- **Un documento por sesión**: no hay búsqueda cruzada entre varios PDFs
  simultáneos.

---

## Accesibilidad

El frontend sigue las pautas **WCAG 2.1** de forma explícita: gestión de foco
en modales (`useFocusTrap`), regiones live (`aria-live="polite"`) para
actualizaciones asíncronas, y anillos de foco visibles
(`focus-visible:ring-2`) en todos los elementos interactivos.
