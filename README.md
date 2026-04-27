# DocBot — Chat con tus documentos PDF

DocBot es una aplicación web que permite subir cualquier PDF y mantener 
una conversación con su contenido. Las respuestas están fundamentadas 
en el documento, con citas de página y memoria de conversación.

> 🚀 Demo: [URL cuando esté deployado]

---

## Arquitectura RAG

```
INGESTA (una vez por documento)
───────────────────────────────
PDF subido por el usuario
       ↓
PyMuPDF / pdfplumber         → extracción de texto limpio
       ↓
Chunking semántico            → LangChain RecursiveCharacterTextSplitter
       ↓                        con overlap, respetando párrafos y secciones
sentence-transformers         → embeddings locales (all-MiniLM-L6-v2, gratuito)
       ↓
ChromaDB (cliente directo)    → vector store persistente en disco por sesión

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
Groq API (LLaMA 3.3 70B)     → genera respuesta con streaming
       ↓
Respuesta + fuentes citadas   → texto en streaming + referencias a páginas/chunks
       ↓
Frontend React                → muestra la respuesta y los fragmentos usados
```

---

## Stack técnico

| Capa | Tecnología |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Frontend | [completar cuando hagas el frontend] |

---

## Arranque local

### Requisitos
- Python 3.13+
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
Documentación en `http://localhost:8000/docs`

### Variables de entorno

| Variable | Descripción | Ejemplo |
|---|---|---|
| GROQ_API_KEY | API key de Groq | gsk_... |
| CHROMA_PERSIST_DIR | Ruta vector store | ./storage/chroma |
| UPLOAD_DIR | Ruta uploads temporales | ./storage/uploads |
| MAX_PDF_SIZE_MB | Tamaño máximo PDF | 50 |
| CORS_ORIGINS | Orígenes permitidos | http://localhost:5173 |

---

## Tests

```bash
cd backend
pytest
```

---

## Mejoras futuras

- Persistir `progress_store` en Redis para tolerancia a reinicios
- Soporte multi-documento con búsqueda cruzada entre PDFs
- Workers dedicados con Celery para ingesta en producción