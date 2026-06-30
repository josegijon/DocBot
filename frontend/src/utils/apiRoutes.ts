// apiRoutes.ts

// Base del backend. En local, si no defines VITE_API_URL, cae a "" y las
// peticiones quedan relativas (compatible con el proxy de Vite en server.proxy).
// En producción, Vercel inyecta VITE_API_URL apuntando al backend de Render.
const API_BASE_URL = import.meta.env.VITE_API_URL ?? ""

export const UPLOAD_DOCUMENT_ENDPOINT = `${API_BASE_URL}/api/documents/upload`

// Debe coincidir exactamente con el nombre del parámetro `uploaded_file`
// definido en el endpoint POST /api/documents/upload (backend: documents.py).
// Si se renombra en el backend, actualizar aquí.
export const UPLOAD_FORM_DATA_KEY = "uploaded_file"

export const getDocumentEndpoint = (docId: string): string =>
    `${API_BASE_URL}/api/documents/${docId}`

export const getDocumentExistsEndpoint = (docId: string): string =>
    `${API_BASE_URL}/api/documents/${docId}/exists`

export const getDocumentStatusEndpoint = (docId: string): string =>
    `${API_BASE_URL}/api/documents/${docId}/status`

export const getDocumentSummaryEndpoint = (docId: string): string =>
    `${API_BASE_URL}/api/documents/${docId}/summary`

export const CHAT_ENDPOINT = `${API_BASE_URL}/api/chat/`
