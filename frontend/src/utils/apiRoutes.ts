// src/utils/apiRoutes.ts (archivo nuevo)

export const UPLOAD_DOCUMENT_ENDPOINT = "/api/documents/upload"

// Debe coincidir exactamente con el nombre del parámetro `uploaded_file`
// definido en el endpoint POST /api/documents/upload (backend: documents.py).
// Si se renombra en el backend, actualizar aquí.
export const UPLOAD_FORM_DATA_KEY = "uploaded_file"
