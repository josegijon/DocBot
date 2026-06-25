import { getDocumentEndpoint, getDocumentExistsEndpoint } from "./apiRoutes"
import { NETWORK_ERROR_MESSAGE } from "./errorMessages"
import type { DocumentExistsResponse } from "../types/document.types"

const GENERIC_DELETE_ERROR_MESSAGE = "No se pudo eliminar el documento"

export interface DeleteOutcome {
    success: boolean
    errorMessage: string | null
}

export interface ExistenceOutcome {
    exists: boolean | null
    errorMessage: string | null
}

export const deleteDocument = async (docId: string): Promise<DeleteOutcome> => {
    try {
        const response = await fetch(getDocumentEndpoint(docId), { method: "DELETE" })

        if (!response.ok) {
            return { success: false, errorMessage: GENERIC_DELETE_ERROR_MESSAGE }
        }

        return { success: true, errorMessage: null }
    } catch (error) {
        console.error("[documentApi] Fallo al eliminar el documento:", error)
        return { success: false, errorMessage: NETWORK_ERROR_MESSAGE }
    }
}

export const checkDocumentExists = async (
    docId: string,
    signal?: AbortSignal
): Promise<ExistenceOutcome> => {
    try {
        const response = await fetch(getDocumentExistsEndpoint(docId), { signal })

        if (!response.ok) {
            return { exists: null, errorMessage: NETWORK_ERROR_MESSAGE }
        }

        const data = await response.json() as DocumentExistsResponse
        return { exists: data.exists, errorMessage: null }
    } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
            return { exists: null, errorMessage: null }
        }
        console.error("[documentApi] Fallo al comprobar la existencia del documento:", error)
        return { exists: null, errorMessage: NETWORK_ERROR_MESSAGE }
    }
}
