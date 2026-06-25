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
    } catch {
        return { success: false, errorMessage: NETWORK_ERROR_MESSAGE }
    }
}

export const checkDocumentExists = async (docId: string): Promise<ExistenceOutcome> => {
    try {
        const response = await fetch(getDocumentExistsEndpoint(docId))

        if (!response.ok) {
            return { exists: null, errorMessage: NETWORK_ERROR_MESSAGE }
        }

        const data = await response.json() as DocumentExistsResponse
        return { exists: data.exists, errorMessage: null }
    } catch {
        return { exists: null, errorMessage: NETWORK_ERROR_MESSAGE }
    }
}
