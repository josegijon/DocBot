import { useCallback, useEffect, useRef, useState } from "react"
import { UPLOAD_DOCUMENT_ENDPOINT, UPLOAD_FORM_DATA_KEY } from "../utils/apiRoutes"
import type { UploadResponse } from "../types/document.types"
import { isApiErrorResponse } from "../types/api.types"

const GENERIC_UPLOAD_ERROR_MESSAGE = "No se pudo procesar el archivo. Inténtalo de nuevo."
const NETWORK_ERROR_MESSAGE = "No se pudo conectar con el servidor"

export interface UploadOutcome {
    document: UploadResponse | null
    errorMessage: string | null
}

interface UseDocumentUploadReturn {
    uploadFile: (file: File) => Promise<UploadOutcome>
    isUploading: boolean
}

export const useDocumentUpload = (): UseDocumentUploadReturn => {
    const [isUploading, setIsUploading] = useState<boolean>(false)
    const abortControllerRef = useRef<AbortController | null>(null)

    useEffect(() => {
        return () => {
            abortControllerRef.current?.abort()
        }
    }, [])

    const uploadFile = useCallback(async (file: File): Promise<UploadOutcome> => {
        const formData = new FormData()
        formData.append(UPLOAD_FORM_DATA_KEY, file)

        abortControllerRef.current?.abort()
        abortControllerRef.current = new AbortController()
        setIsUploading(true)

        try {
            const response = await fetch(UPLOAD_DOCUMENT_ENDPOINT, {
                method: "POST",
                body: formData,
                signal: abortControllerRef.current.signal,
            })

            if (!response.ok) {
                const errorPayload: unknown = await response.json()
                const errorMessage = isApiErrorResponse(errorPayload)
                    ? errorPayload.message
                    : GENERIC_UPLOAD_ERROR_MESSAGE

                return { document: null, errorMessage }
            }

            const document = await response.json() as UploadResponse
            return { document, errorMessage: null }
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") {
                return { document: null, errorMessage: null }
            }

            console.error("[useDocumentUpload] Fallo al subir el documento:", error)
            return { document: null, errorMessage: NETWORK_ERROR_MESSAGE }
        } finally {
            setIsUploading(false)
        }
    }, [])

    return { uploadFile, isUploading }
}
