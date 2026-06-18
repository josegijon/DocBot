import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { UPLOAD_DOCUMENT_ENDPOINT, UPLOAD_FORM_DATA_KEY } from "../utils/apiRoutes"
import type { UploadResponse } from "../types/document.types"
import { isApiErrorResponse } from "../types/api.types"

interface UseDocumentUploadReturn {
    uploadFile: (file: File) => Promise<UploadResponse | null>
    isUploading: boolean
}

const GENERIC_UPLOAD_ERROR_MESSAGE = "No se pudo procesar el archivo. Inténtalo de nuevo."

export const useDocumentUpload = (): UseDocumentUploadReturn => {
    const [isUploading, setIsUploading] = useState<boolean>(false)
    const abortControllerRef = useRef<AbortController | null>(null)

    useEffect(() => {
        return () => {
            abortControllerRef.current?.abort()
        }
    }, [])

    const uploadFile = async (file: File): Promise<UploadResponse | null> => {
        const formData = new FormData()
        formData.append(UPLOAD_FORM_DATA_KEY, file)

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

                toast.error(errorMessage)
                return null
            }

            return await response.json() as UploadResponse
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") return null

            console.error("[useDocumentUpload] Fallo al subir el documento:", error)

            toast.error("No se pudo conectar con el servidor")
            return null
        } finally {
            setIsUploading(false)
        }
    }

    return { uploadFile, isUploading }
}
