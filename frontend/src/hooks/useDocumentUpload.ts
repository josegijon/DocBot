import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import type { UploadResponse } from "../types/document.types"
import type { ApiErrorResponse } from "../types/api.types"

interface UseDocumentUploadReturn {
    uploadFile: (file: File) => Promise<UploadResponse | null>
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

    const uploadFile = async (file: File): Promise<UploadResponse | null> => {
        const formData = new FormData()
        formData.append("uploaded_file", file)

        abortControllerRef.current = new AbortController()
        setIsUploading(true)

        try {
            const response = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
                signal: abortControllerRef.current.signal,
            })

            if (!response.ok) {
                const errorData: ApiErrorResponse = await response.json()
                toast.error(errorData.message)
                return null
            }

            return await response.json() as UploadResponse
        } catch (error) {
            if (error instanceof Error && error.name === "AbortError") return null
            toast.error("No se pudo conectar con el servidor")
            return null
        } finally {
            setIsUploading(false)
        }
    }

    return { uploadFile, isUploading }
}
