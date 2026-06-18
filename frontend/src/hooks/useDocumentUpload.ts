import { useState } from "react"
import { toast } from "sonner"
import type { UploadResponse } from "../types/document.types"
import type { ApiErrorResponse } from "../types/api.types"

interface UseDocumentUploadReturn {
    uploadFile: (file: File) => Promise<UploadResponse | null>
    isUploading: boolean
}

export const useDocumentUpload = (): UseDocumentUploadReturn => {
    const [isUploading, setIsUploading] = useState<boolean>(false)

    const uploadFile = async (file: File): Promise<UploadResponse | null> => {
        const formData = new FormData()
        formData.append("uploaded_file", file)

        setIsUploading(true)

        try {
            const response = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const errorData: ApiErrorResponse = await response.json()
                toast.error(errorData.message)
                return null
            }

            return await response.json() as UploadResponse
        } catch {
            toast.error("No se pudo conectar con el servidor")
            return null
        } finally {
            setIsUploading(false)
        }
    }

    return { uploadFile, isUploading }
}
