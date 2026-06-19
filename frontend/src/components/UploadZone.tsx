import { Loader2, Upload } from "lucide-react"
import { useRef, useState } from "react"
import { toast } from "sonner";
import { useDocumentUpload } from "../hooks/useDocumentUpload";

const KEY_ENTER = "Enter"
const KEY_SPACE = " "

const UPLOAD_ZONE_LABEL = "Subir documento PDF. Arrastra un archivo o pulsa Enter para seleccionar uno. Máximo 50MB."


interface UploadZoneProps {
    onUploadSuccess: (docId: string, filename: string, fileSizeBytes: number) => void;
}

export const UploadZone = ({ onUploadSuccess }: UploadZoneProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const inputRef = useRef<HTMLInputElement>(null)
    const { uploadFile, isUploading } = useDocumentUpload()

    const handleFile = async (file: File) => {
        if (isUploading) return

        setSelectedFile(file)

        const { document, errorMessage } = await uploadFile(file)

        if (errorMessage) toast.error(errorMessage)

        if (!document) {
            setSelectedFile(null)
            return
        }

        onUploadSuccess(document.doc_id, document.filename, file.size)
    }

    const handleZoneClick = () => {
        if (isUploading) return
        inputRef.current?.click()
    }

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault()
    }

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (isUploading) return
        handleFile(e.dataTransfer.files[0])
    }

    const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) handleFile(file)
    }

    const handleZoneKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
        if (e.key !== KEY_ENTER && e.key !== KEY_SPACE) return
        e.preventDefault()
        handleZoneClick()
    }

    return (
        <>
            <div
                role="button"
                tabIndex={isUploading ? -1 : 0}
                aria-label={UPLOAD_ZONE_LABEL}
                aria-disabled={isUploading}
                aria-busy={isUploading}
                onKeyDown={handleZoneKeyDown}
                className={`group mt-6 border-dashed border-2 border-outline-variant transition-all duration-300 rounded-lg p-12 flex flex-col items-center justify-center gap-4 bg-surface-container relative focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface ${isUploading
                    ? "opacity-60 cursor-not-allowed"
                    : "hover:border-primary cursor-pointer"
                    }`}
                onClick={handleZoneClick}
                onDragOver={handleDragOver}
                onDragEnter={handleDragOver}
                onDrop={handleDrop}
            >
                <div className="w-16 h-16 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center group-hover:scale-110 group-hover:bg-primary/10 transition-transform">
                    {isUploading
                        ? <Loader2 className="text-primary animate-spin" size={32} />
                        : <Upload className="text-primary" size={32} />
                    }
                </div>

                <div className="text-center flex flex-col gap-2">
                    {isUploading ? (
                        <>
                            <p className="font-geist text-headline-md text-on-surface">
                                Subiendo documento...
                            </p>
                            <p className="font-jetbrains text-code-sm text-on-surface">{selectedFile?.name}</p>
                        </>
                    ) : (
                        <>
                            <p className="font-geist text-headline-md text-on-surface">
                                Arrastra aquí tu PDF o <span className="text-primary underline cursor-pointer">Click para navegar</span>
                            </p>
                            <p className="font-jetbrains text-label-md text-on-surface-variant">
                                (Máx 50MB)
                            </p>
                        </>
                    )}
                </div>

                <input
                    ref={inputRef}
                    accept=".pdf"
                    type="file"
                    className="hidden"
                    disabled={isUploading}
                    onChange={handleFileInputChange}
                />
            </div>

        </>
    )
}
