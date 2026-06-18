import { Loader2, Upload } from "lucide-react"
import { useRef, useState } from "react"
import { useDocumentUpload } from "../hooks/useDocumentUpload";

interface UploadZoneProps {
    onUploadSuccess: (docId: string, filename: string, fileSizeBytes: number) => void;
}

export const UploadZone = ({ onUploadSuccess }: UploadZoneProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const inputRef = useRef<HTMLInputElement>(null)
    const { uploadFile, isUploading } = useDocumentUpload()

    const handleFile = async (file: File) => {
        if (isUploading) return

        const uploadedDocument = await uploadFile(file)
        if (!uploadedDocument) return

        setSelectedFile(file)
        onUploadSuccess(uploadedDocument.doc_id, uploadedDocument.filename, file.size)
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

    return (
        <>
            <div
                className={`group mt-6 border-dashed border-2 border-outline-variant transition-all duration-300 rounded-lg p-12 flex flex-col items-center justify-center gap-4 bg-surface-container relative ${isUploading
                        ? "opacity-60 cursor-not-allowed"
                        : "hover:border-primary cursor-pointer"
                    }`}
                aria-busy={isUploading}
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
                        <p className="font-geist text-headline-md text-on-surface">
                            Subiendo documento...
                        </p>
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

            {selectedFile && <p>{selectedFile?.name}</p>}
        </>
    )
}
