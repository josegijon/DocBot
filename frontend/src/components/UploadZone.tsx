import { Upload } from "lucide-react"
import { useRef, useState } from "react"
import { toast } from "sonner";

interface UploadZoneProps {
    onUploadSuccess: (docId: string, filename: string, filesize: number) => void;
}

export const UploadZone = ({ onUploadSuccess }: UploadZoneProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const inputRef = useRef<HTMLInputElement>(null)

    const handleFile = async (file: File) => {

        const formData = new FormData()
        formData.append("uploaded_file", file)

        try {
            const response = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const data = await response.json()
                toast.error(data.message)
                return
            }

            const data = await response.json()
            const fileSize = +(file.size / (1024 * 1024)).toFixed(2)
            setSelectedFile(file)
            onUploadSuccess(data.doc_id, data.filename, fileSize)
        } catch {
            toast.error("No se pudo conectar con el servidor")
        }
    }

    return (
        <>
            <div
                className="group mt-6 border-dashed border-2 border-outline-variant hover:border-primary transition-all duration-300 rounded-lg p-12 flex flex-col items-center justify-center gap-4 cursor-pointer bg-surface-container relative"
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDragEnter={(e) => e.preventDefault()}
                onDrop={(e) => {
                    e.preventDefault()
                    handleFile(e.dataTransfer.files[0])
                }}
            >
                <div className="w-16 h-16 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center group-hover:scale-110 group-hover:bg-primary/10 transition-transform">
                    <Upload
                        color="#c0c1ff"
                        size={32}
                    />
                </div>
                <div className="text-center flex flex-col gap-2">
                    <p className="font-geist text-headline-md text-on-surface">
                        Arrastra aquí tu PDF o <span className="text-primary underline cursor-pointer">Click para navegar</span>
                    </p>
                    <p className="font-jetbrains text-label-md text-on-surface-variant">
                        (Máx 50MB)
                    </p>
                </div>
                <input
                    ref={inputRef}
                    accept=".pdf"
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (file) handleFile(file)
                    }}
                />
            </div>

            {selectedFile && <p>{selectedFile?.name}</p>}
        </>
    )
}
