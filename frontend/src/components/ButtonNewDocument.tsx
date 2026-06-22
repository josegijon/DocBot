import { Upload } from "lucide-react"

interface ButtonNewDocumentProps {
    onClick: () => void
}

export const ButtonNewDocument = ({ onClick }: ButtonNewDocumentProps) => {
    return (
        <button
            className="flex items-center justify-center gap-2 bg-primary text-on-primary py-2 px-4 rounded-lg font-jetbrains text-label-md hover:opacity-90 transition-all active:scale-[0.98] cursor-pointer"
            onClick={onClick}
        >
            <Upload size={18} />
            Nuevo PDF
        </button>
    )
}
