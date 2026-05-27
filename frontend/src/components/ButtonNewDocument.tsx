import { Upload } from "lucide-react"

export const ButtonNewDocument = () => {
    return (
        <button className="flex items-center justify-center gap-2 bg-primary text-on-primary py-2 px-4 rounded-lg font-jetbrains text-label-md hover:opacity-90 transition-all activate:scale-[0.98] cursor-pointer">
            <Upload size={18} />
            Nuevo PDF
        </button>
    )
}
