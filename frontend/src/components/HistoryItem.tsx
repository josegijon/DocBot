import { FileText, MessageSquare, Trash2 } from "lucide-react"
import { formatDate } from "../utils/format"
import type { DocumentHistory } from "../types/document.types"

interface HistoryItemProps extends Pick<DocumentHistory, "doc_id" | "session_id" | "filename" | "saved_at"> {
    isActive: boolean
    onSelect: (doc_id: string, session_id: string) => void
    onRemove: (doc_id: string) => void
}

export const HistoryItem = ({ doc_id, session_id, filename, saved_at, isActive, onSelect, onRemove }: HistoryItemProps) => {

    const itemButtonClass = isActive
        ? "bg-secondary-container/20 border-r-2 border-primary rounded-l-lg"
        : "hover:bg-surface-container-high border-transparent rounded-lg group"

    const iconClass = isActive
        ? "text-primary"
        : "text-on-surface-variant group-hover:text-primary"

    const textClass = isActive
        ? "font-bold text-on-surface"
        : "font-medium text-on-surface-variant group-hover:text-on-surface"

    const handleRemove = (e: React.MouseEvent) => {
        e.stopPropagation()
        onRemove(doc_id)
    }

    return (
        <div className="relative group/item">
            <button
                onClick={() => onSelect(doc_id, session_id)}
                aria-current={isActive ? "true" : undefined}
                className={`w-full text-left p-3 relative cursor-pointer transition-all active:scale-[0.98] border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${itemButtonClass}`}>
                <div className="flex items-start gap-3">
                    <span className={`mt-0.5 ${iconClass}`}>
                        <MessageSquare size={18} aria-hidden="true" />
                    </span>
                    <div className="overflow-hidden pr-7 flex flex-col gap-1">
                        <p
                            title={filename}
                            className={`text-body-md font-geist truncate ${textClass}`}
                        >
                            {filename}
                        </p>
                        <div className="flex items-center gap-2">
                            <span className="text-outline">
                                <FileText size={12} aria-hidden="true" />
                            </span>
                            <p className="text-2xs leading-2xs font-jetbrains text-outline uppercase">
                                {formatDate(saved_at)}
                            </p>
                        </div>
                    </div>
                </div>
            </button>

            <button
                onClick={handleRemove}
                aria-label={`Eliminar ${filename}`}
                title={`Eliminar ${filename}`}
                className="text-on-surface-variant hover:text-error opacity-100 lg:opacity-0 lg:group-hover/item:opacity-100 lg:focus-visible:opacity-100 transition-opacity absolute right-3 top-3 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-error focus-visible:rounded"
            >
                <Trash2 size={18} aria-hidden="true" />
            </button>
        </div>
    )
}
