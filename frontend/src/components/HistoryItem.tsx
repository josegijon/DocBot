import { FileText, MessageSquare, Trash2 } from "lucide-react"
import { formatDate } from "../utils/format"

interface HistoryItemProps {
    doc_id: string
    session_id: string
    filename: string
    saved_at: string
    isActive: boolean
    onSelect: (doc_id: string, session_id: string) => void
    onRemove: (doc_id: string) => void
}

export const HistoryItem = ({ doc_id, session_id, filename, saved_at, isActive, onSelect, onRemove }: HistoryItemProps) => {
    return (
        <div
            onClick={() => onSelect(doc_id, session_id)}
            className={`p-3 relative group/item cursor-pointer transition-all active:scale-[0.98] border ${isActive
                ? "bg-secondary-container/20 border-r-2 border-primary rounded-l-lg"
                : "hover:bg-surface-container-high border-transparent rounded-lg group"
                }`}
        >
            <div className="flex items-start gap-3">
                <span className={`mt-0.5 ${isActive ? "text-primary" : "text-on-surface-variant group-hover:text-primary"
                    }`}>
                    <MessageSquare size={18} />
                </span>
                <div className="overflow-hidden pr-7 flex flex-col gap-1">
                    <p className={`text-body-md font-geist truncate ${isActive ? "font-bold text-on-surface" : "font-medium text-on-surface-variant group-hover:text-on-surface"
                        }`}>
                        {filename}
                    </p>
                    <div className="flex items-center gap-2">
                        <span className="text-outline">
                            <FileText size={12} />
                        </span>
                        <p className="text-[10px] font-jetbrains text-outline uppercase">
                            {formatDate(saved_at)}
                        </p>
                    </div>
                </div>
                <button
                    onClick={(e) => {
                        e.stopPropagation()
                        onRemove(doc_id)
                    }}
                    className="text-on-surface-variant hover:text-error opacity-100 lg:opacity-0 lg:group-hover/item:opacity-100 transition-opacity absolute right-3 top-3 cursor-pointer"
                >
                    <Trash2 size={18} />
                </button>
            </div>
        </div>
    )
}
