import { AlertTriangle } from "lucide-react"
import { formatMessageContent } from "../utils/format"

interface DocumentSummaryProps {
    summary: string
    isDone: boolean
    error: string | null
}

export const DocumentSummary = ({ summary, isDone, error }: DocumentSummaryProps) => {
    if (error) {
        return (
            <div className="flex-1 overflow-y-auto chat-scrollbar flex flex-col gap-3">
                <h3 className="font-jetbrains text-label-md uppercase text-primary tracking-widest">
                    Resumen del documento
                </h3>
                <div className="flex items-start gap-2 text-error">
                    <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                    <p className="font-geist text-body-md">{error}</p>
                </div>
            </div>
        )
    }

    return (
        <div className="flex-1 overflow-y-auto chat-scrollbar flex flex-col gap-3">
            <h3 className="font-jetbrains text-label-md uppercase text-primary tracking-widest">
                Resumen del documento
            </h3>
            <p className="font-geist text-body-md text-on-surface-variant whitespace-pre-wrap">
                {formatMessageContent(summary)}
                {!isDone && (
                    <span className="animate-pulse text-primary font-bold ml-0.5">|</span>
                )}
            </p>
        </div>
    )
}
