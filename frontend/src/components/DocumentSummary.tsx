
interface DocumentSummaryProps {
    summary: string
    isDone: boolean
}

export const DocumentSummary = ({ summary, isDone }: DocumentSummaryProps) => {
    return (
        <div className="flex-1 overflow-y-auto pt-6 chat-scrollbar flex flex-col gap-3">
            <h3 className="font-jetbrains text-label-md uppercase text-primary tracking-widest">
                Resumen del documento
            </h3>

            <p className="font-geist text-body-md text-on-surface-variant">
                {summary}
                {!isDone && (
                    <span className="animate-pulse text-primary">|</span>
                )}
            </p>
        </div>
    )
}
