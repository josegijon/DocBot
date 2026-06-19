import type { IngestionStatus } from "../types/ingestionStatus.types"

interface IngestionProgressProps {
    progress: number
    status: IngestionStatus
    filename: string | null
}

export const IngestionProgress = ({ progress, status, filename }: IngestionProgressProps) => {
    const isFailed = status === "failed"

    const getStatusMessage = (): string => {
        if (status === "ready") return "¡Listo!"
        if (isFailed) return "Error en el procesamiento"
        if (progress < 50) return "Extrayendo texto..."
        return "Generando embeddings..."
    }

    return (
        <div className="flex flex-col gap-6 pt-6">
            <div className="flex justify-between items-end">
                <div className="flex flex-col gap-2">
                    <span className={`font-jetbrains text-label-md flex items-center gap-2 ${isFailed ? "text-error" : "text-primary"}`}>
                        <span className={`w-2 h-2 rounded-full ${isFailed ? "bg-error" : "bg-primary animate-pulse"}`}></span>
                        {getStatusMessage()}
                    </span>
                    <p className="font-jetbrains text-code-sm text-on-surface-variant">
                        {filename}
                    </p>
                </div>

                <span className="font-jetbrains text-code-sm text-primary">{progress}%</span>
            </div>

            {/* Progress bar */}
            <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden border border-outline-variant">
                <div
                    className={`h-full rounded-full transition-all duration-1000 ease-out relative ${isFailed
                            ? "bg-error"
                            : "bg-primary shadow-[0_0_12px_rgba(192,193,255,0.4)]"
                        }`}
                    style={{ width: `${progress}%` }}
                >
                    {!isFailed && (<div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.2)_50%,transparent_100%)] animate-[shimmer_2s_infinite]"></div>)}
                </div>
            </div>
        </div>
    )
}
