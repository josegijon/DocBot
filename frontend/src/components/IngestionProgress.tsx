
interface IngestionProgressProps {
    progress: number
    status: "processing" | "ready" | "failed"
    filename: string | null
}

export const IngestionProgress = ({ progress, status, filename }: IngestionProgressProps) => {
    const getStatusMessage = () => {
        if (status === "ready") return "¡Listo!"
        if (status === "failed") return "Error en el procesamiento"
        if (progress < 50) return "Extrayendo texto..."
        return "Generando embeddings..."
    }

    return (
        // Contenedor de estado de progreso
        <div className="flex flex-col gap-6">
            <div className="flex justify-between items-end">
                <div className="flex flex-col gap-1">
                    <span className="font-jetbrains text-label-md text-primary flex items-center gap-2">
                        <span className="animate-pulse w-2 h-2 rounded-full bg-primary"></span>
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
                    className="h-full bg-primary rounded-full transition-all duration-1000 ease-out relative shadow-[0_0_12px_rgba(192,193,255,0.4)]"
                    style={{ width: `${progress}%` }}
                >
                    <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.2)_50%,transparent_100%)] animate-[shimmer_2s_infinite]"></div>
                </div>
            </div>
        </div>
    )
}
