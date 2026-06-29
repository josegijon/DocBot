import { useEffect, useState } from "react"
import type { IngestionStatus } from "../types/ingestionStatus.types"

// Debe coincidir con la duración de la transición CSS de la barra de progreso
// en IngestionProgress.tsx (clase `duration-1000`). Si se cambia ahí, actualizar aquí.
const READY_DISPLAY_DELAY_MS = 1000

interface UseDelayedIngestionReadyReturn {
    isReadyToDisplay: boolean
}

export const useDelayedIngestionReady = (
    status: IngestionStatus,
    isKnownReady: boolean
): UseDelayedIngestionReadyReturn => {
    const [isReadyToDisplay, setIsReadyToDisplay] = useState<boolean>(() => isKnownReady)

    useEffect(() => {
        if (isKnownReady) {
            setIsReadyToDisplay(true)
            return
        }

        if (status !== "ready") {
            setIsReadyToDisplay(false)
            return
        }

        const timeoutId = setTimeout(() => {
            setIsReadyToDisplay(true)
        }, READY_DISPLAY_DELAY_MS)

        return () => {
            clearTimeout(timeoutId)
        }
    }, [status, isKnownReady])

    return { isReadyToDisplay }
}
