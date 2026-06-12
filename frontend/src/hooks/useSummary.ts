import { useCallback, useEffect, useState } from "react"
import { getSummaryStorageKey } from "../utils/storageKeys"
import type { IngestionStatus } from "../types/ingestionStatus.types";

type SummaryStreamEvent =
    | { token: string; done?: never; message?: never }
    | { done: true; token?: never; message?: never }
    | { message: string; token?: never; done?: never }

interface UseSummaryReturn {
    summary: string;
    isDone: boolean;
    error: string | null;
    resetSummary: () => void;
}

export const useSummary = (docId: string | null, status: IngestionStatus): UseSummaryReturn => {
    const [summary, setSummary] = useState<string>("")
    const [isDone, setIsDone] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    const resetSummary = useCallback(() => {
        setSummary("")
        setError(null)
        setIsDone(false)
    }, [])

    useEffect(() => {
        if (!docId || status !== "ready") return

        const cached = localStorage.getItem(getSummaryStorageKey(docId))
        if (cached) {
            setSummary(cached)
            setIsDone(true)
            return
        }

        const source = new EventSource(`/api/documents/${docId}/summary`)

        let accumulatedSummary = ""

        source.onmessage = (event) => {
            const data = JSON.parse(event.data) as SummaryStreamEvent

            if (data.done) {
                localStorage.setItem(getSummaryStorageKey(docId), accumulatedSummary)
                setIsDone(true)
                source.close()
                return
            }

            if (data.token) {
                accumulatedSummary += data.token
                setSummary((prev) => prev + data.token)
            }
        }

        source.addEventListener("stream_error", (event) => {
            const data = JSON.parse((event as MessageEvent).data) as SummaryStreamEvent
            setError(data.message ?? "Error desconocido en el stream")
            source.close()
        })

        source.onerror = () => {
            source.close()
        }

        return () => {
            source.close()
        }
    }, [docId, status])

    return { summary, isDone, resetSummary, error }
}
