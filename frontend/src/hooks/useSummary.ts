import { useCallback, useEffect, useState } from "react"

import type { IngestionStatus } from "../types/ingestionStatus.types";

import { getDocumentSummaryEndpoint } from "../utils/apiRoutes";
import { getSummaryStorageKey } from "../utils/storageKeys"
import { NETWORK_ERROR_MESSAGE } from "../utils/errorMessages";

const SSE_EVENT_STREAM_ERROR = "stream_error"

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

        const source = new EventSource(getDocumentSummaryEndpoint(docId))

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

        source.addEventListener(SSE_EVENT_STREAM_ERROR, (event) => {
            const data = JSON.parse((event as MessageEvent).data) as SummaryStreamEvent
            setError(data.message ?? "Error desconocido en el stream")
            source.close()
        })

        source.onerror = (event) => {
            console.error("EventSource error en useSummary:", event)
            setError(NETWORK_ERROR_MESSAGE)
            source.close()
        }

        return () => {
            source.close()
        }
    }, [docId, status])

    return { summary, isDone, resetSummary, error }
}
