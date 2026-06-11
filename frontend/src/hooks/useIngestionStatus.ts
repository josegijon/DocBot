import { useCallback, useEffect, useState } from "react"
import type { IngestionStatus } from "../types/ingestionStatus.types";

interface IngestionState {
    status: IngestionStatus
    progress: number
}

interface UseIngestionStatusReturn extends IngestionState {
    resetStatus: () => void;
}

export const useIngestionStatus = (docId: string | null, isKnownReady: boolean = false): UseIngestionStatusReturn => {
    const [ingestionStatus, setIngestionStatus] = useState<IngestionState>(() =>
        isKnownReady
            ? { status: "ready", progress: 100 }
            : { status: "processing", progress: 0 }
    );

    const resetStatus = useCallback(() => {
        setIngestionStatus({ status: "processing", progress: 0 })
    }, []);

    useEffect(() => {
        if (!docId) return;

        const source = new EventSource(`/api/documents/${docId}/status`);

        source.onmessage = (event) => {
            const data = JSON.parse(event.data) as IngestionState;
            setIngestionStatus(data);

            if (data.progress >= 100 || data.status === "failed") {
                source.close();
            };
        }

        source.onerror = () => {
            setIngestionStatus({ status: "failed", progress: 0 })
            source.close();
        };

        return () => {
            source.close()
        }
    }, [docId, isKnownReady])

    return { ...ingestionStatus, resetStatus }
}
