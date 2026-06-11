import { useEffect, useState } from "react"

interface IngestionStatus {
    status: "processing" | "ready" | "failed"
    progress: number
}

interface UseIngestionStatusReturn extends IngestionStatus {
    resetStatus: () => void;
}

export const useIngestionStatus = (docId: string | null, isKnownReady: boolean = false): UseIngestionStatusReturn => {
    const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus>({
        status: "processing",
        progress: 0,
    })

    const resetStatus = () => {
        setIngestionStatus({ status: "processing", progress: 0 })
    }

    useEffect(() => {
        if (!docId) return;

        if (isKnownReady) {
            setIngestionStatus({ status: "ready", progress: 100 });
            return;
        }

        const source = new EventSource(`/api/documents/${docId}/status`);

        source.onmessage = (event) => {
            const data = JSON.parse(event.data) as IngestionStatus;
            console.log(data);
            setIngestionStatus(data);

            if (data.progress >= 100 || data.status === "failed") {
                source.close();
            };
        }

        source.onerror = () => {
            source.close();
        };

        return () => {
            source.close()
        }
    }, [docId, isKnownReady])

    return { ...ingestionStatus, resetStatus }
}
