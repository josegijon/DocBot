import { useEffect, useState } from "react"

export const useSummary = (docId: string | null, status: string) => {
    const [summary, setSummary] = useState<string>("")
    const [isDone, setIsDone] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    const resetSummary = () => {
        setSummary("")
        setError(null)
        setIsDone(false)
    }

    useEffect(() => {
        if (!docId || status !== "ready") return

        const source = new EventSource(`/api/documents/${docId}/summary`)

        source.onmessage = (event) => {
            const data = JSON.parse(event.data)

            if (data.done) {
                setIsDone(true)
                source.close()
                return
            }

            if (data.token) {
                setSummary((prev) => prev + data.token)
            }
        }

        source.onerror = () => {
            source.close()
        }

        source.addEventListener("stream_error", (event) => {
            const data = JSON.parse(event.data)
            setError(data.message)
            source.close()
        })

        return () => {
            source.close()
        }
    }, [docId, status])

    return { summary, isDone, resetSummary, error }
}
