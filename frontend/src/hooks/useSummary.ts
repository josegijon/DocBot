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

        const cached = localStorage.getItem(`docbot_summary_${docId}`)
        if (cached) {
            setSummary(cached)
            setIsDone(true)
            return
        }

        const source = new EventSource(`/api/documents/${docId}/summary`)

        let accumulatedSummary = ""

        source.onmessage = (event) => {
            const data = JSON.parse(event.data)

            if (data.done) {
                localStorage.setItem(`docbot_summary_${docId}`, accumulatedSummary)
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
            const data = JSON.parse(event.data)
            setError(data.message)
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
