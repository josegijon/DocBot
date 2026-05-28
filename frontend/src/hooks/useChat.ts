import { useState } from "react"

interface Source {
    page: number
    text: string
}

interface Message {
    role: "user" | "assistant"
    content: string
    sources?: Source[]
}

export const useChat = (docId: string | null, sessionId: string) => {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState<boolean>(false)

    const sendMessage = async (userMessage: string) => {
        if (!docId || isLoading) return

        setMessages(prev => [...prev, { role: "user", content: userMessage }])

        setMessages(prev => [...prev, { role: "assistant", content: "" }])

        setIsLoading(true)

        const response = await fetch("/api/chat/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                doc_id: docId,
                session_id: sessionId,
                message: userMessage,
            }),
        })

        const reader = response.body!.getReader()
        const decoder = new TextDecoder()

        while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const chunk = decoder.decode(value)
            const lines = chunk.split("\n")

            for (const line of lines) {
                if (!line.startsWith("data:")) continue

                const jsonStr = line.replace("data:", "").trim()
                if (!jsonStr) continue

                try {
                    const data = JSON.parse(jsonStr)

                    if (data.token) {
                        setMessages(prev => {
                            const updated = [...prev]
                            updated[updated.length - 1] = {
                                ...updated[updated.length - 1],
                                content: updated[updated.length - 1].content + data.token,
                            }
                            return updated
                        })
                    }

                    if (data.sources) {
                        setMessages(prev => {
                            const updated = [...prev]
                            updated[updated.length - 1] = {
                                ...updated[updated.length - 1],
                                sources: data.sources,
                            }
                            return updated
                        })
                    }
                } catch {
                    // chunk parcial, ignorar
                }
            }
        }

        setIsLoading(false)
    }

    return { messages, isLoading, sendMessage }
}