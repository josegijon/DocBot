import { useEffect, useRef, useState } from "react"

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

    const currentSessionRef = useRef(sessionId)
    const isInitialLoadRef = useRef(false)

    useEffect(() => {
        currentSessionRef.current = sessionId

        if (!docId) {
            setMessages([])
            return
        }

        const stored = localStorage.getItem(`docbot_chat_${sessionId}`)
        isInitialLoadRef.current = true
        setMessages(stored ? JSON.parse(stored) : [])

    }, [docId, sessionId])

    useEffect(() => {
        if (isInitialLoadRef.current) {
            isInitialLoadRef.current = false
            return
        }

        if (!sessionId || !docId || messages.length === 0 || sessionId !== currentSessionRef.current) return
        localStorage.setItem(`docbot_chat_${sessionId}`, JSON.stringify(messages))
    }, [messages, sessionId])

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

        let isErrorEvent = false;
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");

            buffer = lines.pop() || "";

            for (const line of lines) {
                const trimmedLine = line.trim();
                if (!trimmedLine) continue;

                if (trimmedLine.startsWith("event: stream_error")) {
                    isErrorEvent = true;
                    continue;
                }

                if (!trimmedLine.startsWith("data:")) continue;

                const jsonStr = trimmedLine.replace("data:", "").trim();
                if (!jsonStr) continue;

                try {
                    const data = JSON.parse(jsonStr);

                    setMessages(prev => {
                        if (prev.length === 0) return prev;

                        const updated = [...prev];
                        const lastIdx = updated.length - 1;
                        const lastMessage = updated[lastIdx];

                        if (isErrorEvent) {
                            updated[lastIdx] = {
                                ...lastMessage,
                                content: lastMessage.content + (data.message || "")
                            };
                        } else {
                            updated[lastIdx] = {
                                ...lastMessage,
                                ...(data.token && { content: lastMessage.content + data.token }),
                                ...(data.sources && { sources: data.sources })
                            };
                        }

                        return updated;
                    });

                    if (isErrorEvent) break;

                } catch (e) {
                    console.error("Error parseando el JSON de SSE:", e);
                }
            }

            if (isErrorEvent) break;
        }

        setIsLoading(false)
    }

    const resetMessages = () => {
        localStorage.removeItem(`docbot_chat_${sessionId}`)
        setMessages([])
    }

    return { messages, isLoading, sendMessage, resetMessages }
}
