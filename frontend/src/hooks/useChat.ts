import { useEffect, useRef, useState } from "react"
import type { Message } from "../types/chat.types"
import { getChatStorageKey } from "../utils/storageKeys"

const SSE_EVENT_STREAM_ERROR = "event: stream_error"
const SSE_DATA_PREFIX = "data:"

export const useChat = (docId: string | null, sessionId: string) => {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState<boolean>(false)

    const currentSessionRef = useRef(sessionId)
    const isInitialLoadRef = useRef(false)
    const abortControllerRef = useRef<AbortController | null>(null)

    useEffect(() => {
        currentSessionRef.current = sessionId

        if (!docId) {
            setMessages([])
            return
        }

        const stored = localStorage.getItem(getChatStorageKey(sessionId))
        isInitialLoadRef.current = true
        setMessages(stored ? JSON.parse(stored) : [])

        return () => {
            abortControllerRef.current?.abort()
        }
    }, [docId, sessionId])

    useEffect(() => {
        if (isInitialLoadRef.current) {
            isInitialLoadRef.current = false
            return
        }

        if (!sessionId || !docId || messages.length === 0 || sessionId !== currentSessionRef.current) return

        const lastMessage = messages[messages.length - 1]
        if (lastMessage.role === 'assistant' && lastMessage.content === '') return
        if (lastMessage.role === 'assistant' && !lastMessage.sources) return

        localStorage.setItem(getChatStorageKey(sessionId), JSON.stringify(messages))
    }, [messages, sessionId])

    const sendMessage = async (userMessage: string) => {
        if (!docId || isLoading) return

        setMessages(prev => [...prev, { role: "user", content: userMessage }])
        setMessages(prev => [...prev, { role: "assistant", content: "" }])
        setIsLoading(true)

        try {
            abortControllerRef.current?.abort()
            abortControllerRef.current = new AbortController()

            const response = await fetch("/api/chat/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: abortControllerRef.current.signal,
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

                    if (trimmedLine.startsWith(SSE_EVENT_STREAM_ERROR)) {
                        isErrorEvent = true;
                        continue;
                    }

                    if (!trimmedLine.startsWith(SSE_DATA_PREFIX)) continue;

                    const jsonStr = trimmedLine.replace(SSE_DATA_PREFIX, "").trim();
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

                    } catch (parseError) {
                        console.error("Error parseando el JSON de SSE:", parseError);
                    }
                }

                if (isErrorEvent) break;
            }

            setIsLoading(false)
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                setIsLoading(false)
                setMessages(prev => {
                    const lastMessage = prev[prev.length - 1]
                    if (lastMessage.role === 'assistant' && !lastMessage.sources) {
                        const updated = prev.slice(0, -1)
                        return updated
                    }
                    return prev
                })
                return
            }
            setIsLoading(false)
            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: "assistant", content: "No se pudo conectar con el servidor. Por favor, inténtalo de nuevo." };
                return updated;
            })
        }
    }

    const resetMessages = () => {
        localStorage.removeItem(getChatStorageKey(sessionId))
        setMessages([])
    }

    return { messages, isLoading, sendMessage, resetMessages }
}
