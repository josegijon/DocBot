import { useChat } from "../hooks/useChat"
import { MessageBubble } from "./MessageBubble"
import { InputBar } from "./InputBar"
import { useEffect, useRef } from "react"

interface ChatWindowProps {
    docId: string | null
    sessionId: string
}

export const ChatWindow = ({ docId, sessionId }: ChatWindowProps) => {
    const { messages, isLoading, sendMessage } = useChat(docId, sessionId)

    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    return (
        <div className="flex-1 flex flex-col h-full relative">
            {/* Lista de mensajes */}
            <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-10 chat-scrollbar">
                {messages.length === 0 && (
                    <p className="text-on-surface-variant text-center">
                        Haz una pregunta sobre el documento
                    </p>
                )}
                {messages.map((message, index) => (
                    <MessageBubble key={index} message={message} />
                ))}
                <div ref={bottomRef} />
            </div>

            <InputBar onSend={sendMessage} isLoading={isLoading} />
        </div>
    )
}