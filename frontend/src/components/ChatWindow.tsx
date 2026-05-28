import { useChat } from "../hooks/useChat"
import { MessageBubble } from "./MessageBubble"
import { InputBar } from "./InputBar"
import { useEffect, useRef } from "react"
import { ChatOnboarding } from "./ChatOnboarding"

interface ChatWindowProps {
    docId: string | null
    sessionId: string
    filename: string | null
}

export const ChatWindow = ({ docId, sessionId, filename }: ChatWindowProps) => {
    const { messages, isLoading, sendMessage } = useChat(docId, sessionId)

    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    return (
        <div className="flex-1 flex flex-col h-full relative">
            <div className="relative flex-1 overflow-hidden">
                <div className="flex-1 h-full overflow-y-auto p-8 flex flex-col gap-10 chat-scrollbar">
                    {messages.length === 0 && (
                        <ChatOnboarding filename={filename} />
                    )}
                    {messages.map((message, index) => (
                        <MessageBubble key={index} message={message} />
                    ))}
                    <div ref={bottomRef} />
                </div>

                <div className="absolute bottom-0 left-0 right-0 h-12 bg-linear-to-t from-surface to-transparent pointer-events-none" />
            </div>

            <InputBar onSend={sendMessage} isLoading={isLoading} disabled={!docId} />
        </div>
    )
}