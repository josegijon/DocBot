import { useCallback, useEffect, useRef, useState } from "react"
import type { Message } from "../types/chat.types"

const VISIBILITY_THRESHOLD = 0.1

interface UseAutoScrollReturn {
    bottomRef: React.RefObject<HTMLDivElement | null>
    showScrollButton: boolean
    scrollToBottom: (behavior?: ScrollBehavior) => void
}

export const useAutoScroll = (messages: Message[]): UseAutoScrollReturn => {
    const [showScrollButton, setShowScrollButton] = useState(false)
    const bottomRef = useRef<HTMLDivElement>(null)
    const showScrollButtonRef = useRef(showScrollButton)

    const lastMessage = messages[messages.length - 1]

    useEffect(() => {
        showScrollButtonRef.current = showScrollButton
    })

    const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
        bottomRef.current?.scrollIntoView({ behavior })
    }, [])

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => setShowScrollButton(!entry.isIntersecting),
            { threshold: VISIBILITY_THRESHOLD }
        )
        if (bottomRef.current) observer.observe(bottomRef.current)
        return () => observer.disconnect()
    }, [])

    useEffect(() => {
        if (!lastMessage) return

        if (lastMessage.role === "user") {
            scrollToBottom("smooth")
        } else if (!showScrollButtonRef.current) {
            scrollToBottom("auto")
        }
    }, [lastMessage, scrollToBottom])

    return { bottomRef, showScrollButton, scrollToBottom }
}
