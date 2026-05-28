import { ArrowDown } from "lucide-react"

interface ScrollToBottomProps {
    onClick: () => void
    isVisible: boolean
}

export const ScrollToBottom = ({ onClick, isVisible }: ScrollToBottomProps) => {
    return (
        <button
            className={`absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center justify-center w-10 h-10 rounded-full bg-surface-bright border border-outline-variant shadow-lg text-on-surface hover:bg-primary hover:text-on-primary transition-all duration-300 z-10 cursor-pointer ${isVisible
                ? "opacity-90 translate-y-0"
                : "opacity-0 translate-y-4 pointer-events-none"
                }`}
            id="scroll-to-bottom-btn"
            onClick={onClick}
        >
            <ArrowDown />
        </button>
    )
}
