import { ArrowDown } from "lucide-react"

interface ScrollToBottomProps {
    onClick: () => void
    isVisible: boolean
}

const SCROLL_TO_BOTTOM_BUTTON_ID = "scroll-to-bottom-btn"

const VISIBLE_STATE_CLASSES = "opacity-90 translate-y-0"
const HIDDEN_STATE_CLASSES = "opacity-0 translate-y-4 pointer-events-none"

export const ScrollToBottom = ({ onClick, isVisible }: ScrollToBottomProps) => {
    const visibilityClass = isVisible ? VISIBLE_STATE_CLASSES : HIDDEN_STATE_CLASSES

    return (
        <button
            className={`absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center justify-center w-10 h-10 rounded-full bg-surface-bright border border-outline-variant shadow-lg text-on-surface hover:bg-primary hover:text-on-primary transition-all duration-300 z-10 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${visibilityClass}`}
            id={SCROLL_TO_BOTTOM_BUTTON_ID}
            inert={!isVisible}
            onClick={onClick}
        >
            <ArrowDown />
        </button>
    )
}
