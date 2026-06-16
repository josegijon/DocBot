import { useEffect, useRef } from "react"

interface UseFocusTrapOptions {
    isOpen: boolean
    onClose: () => void
}

interface UseFocusTrapReturn<T extends HTMLElement> {
    containerRef: React.RefObject<T>
}

export const useFocusTrap = <T extends HTMLElement>({
    isOpen,
    onClose,
}: UseFocusTrapOptions): UseFocusTrapReturn<T> => {
    const containerRef = useRef<T>(null)
    const previousFocusRef = useRef<HTMLElement | null>(null)

    useEffect(() => {
        if (!isOpen) return

        const activeEl = document.activeElement
        if (activeEl instanceof HTMLElement) {
            previousFocusRef.current = activeEl
        }

        containerRef.current?.focus()

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                e.preventDefault()
                onClose()
                return
            }

            if (e.key !== "Tab") return

            const focusable = containerRef.current?.querySelectorAll<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            )
            if (!focusable || focusable.length === 0) return

            const first = focusable[0]
            const last = focusable[focusable.length - 1]

            if (e.shiftKey) {
                if (document.activeElement === first) {
                    e.preventDefault()
                    last.focus()
                }
            } else {
                if (document.activeElement === last) {
                    e.preventDefault()
                    first.focus()
                }
            }
        }

        document.addEventListener("keydown", handleKeyDown)

        return () => {
            document.removeEventListener("keydown", handleKeyDown)
            previousFocusRef.current?.focus()
        }
    }, [isOpen, onClose])

    return { containerRef }
}
