import { useEffect, useRef } from "react"

const FOCUSABLE_ELEMENTS_SELECTOR = [
    'button',
    '[href]',
    'input',
    'select',
    'textarea',
    '[tabindex]:not([tabindex="-1"])',
].join(', ')

const KEY_ESCAPE = "Escape"
const KEY_TAB = "Tab"

interface UseFocusTrapOptions {
    isOpen: boolean
    onClose: () => void
}

interface UseFocusTrapReturn<T extends HTMLElement> {
    containerRef: React.RefObject<T | null>
}

export const useFocusTrap = <T extends HTMLElement>({
    isOpen,
    onClose,
}: UseFocusTrapOptions): UseFocusTrapReturn<T> => {
    const containerRef = useRef<T>(null)
    const previousFocusRef = useRef<HTMLElement | null>(null)
    const onCloseRef = useRef(onClose)

    useEffect(() => {
        onCloseRef.current = onClose
    })

    useEffect(() => {
        if (!isOpen) return

        const activeEl = document.activeElement
        if (activeEl instanceof HTMLElement) {
            previousFocusRef.current = activeEl
        }

        containerRef.current?.focus()

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === KEY_ESCAPE) {
                e.preventDefault()
                onCloseRef.current()
                return
            }

            if (e.key !== KEY_TAB) return

            const focusable = containerRef.current?.querySelectorAll<HTMLElement>(
                FOCUSABLE_ELEMENTS_SELECTOR
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
    }, [isOpen])

    return { containerRef }
}
