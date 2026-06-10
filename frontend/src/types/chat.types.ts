export interface Source {
    page: number
    text: string
}

export interface Message {
    role: "user" | "assistant"
    content: string
    sources?: Source[]
}
