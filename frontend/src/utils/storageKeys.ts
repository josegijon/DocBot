export const getChatStorageKey = (sessionId: string): string =>
    `docbot_chat_${sessionId}`

export const getSummaryStorageKey = (documentId: string): string =>
    `docbot_summary_${documentId}`

export const DOCUMENTS_STORAGE_KEY = "docbot_documents"
