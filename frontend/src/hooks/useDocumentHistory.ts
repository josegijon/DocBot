import { useState } from "react";
import { DOCUMENTS_STORAGE_KEY } from "../utils/storageKeys";

export interface DocumentHistory {
    doc_id: string;
    session_id: string;
    filename: string;
    saved_at: string;
}

export const useDocumentHistory = () => {
    const [documents, setDocuments] = useState<DocumentHistory[]>(() => {
        const stored = localStorage.getItem(DOCUMENTS_STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    });

    const addDocument = (doc_id: string, session_id: string, filename: string) => {
        const newDocument = {
            doc_id,
            session_id,
            filename,
            saved_at: new Date().toISOString()
        }

        const updateDocuments = [newDocument, ...documents];

        setDocuments(updateDocuments);
        localStorage.setItem(DOCUMENTS_STORAGE_KEY, JSON.stringify(updateDocuments))
    }

    const removeDocument = (doc_id: string) => {
        const updateDocument = documents.filter((doc: DocumentHistory) => doc.doc_id !== doc_id);
        setDocuments(updateDocument);
        localStorage.setItem(DOCUMENTS_STORAGE_KEY, JSON.stringify(updateDocument))
    }

    const clearHistory = () => {
        setDocuments([]);
        localStorage.removeItem(DOCUMENTS_STORAGE_KEY);
    }

    return { documents, addDocument, removeDocument, clearHistory }
}
