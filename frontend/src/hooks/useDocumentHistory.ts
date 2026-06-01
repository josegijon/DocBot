import { useState } from "react";

export interface DocumentHistory {
    doc_id: string;
    session_id: string;
    filename: string;
    saved_at: string;
}

export const useDocumentHistory = () => {
    const [documents, setDocuments] = useState<DocumentHistory[]>(() => {
        const stored = localStorage.getItem("docbot_documents");
        return stored ? JSON.parse(stored) : [];
    });

    const addDocument = (doc_id: string, session_id: string, filename: string) => {
        const newDocument = {
            doc_id,
            session_id,
            filename,
            saved_at: new Date().toISOString()
        }

        const updateDocuments = [newDocument, ...documents].slice(0, 10);

        setDocuments(updateDocuments);
        localStorage.setItem("docbot_documents", JSON.stringify(updateDocuments))
    }

    const removeDocument = (doc_id: string) => {
        const updateDocument = documents.filter((doc: DocumentHistory) => doc.doc_id !== doc_id);
        setDocuments(updateDocument);
        localStorage.setItem("docbot_documents", JSON.stringify(updateDocument))
    }

    const clearHistory = () => {
        setDocuments([]);
        localStorage.removeItem("docbot_documents");
    }

    return { documents, addDocument, removeDocument, clearHistory }
}
