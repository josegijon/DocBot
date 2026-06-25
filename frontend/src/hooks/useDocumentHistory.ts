import { useCallback, useState } from "react";

import { DOCUMENTS_STORAGE_KEY, getChatStorageKey, getSummaryStorageKey } from "../utils/storageKeys";
import type { DocumentHistory } from "../types/document.types";

interface UseDocumentHistoryReturn {
    documents: DocumentHistory[];
    addDocument: (doc_id: string, session_id: string, filename: string) => void;
    removeDocument: (doc_id: string) => void;
    clearHistory: () => void;
}

const saveDocuments = (docs: DocumentHistory[]): void => {
    localStorage.setItem(DOCUMENTS_STORAGE_KEY, JSON.stringify(docs));
};

const loadDocuments = (): DocumentHistory[] => {
    try {
        const stored = localStorage.getItem(DOCUMENTS_STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    } catch {
        localStorage.removeItem(DOCUMENTS_STORAGE_KEY);
        return [];
    }
};

export const useDocumentHistory = (): UseDocumentHistoryReturn => {
    const [documents, setDocuments] = useState<DocumentHistory[]>(loadDocuments);

    const addDocument = useCallback((doc_id: string, session_id: string, filename: string) => {
        const newDocument: DocumentHistory = {
            doc_id,
            session_id,
            filename,
            saved_at: new Date().toISOString()
        };

        setDocuments(prev => {
            if (prev.some(doc => doc.doc_id === doc_id)) return prev;
            const next = [newDocument, ...prev];
            saveDocuments(next);
            return next;
        });
    }, []);

    const removeDocument = useCallback((doc_id: string) => {
        setDocuments(prev => {
            const docToRemove = prev.find(doc => doc.doc_id === doc_id)
            if (docToRemove) {
                localStorage.removeItem(getSummaryStorageKey(docToRemove.doc_id))
                localStorage.removeItem(getChatStorageKey(docToRemove.session_id))
            }

            const next = prev.filter((doc: DocumentHistory) => doc.doc_id !== doc_id);
            saveDocuments(next);
            return next;
        });
    }, []);

    const clearHistory = useCallback(() => {
        setDocuments([]);
        localStorage.removeItem(DOCUMENTS_STORAGE_KEY);
    }, []);

    return { documents, addDocument, removeDocument, clearHistory }
}
