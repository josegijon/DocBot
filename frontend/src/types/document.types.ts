import type { IngestionStatus } from "./ingestionStatus.types";

export interface DocumentHistory {
    doc_id: string;
    session_id: string;
    filename: string;
    fileSizeBytes: number;
    saved_at: string;
}

export interface UploadResponse {
    doc_id: string;
    filename: string;
    status: IngestionStatus;
}

export interface DocumentExistsResponse {
    exists: boolean
}
