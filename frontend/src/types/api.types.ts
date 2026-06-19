export interface ApiErrorResponse {
    error_type: string
    message: string
    path: string
    status_code: number
}

export const isApiErrorResponse = (payload: unknown): payload is ApiErrorResponse => {
    return (
        typeof payload === "object" &&
        payload !== null &&
        typeof (payload as ApiErrorResponse).message === "string"
    )
}
