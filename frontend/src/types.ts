/** A single ranked candidate returned from the API */
export interface RankedCandidate {
  id: string;
  similarity: number;
  email: string;
  rating_10: number;
  key_matches: string;
  justification: string;
  skills: string;
  experience: string;
  education: string;
  /** Base64-encoded original file bytes for stateless client-side download */
  file_bytes_b64: string;
  file_type: 'pdf' | 'txt';
}

/** Full API response from POST /api/rank */
export interface RankResponse {
  candidates: RankedCandidate[];
  total_uploaded: number;
  total_returned: number;
  scoring_mode: 'llm+sbert' | 'sbert';
}

/** Request payload (maps to FormData sent to the backend) */
export interface RankRequest {
  jobDescText: string;
  jobDescFile?: File | null;
  keywords: string;
  topN: number;
  useLlm: boolean;
  llmModel: string;
  apiKey: string;
  resumeFiles: File[];
}

/** App view state */
export type AppView = 'upload' | 'results';

/** Toast severity */
export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
}
