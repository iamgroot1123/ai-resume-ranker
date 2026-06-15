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
  llm_fallback?: boolean;
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

/** Application mode */
export type AppMode = 'recruiter' | 'applicant';

/** Toast severity */
export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
}

/** Applicant Mode — request payload */
export interface ApplicantRequest {
  jobDescText: string;
  jobDescFile?: File | null;
  resumeFile: File;
  apiKey: string;
  llmModel: string;
}

/** Applicant Mode — result from POST /api/analyze */
export interface ApplicantResult {
  match_score: number;         // 0–100
  summary: string;
  strengths: string[];
  gaps: string[];
  suggestions: string[];
  keywords_to_add: string[];
  semantic_score: number;      // raw SBERT score 0–100
  llm_used: boolean;
  llm_fallback?: boolean;
  filename: string;
}
