import type { RankRequest, RankResponse } from '../types';

const API_BASE = '/api';

/**
 * Rank uploaded documents against a job description.
 *
 * The OpenAI API key is sent per-request inside the FormData body.
 * It is NEVER stored anywhere — not in localStorage, sessionStorage, or cookies.
 */
export async function rankResumes(request: RankRequest): Promise<RankResponse> {
  const formData = new FormData();

  formData.append('job_desc_text', request.jobDescText);
  if (request.jobDescFile) {
    formData.append('job_desc_file', request.jobDescFile);
  }
  formData.append('keywords', request.keywords);
  formData.append('top_n', String(request.topN));
  formData.append('use_llm', String(request.useLlm));
  formData.append('llm_model', request.llmModel);
  // API key travels in the request body only — no persistence.
  formData.append('api_key', request.apiKey);

  for (const file of request.resumeFiles) {
    formData.append('resumes', file);
  }

  const response = await fetch(`${API_BASE}/rank`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(payload.detail ?? `Server error: ${response.status}`);
  }

  return response.json() as Promise<RankResponse>;
}

/** Check that the API backend is alive and the model is loaded. */
export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Backend unreachable');
  return response.json();
}
