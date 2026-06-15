import { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { FileText, ArrowRight, AlertCircle, UploadCloud, Sparkles } from 'lucide-react';
import FileDropZone from '../components/FileDropZone';
import ConfigPanel from '../components/ConfigPanel';
import type { RankRequest } from '../types';

interface UploadViewProps {
  onSubmit: (req: RankRequest) => void;
  isLoading: boolean;
  error: string | null;
}

export default function UploadView({ onSubmit, isLoading, error }: UploadViewProps) {
  const [jobDescText, setJobDescText] = useState('');
  const [jobDescFile, setJobDescFile] = useState<File | null>(null);
  const [resumeFiles, setResumeFiles] = useState<File[]>([]);
  const [apiKey, setApiKey] = useState('');
  const [keywords, setKeywords] = useState('');
  const [topN, setTopN] = useState(5);
  const [useLlm, setUseLlm] = useState(false);
  const llmModel = 'gpt-3.5-turbo-1106';
  const jdFileRef = useRef<HTMLInputElement>(null);

  const charCount = jobDescText.length;
  const hasJD = !!jobDescText.trim() || !!jobDescFile;
  const hasResumes = resumeFiles.length > 0;
  const isReady = hasJD && hasResumes && (!useLlm || !!apiKey.trim());

  const handleSubmit = useCallback(() => {
    if (!isReady || isLoading) return;
    onSubmit({
      jobDescText,
      jobDescFile,
      keywords,
      topN,
      useLlm,
      llmModel,
      apiKey,
      resumeFiles,
    });
  }, [isReady, isLoading, onSubmit, jobDescText, jobDescFile, keywords, topN, useLlm, llmModel, apiKey, resumeFiles]);

  return (
    <div className="container" style={{ paddingTop: 48, paddingBottom: 64 }}>
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        style={{ textAlign: 'center', marginBottom: 52 }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '5px 14px',
            borderRadius: 'var(--radius-full)',
            background: 'rgba(124,97,255,0.1)',
            border: '1px solid rgba(124,97,255,0.25)',
            marginBottom: 20,
          }}
        >
          <Sparkles size={13} color="var(--color-violet-light)" />
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-violet-light)', letterSpacing: '0.05em' }}>
            AI-Powered Semantic Matching
          </span>
        </div>

        <h1
          style={{
            fontFamily: 'var(--font-heading)',
            fontSize: 'clamp(2rem, 5vw, 3rem)',
            fontWeight: 800,
            lineHeight: 1.15,
            marginBottom: 16,
          }}
        >
          Find Your{' '}
          <span className="gradient-text">Perfect Candidates</span>
        </h1>
        <p
          style={{
            fontSize: '1.05rem',
            color: 'var(--color-text-secondary)',
            maxWidth: 560,
            margin: '0 auto',
            lineHeight: 1.65,
          }}
        >
          Upload a job description and candidate documents. Our hybrid
          SBERT&nbsp;+&nbsp;LLM pipeline ranks them by semantic fit — not just keywords.
        </p>
      </motion.div>



      {/* Main two-column input area */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 24,
          marginBottom: 24,
        }}
      >
        {/* Left: Job Description */}
        <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(0,212,170,0.12)',
                border: '1px solid rgba(0,212,170,0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <FileText size={16} color="var(--color-teal-light)" />
            </div>
            <div>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
                Job Description
              </h2>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 1 }}>
                Paste text or upload a .txt file
              </p>
            </div>
            <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>
              {charCount > 0 ? `${charCount} chars` : ''}
            </span>
          </div>

          <div className="form-field">
            <textarea
              id="jd-textarea"
              className="input textarea"
              placeholder="Paste the full job description here…&#10;&#10;e.g. We are looking for a Machine Learning Engineer with experience in PyTorch, AWS, and distributed training…"
              value={jobDescText}
              onChange={(e) => {
                setJobDescText(e.target.value);
                if (e.target.value) setJobDescFile(null);
              }}
              style={{ minHeight: 220, flex: 1 }}
              aria-label="Job description text"
            />
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              color: 'var(--color-text-muted)',
              fontSize: '0.78rem',
            }}
          >
            <hr className="divider" style={{ flex: 1 }} />
            <span>OR</span>
            <hr className="divider" style={{ flex: 1 }} />
          </div>

          {/* JD file upload */}
          <div>
            <input
              ref={jdFileRef}
              type="file"
              accept=".txt"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setJobDescFile(f);
                if (f) setJobDescText('');
              }}
              aria-label="Upload job description file"
            />
            <button
              className="btn btn--ghost btn--sm"
              style={{ width: '100%' }}
              onClick={() => jdFileRef.current?.click()}
            >
              <UploadCloud size={14} />
              {jobDescFile ? jobDescFile.name : 'Upload .txt file'}
            </button>
            {jobDescFile && (
              <button
                className="btn btn--sm"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--color-error)',
                  marginTop: 6,
                  padding: '2px 6px',
                  fontSize: '0.72rem',
                  cursor: 'pointer',
                }}
                onClick={() => {
                  setJobDescFile(null);
                  if (jdFileRef.current) jdFileRef.current.value = '';
                }}
              >
                ✕ Remove file
              </button>
            )}
          </div>
        </div>

        {/* Right: Resume Upload */}
        <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(124,97,255,0.12)',
                border: '1px solid rgba(124,97,255,0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <UploadCloud size={16} color="var(--color-violet-light)" />
            </div>
            <div>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
                Candidate Documents
              </h2>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 1 }}>
                Upload PDF or TXT resumes
              </p>
            </div>
            {resumeFiles.length > 0 && (
              <span
                className="chip chip--violet"
                style={{ marginLeft: 'auto', fontSize: '0.72rem' }}
              >
                {resumeFiles.length} ready
              </span>
            )}
          </div>

          <FileDropZone
            files={resumeFiles}
            onChange={setResumeFiles}
            label="Resume upload"
          />
        </div>
      </motion.div>

      {/* Configuration accordion */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.18 }}
        style={{ marginBottom: 32 }}
      >
        <ConfigPanel
          apiKey={apiKey}
          onApiKeyChange={setApiKey}
          keywords={keywords}
          onKeywordsChange={setKeywords}
          topN={topN}
          onTopNChange={setTopN}
          useLlm={useLlm}
          onUseLlmChange={setUseLlm}
        />
      </motion.div>

      {/* Submit CTA */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.25 }}
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}
      >
        {/* Error banner */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '12px 18px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              marginBottom: 12,
              color: '#fca5a5',
              fontSize: '0.88rem',
            }}
          >
            <AlertCircle size={16} style={{ flexShrink: 0 }} />
            {error}
          </motion.div>
        )}

        <motion.button
          className="btn btn--primary btn--lg"
          onClick={handleSubmit}
          disabled={!isReady || isLoading}
          whileHover={isReady && !isLoading ? { scale: 1.02 } : {}}
          whileTap={isReady && !isLoading ? { scale: 0.97 } : {}}
          style={{
            minWidth: 260,
            gap: 12,
            fontSize: '1.05rem',
            fontFamily: 'var(--font-heading)',
          }}
          aria-label="Rank candidates"
        >
          <Sparkles size={18} />
          Analyze Candidates
          <ArrowRight size={18} />
        </motion.button>

        {/* Readiness hints */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
          <span className={`chip ${hasJD ? 'chip--success' : 'chip--neutral'}`} style={{ fontSize: '0.72rem' }}>
            {hasJD ? '✓' : '○'} Job description
          </span>
          <span className={`chip ${hasResumes ? 'chip--success' : 'chip--neutral'}`} style={{ fontSize: '0.72rem' }}>
            {hasResumes ? `✓ ${resumeFiles.length} resumes` : '○ Resumes'}
          </span>
          {useLlm && (
            <span className={`chip ${apiKey ? 'chip--success' : 'chip--error'}`} style={{ fontSize: '0.72rem' }}>
              {apiKey ? '✓' : '!'} API key
            </span>
          )}
        </div>
      </motion.div>

      {/* Responsive overrides */}
      <style>{`
        @media (max-width: 720px) {
          .upload-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
