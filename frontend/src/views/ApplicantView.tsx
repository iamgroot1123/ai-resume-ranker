import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, X, Sparkles, Briefcase, UploadCloud } from 'lucide-react';
import type { ApplicantRequest } from '../types';

interface ApplicantViewProps {
  onSubmit: (req: ApplicantRequest) => void;
  isLoading: boolean;
  error: string | null;
}

export default function ApplicantView({ onSubmit, isLoading, error }: ApplicantViewProps) {
  const [jobDesc, setJobDesc] = useState('');
  const [jobDescFile, setJobDescFile] = useState<File | null>(null);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [apiKey, setApiKey] = useState('');
  const llmModel = 'gpt-3.5-turbo-1106';
  const [useLlm, setUseLlm] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const jdFileRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.pdf') || file.name.endsWith('.txt'))) {
      setResumeFile(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setResumeFile(file);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((!jobDesc.trim() && !jobDescFile) || !resumeFile) return;
    onSubmit({
      jobDescText: jobDesc,
      jobDescFile,
      resumeFile,
      apiKey: useLlm ? apiKey : '',
      llmModel: useLlm ? llmModel : '',
    });
  };

  const hasJD = !!jobDesc.trim() || !!jobDescFile;
  const canSubmit = hasJD && resumeFile !== null && (!useLlm || !!apiKey.trim());

  return (
    <div style={{ maxWidth: 780, margin: '0 auto', padding: '40px 24px' }}>
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ textAlign: 'center', marginBottom: 40 }}
      >
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          background: 'rgba(124,97,255,0.12)', border: '1px solid rgba(124,97,255,0.3)',
          borderRadius: 20, padding: '4px 14px', marginBottom: 16,
        }}>
          <Sparkles size={13} color="var(--color-violet-light)" />
          <span style={{ fontSize: '0.76rem', color: 'var(--color-violet-light)', fontWeight: 600 }}>
            Applicant Mode
          </span>
        </div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: 10, background: 'linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.6) 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          How well does your resume fit?
        </h1>
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.95rem', maxWidth: 500, margin: '0 auto' }}>
          Paste the job description and upload your resume. Get a match score, strengths, gaps, and actionable suggestions.
        </p>
      </motion.div>

      <form onSubmit={handleSubmit}>


        {/* Job Description */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card" style={{ padding: 24, marginBottom: 20 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            <Briefcase size={15} color="var(--color-violet-light)" />
            Job Description <span style={{ color: 'var(--color-violet-light)' }}>*</span>
          </label>
          <textarea
            id="applicant-jd"
            value={jobDesc}
            onChange={(e) => {
              setJobDesc(e.target.value);
              if (e.target.value) setJobDescFile(null);
            }}
            placeholder="Paste the full job description here…"
            required={!jobDescFile}
            rows={8}
            style={{
              width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-xs)', color: 'var(--color-text-primary)',
              fontSize: '0.85rem', padding: '12px 14px', resize: 'vertical',
              fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box',
              transition: 'border-color 0.2s',
              marginBottom: 16,
            }}
            onFocus={(e) => (e.target.style.borderColor = 'var(--color-violet)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--color-border)')}
          />
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              color: 'var(--color-text-muted)',
              fontSize: '0.78rem',
              marginBottom: 16,
            }}
          >
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--color-border)' }} />
            <span>OR</span>
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--color-border)' }} />
          </div>
          <div>
            <input
              ref={jdFileRef}
              type="file"
              accept=".txt"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setJobDescFile(f);
                if (f) setJobDesc('');
              }}
            />
            <button
              type="button"
              onClick={() => jdFileRef.current?.click()}
              style={{
                width: '100%',
                padding: '10px 16px',
                borderRadius: 'var(--radius-xs)',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid var(--color-border)',
                color: 'var(--color-text-primary)',
                fontSize: '0.82rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                transition: 'background 0.2s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.08)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.05)')}
            >
              <UploadCloud size={16} />
              {jobDescFile ? jobDescFile.name : 'Upload JD .txt file'}
            </button>
            {jobDescFile && (
              <button
                type="button"
                onClick={() => {
                  setJobDescFile(null);
                  if (jdFileRef.current) jdFileRef.current.value = '';
                }}
                style={{
                  background: 'none', border: 'none', color: '#f87171',
                  marginTop: 8, fontSize: '0.72rem', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 4,
                }}
              >
                <X size={12} /> Remove file
              </button>
            )}
          </div>
        </motion.div>

        {/* Resume Upload */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="glass-card" style={{ padding: 24, marginBottom: 20 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            <FileText size={15} color="var(--color-violet-light)" />
            Your Resume <span style={{ color: 'var(--color-violet-light)' }}>*</span>
          </label>

          {resumeFile ? (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '14px 16px', borderRadius: 'var(--radius-xs)',
              background: 'rgba(124,97,255,0.08)', border: '1px solid rgba(124,97,255,0.25)',
            }}>
              <FileText size={18} color="var(--color-violet-light)" />
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>{resumeFile.name}</p>
                <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>{(resumeFile.size / 1024).toFixed(1)} KB</p>
              </div>
              <button
                type="button"
                onClick={() => setResumeFile(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)', padding: 4 }}
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleFileDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? 'var(--color-violet)' : 'var(--color-border)'}`,
                borderRadius: 'var(--radius-xs)', padding: '36px 24px',
                textAlign: 'center', cursor: 'pointer',
                background: dragging ? 'rgba(124,97,255,0.06)' : 'rgba(0,0,0,0.1)',
                transition: 'all 0.2s',
              }}
            >
              <Upload size={28} color={dragging ? 'var(--color-violet-light)' : 'var(--color-text-muted)'} style={{ margin: '0 auto 10px' }} />
              <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>
                Drop your resume here or <span style={{ color: 'var(--color-violet-light)', fontWeight: 600 }}>click to browse</span>
              </p>
              <p style={{ fontSize: '0.73rem', color: 'var(--color-text-muted)', marginTop: 6 }}>PDF or TXT · Max 5 MB</p>
              <input ref={fileRef} type="file" accept=".pdf,.txt" onChange={handleFileSelect} style={{ display: 'none' }} />
            </div>
          )}
        </motion.div>

        {/* LLM Config */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card" style={{ padding: 24, marginBottom: 28 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: useLlm ? 20 : 0 }}>
            <div>
              <p style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Enable AI Analysis</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 3 }}>
                Get detailed strengths, gaps &amp; suggestions (requires API key)
              </p>
            </div>
            <button
              type="button"
              id="applicant-llm-toggle"
              onClick={() => setUseLlm((p) => !p)}
              role="switch"
              aria-checked={useLlm}
              style={{
                width: 44, height: 24, borderRadius: 12, border: 'none', cursor: 'pointer',
                background: useLlm ? 'var(--color-violet)' : 'rgba(255,255,255,0.1)',
                position: 'relative', transition: 'background 0.2s', flexShrink: 0,
              }}
            >
              <span style={{
                position: 'absolute', top: 3, left: useLlm ? 23 : 3, width: 18, height: 18,
                borderRadius: '50%', background: '#fff', transition: 'left 0.2s',
              }} />
            </button>
          </div>

          {useLlm && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: 6, display: 'block' }}>API Key</label>
                <input
                  id="applicant-api-key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-proj-..."
                  style={{
                    width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-xs)', color: 'var(--color-text-primary)',
                    fontSize: '0.83rem', padding: '9px 12px', outline: 'none', boxSizing: 'border-box',
                  }}
                />
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Error banner */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--radius-sm)', padding: '12px 16px', marginBottom: 20,
              color: '#f87171', fontSize: '0.85rem',
            }}
          >
            {error}
          </motion.div>
        )}

        {/* Submit */}
        <motion.button
          id="applicant-submit-btn"
          type="submit"
          disabled={!canSubmit || isLoading}
          whileHover={canSubmit ? { scale: 1.02 } : {}}
          whileTap={canSubmit ? { scale: 0.98 } : {}}
          style={{
            width: '100%', padding: '15px 24px',
            background: canSubmit ? 'var(--color-violet)' : 'rgba(255,255,255,0.06)',
            border: 'none', borderRadius: 'var(--radius-sm)',
            color: canSubmit ? '#fff' : 'var(--color-text-muted)',
            fontSize: '0.95rem', fontWeight: 700, cursor: canSubmit ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s',
          }}
        >
          {isLoading ? 'Analyzing…' : '✦ Analyze My Resume'}
        </motion.button>
      </form>
    </div>
  );
}
