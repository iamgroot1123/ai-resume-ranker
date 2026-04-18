import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Key, Tag, Hash, Zap, ChevronDown, Info } from 'lucide-react';

interface ConfigPanelProps {
  apiKey: string;
  onApiKeyChange: (v: string) => void;
  keywords: string;
  onKeywordsChange: (v: string) => void;
  topN: number;
  onTopNChange: (v: number) => void;
  useLlm: boolean;
  onUseLlmChange: (v: boolean) => void;
  llmModel: string;
  onLlmModelChange: (v: string) => void;
}

export default function ConfigPanel({
  apiKey,
  onApiKeyChange,
  keywords,
  onKeywordsChange,
  topN,
  onTopNChange,
  useLlm,
  onUseLlmChange,
  llmModel,
  onLlmModelChange,
}: ConfigPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="glass-card"
      style={{ border: '1px solid var(--color-border)', overflow: 'hidden' }}
    >
      {/* Accordion header */}
      <button
        onClick={() => setOpen((p) => !p)}
        style={{
          width: '100%',
          padding: '16px 20px',
          background: 'none',
          border: 'none',
          color: 'var(--color-text-primary)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          textAlign: 'left',
          letterSpacing: 0,
        }}
        aria-expanded={open}
        id="config-panel-toggle"
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 'var(--radius-xs)',
            background: 'rgba(124,97,255,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Zap size={15} color="var(--color-violet-light)" />
        </div>
        <div style={{ flex: 1 }}>
          <p style={{ fontWeight: 600, fontSize: '0.9rem' }}>Configuration</p>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: 1 }}>
            LLM mode, keywords, and ranking options
          </p>
        </div>
        <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.22 }}>
          <ChevronDown size={16} color="var(--color-text-muted)" />
        </motion.div>
      </button>

      {/* Accordion body */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            style={{ overflow: 'hidden' }}
          >
            <div
              style={{
                padding: '0 20px 20px',
                display: 'flex',
                flexDirection: 'column',
                gap: 20,
                borderTop: '1px solid var(--color-border)',
                paddingTop: 20,
              }}
            >
              {/* LLM Toggle */}
              <div className="form-field">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <label className="form-label" htmlFor="llm-toggle">
                    <Zap size={13} />
                    LLM Scoring
                  </label>
                  <label className="toggle" htmlFor="llm-toggle">
                    <input
                      id="llm-toggle"
                      type="checkbox"
                      checked={useLlm}
                      onChange={(e) => onUseLlmChange(e.target.checked)}
                    />
                    <span className="toggle__slider" />
                  </label>
                </div>
                {useLlm && (
                  <motion.p
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{
                      fontSize: '0.76rem',
                      color: 'var(--color-text-muted)',
                      display: 'flex',
                      gap: 5,
                      alignItems: 'flex-start',
                    }}
                  >
                    <Info size={11} style={{ marginTop: 2, flexShrink: 0 }} />
                    Uses GPT-3.5-turbo for qualitative fit scores and justifications. Requires an API key.
                  </motion.p>
                )}
              </div>

              {/* API Key */}
              <AnimatePresence>
                {useLlm && (
                  <motion.div
                    key="apikey"
                    className="form-field"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.22 }}
                    style={{ overflow: 'hidden' }}
                  >
                    <label className="form-label" htmlFor="api-key-input">
                      <Key size={13} />
                      API Key (OpenAI or OpenRouter)
                    </label>
                    <input
                      id="api-key-input"
                      type="password"
                      className="input input--password"
                      placeholder="sk-proj-..."
                      value={apiKey}
                      onChange={(e) => onApiKeyChange(e.target.value)}
                      autoComplete="off"
                      spellCheck={false}
                    />
                    <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', display: 'flex', gap: 4 }}>
                      <Key size={10} style={{ marginTop: 2, flexShrink: 0 }} />
                      Your key is sent per-request only and is never stored anywhere.
                    </p>

                    <label className="form-label" htmlFor="model-select" style={{ marginTop: 12 }}>
                      <Zap size={13} />
                      Model Selection
                    </label>
                    <select
                      id="model-select"
                      className="input"
                      value={llmModel}
                      onChange={(e) => onLlmModelChange(e.target.value)}
                    >
                      <option value="gpt-3.5-turbo-1106">GPT-3.5 Turbo</option>
                      <option value="gpt-4o">GPT-4o</option>
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                      <option value="meta-llama/llama-3.3-70b-instruct:free">Llama 3.3 70B (OpenRouter Free)</option>
                    </select>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Keywords */}
              <div className="form-field">
                <label className="form-label" htmlFor="keywords-input">
                  <Tag size={13} />
                  Keyword Filter
                  <span style={{ fontWeight: 400, textTransform: 'none', letterSpacing: 0, fontSize: '0.73rem', color: 'var(--color-text-muted)', marginLeft: 4 }}>
                    (optional)
                  </span>
                </label>
                <input
                  id="keywords-input"
                  type="text"
                  className="input"
                  placeholder="e.g. Python, SQL, AWS  (comma-separated)"
                  value={keywords}
                  onChange={(e) => onKeywordsChange(e.target.value)}
                />
                <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>
                  Only documents containing at least one keyword will be ranked.
                </p>
              </div>

              {/* Top-N slider */}
              <div className="form-field">
                <label className="form-label" htmlFor="topn-slider">
                  <Hash size={13} />
                  Show Top&nbsp;
                  <span style={{ color: 'var(--color-violet-light)', fontWeight: 800, fontSize: '0.9rem', letterSpacing: 0, textTransform: 'none' }}>
                    {topN}
                  </span>
                  &nbsp;Results
                </label>
                <input
                  id="topn-slider"
                  type="range"
                  className="input"
                  min={1}
                  max={30}
                  value={topN}
                  onChange={(e) => onTopNChange(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--color-violet)' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                  <span>1</span>
                  <span>30</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
