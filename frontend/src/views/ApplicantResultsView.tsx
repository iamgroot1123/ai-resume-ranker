import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Lightbulb, Tag, ArrowLeft, Cpu } from 'lucide-react';
import type { ApplicantResult } from '../types';

interface ApplicantResultsViewProps {
  result: ApplicantResult;
  onBack: () => void;
}

function ScoreRing({ score }: { score: number }) {
  const r = 54;
  const circ = 2 * Math.PI * r;
  const fill = (score / 100) * circ;
  const color = score >= 70 ? '#22c55e' : score >= 45 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ position: 'relative', width: 150, height: 150 }}>
      <svg width={150} height={150} viewBox="0 0 150 150" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={75} cy={75} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={12} />
        <circle
          cx={75} cy={75} r={r} fill="none"
          stroke={color} strokeWidth={12}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ - fill}
          style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: '2rem', fontWeight: 800, color, lineHeight: 1 }}>{score}</span>
        <span style={{ fontSize: '0.65rem', color: 'var(--color-text-muted)', fontWeight: 600, letterSpacing: 1 }}>/ 100</span>
      </div>
    </div>
  );
}

function Section({
  icon, title, items, accent,
}: {
  icon: React.ReactNode;
  title: string;
  items: string[];
  accent: string;
}) {
  if (!items || items.length === 0) return null;
  return (
    <div
      style={{
        background: 'var(--color-surface)', border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-sm)', padding: 24,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        {icon}
        <h3 style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>{title}</h3>
      </div>
      <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {items.map((item, i) => (
          <li key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', background: accent,
              marginTop: 7, flexShrink: 0,
            }} />
            <span style={{ fontSize: '0.84rem', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ApplicantResultsView({ result, onBack }: ApplicantResultsViewProps) {
  const scoreLabel =
    result.match_score >= 70 ? 'Strong Match' :
    result.match_score >= 45 ? 'Moderate Match' : 'Weak Match';

  const scoreLabelColor =
    result.match_score >= 70 ? '#22c55e' :
    result.match_score >= 45 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
      {/* Back */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        onClick={onBack}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--color-text-muted)', fontSize: '0.82rem', fontWeight: 600,
          marginBottom: 28, padding: 0,
        }}
        whileHover={{ color: 'var(--color-text-primary)' } as never}
      >
        <ArrowLeft size={15} />
        Analyze Another Resume
      </motion.button>

      {/* Fallback Banner */}
      {result.llm_fallback && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)',
            borderRadius: 'var(--radius-sm)', padding: '14px 18px', marginBottom: 24,
            color: '#fbbf24', fontSize: '0.88rem', fontWeight: 500,
          }}
        >
          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
          <span>
            <strong>LLM analysis unavailable.</strong> The AI failed to respond correctly due to load or limits. 
            The feedback below is based on our fallback <strong>SBERT-only</strong> semantic analysis.
          </span>
        </motion.div>
      )}

      {/* Score hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          background: 'var(--color-surface)', border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-md)', padding: '32px 36px',
          display: 'flex', gap: 36, alignItems: 'center', marginBottom: 28,
          flexWrap: 'wrap',
        }}
      >
        <ScoreRing score={result.match_score} />
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{
              fontSize: '1.1rem', fontWeight: 800,
              color: scoreLabelColor,
            }}>{scoreLabel}</span>
            {result.llm_used && (
              <span style={{
                display: 'flex', alignItems: 'center', gap: 5,
                background: 'rgba(124,97,255,0.12)', border: '1px solid rgba(124,97,255,0.25)',
                borderRadius: 20, padding: '2px 10px',
                fontSize: '0.7rem', color: 'var(--color-violet-light)', fontWeight: 600,
              }}>
                <Cpu size={10} /> AI Analysis
              </span>
            )}
          </div>
          <p style={{ fontSize: '0.88rem', color: 'var(--color-text-secondary)', lineHeight: 1.6, marginBottom: 16 }}>
            {result.summary}
          </p>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <div>
              <p style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', fontWeight: 600, marginBottom: 2 }}>FILE</p>
              <p style={{ fontSize: '0.82rem', color: 'var(--color-text-primary)' }}>{result.filename}</p>
            </div>
            <div>
              <p style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', fontWeight: 600, marginBottom: 2 }}>SEMANTIC SCORE</p>
              <p style={{ fontSize: '0.82rem', color: 'var(--color-text-primary)' }}>{result.semantic_score}%</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Detail sections */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        style={{ display: 'grid', gap: 16 }}
      >
        <Section
          icon={<CheckCircle size={16} color="#22c55e" />}
          title="Strengths"
          items={result.strengths}
          accent="#22c55e"
        />
        <Section
          icon={<AlertTriangle size={16} color="#f59e0b" />}
          title="Gaps & Missing Qualifications"
          items={result.gaps}
          accent="#f59e0b"
        />
        <Section
          icon={<Lightbulb size={16} color="#7c61ff" />}
          title="Actionable Suggestions"
          items={result.suggestions}
          accent="#7c61ff"
        />
        {result.keywords_to_add && result.keywords_to_add.length > 0 && (
          <div
            style={{
              background: 'var(--color-surface)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)', padding: 24,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <Tag size={16} color="#38bdf8" />
              <h3 style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                Keywords to Add
              </h3>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {result.keywords_to_add.map((kw, i) => (
                <span key={i} style={{
                  padding: '4px 12px', borderRadius: 20, fontSize: '0.78rem', fontWeight: 600,
                  background: 'rgba(56,189,248,0.1)', border: '1px solid rgba(56,189,248,0.25)',
                  color: '#38bdf8',
                }}>
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* Analyze again */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        onClick={onBack}
        whileHover={{ scale: 1.02 } as never}
        whileTap={{ scale: 0.98 } as never}
        style={{
          width: '100%', marginTop: 28, padding: '14px',
          background: 'rgba(124,97,255,0.1)', border: '1px solid rgba(124,97,255,0.3)',
          borderRadius: 'var(--radius-sm)', color: 'var(--color-violet-light)',
          fontSize: '0.9rem', fontWeight: 700, cursor: 'pointer',
        }}
      >
        ← Analyze Another Resume
      </motion.button>
    </div>
  );
}
