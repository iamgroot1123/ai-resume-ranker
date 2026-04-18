import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronDown,
  Mail,
  Download,
  Cpu,
  Briefcase,
  GraduationCap,
  Zap,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import type { RankedCandidate } from '../types';
import ScoreGauge from './ScoreGauge';

interface CandidateCardProps {
  candidate: RankedCandidate;
  rank: number;
  index: number;
  scoringMode: string;
  defaultExpanded?: boolean;
}

function rankBadgeClass(rank: number): string {
  if (rank === 1) return 'rank-badge rank-badge--1';
  if (rank === 2) return 'rank-badge rank-badge--2';
  if (rank === 3) return 'rank-badge rank-badge--3';
  return 'rank-badge rank-badge--default';
}

function rankEmoji(rank: number): string {
  if (rank === 1) return '🥇';
  if (rank === 2) return '🥈';
  if (rank === 3) return '🥉';
  return `#${rank}`;
}

function StructuredSection({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  const hasData = text && text !== 'Not specified';
  const items = hasData
    ? text.split(/[,|•]/).map((s) => s.trim()).filter(Boolean)
    : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        fontSize: '0.75rem',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: 'var(--color-text-muted)',
      }}>
        {icon}
        {title}
        {hasData ? (
          <CheckCircle2 size={11} color="var(--color-success)" style={{ marginLeft: 'auto' }} />
        ) : (
          <XCircle size={11} color="var(--color-error)" style={{ marginLeft: 'auto' }} />
        )}
      </div>

      {hasData ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {items.slice(0, 10).map((item, i) => (
            <span key={i} className="chip chip--neutral" style={{ fontSize: '0.72rem' }}>
              {item}
            </span>
          ))}
          {items.length > 10 && (
            <span className="chip chip--neutral" style={{ fontSize: '0.72rem', opacity: 0.6 }}>
              +{items.length - 10} more
            </span>
          )}
        </div>
      ) : (
        <span style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
          Not extracted
        </span>
      )}
    </div>
  );
}

function downloadFile(candidate: RankedCandidate) {
  const byteCharacters = atob(candidate.file_bytes_b64);
  const byteNumbers = new Uint8Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const blob = new Blob([byteNumbers], {
    type: candidate.file_type === 'pdf' ? 'application/pdf' : 'text/plain',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = candidate.id;
  a.click();
  URL.revokeObjectURL(url);
}

export default function CandidateCard({
  candidate,
  rank,
  index,
  scoringMode,
  defaultExpanded = false,
}: CandidateCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded || rank <= 3);

  const cardVariants = {
    hidden: { opacity: 0, y: 28, scale: 0.98 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.4,
        delay: index * 0.08,
      },
    },
  };

  return (
    <motion.article
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className="glass-card"
      style={{
        overflow: 'hidden',
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
        ...(rank === 1 && {
          borderColor: 'rgba(251,191,36,0.22)',
          boxShadow: '0 4px 24px rgba(0,0,0,0.4), 0 0 32px rgba(251,191,36,0.08)',
        }),
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 16,
          padding: '20px 24px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={() => setExpanded((p) => !p)}
        role="button"
        aria-expanded={expanded}
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && setExpanded((p) => !p)}
      >
        {/* Rank badge */}
        <span className={rankBadgeClass(rank)} aria-label={`Rank ${rank}`}>
          {rankEmoji(rank)}
        </span>

        {/* Candidate info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3
            style={{
              fontFamily: 'var(--font-heading)',
              fontWeight: 700,
              fontSize: '1.02rem',
              color: 'var(--color-text-primary)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {candidate.id}
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 5, flexWrap: 'wrap' }}>
            <Mail size={12} color="var(--color-text-muted)" />
            <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
              {candidate.email}
            </span>
            <span className="chip chip--neutral" style={{ fontSize: '0.72rem', marginLeft: 4 }}>
              SBERT {(candidate.similarity * 100).toFixed(0)}%
            </span>
            <span
              className={`chip ${scoringMode === 'llm+sbert' ? 'chip--violet' : 'chip--teal'}`}
              style={{ fontSize: '0.72rem' }}
            >
              <Zap size={10} />
              {scoringMode === 'llm+sbert' ? 'LLM + SBERT' : 'SBERT'}
            </span>
          </div>

          {/* Justification */}
          <p
            style={{
              marginTop: 10,
              fontSize: '0.85rem',
              color: 'var(--color-text-secondary)',
              lineHeight: 1.55,
            }}
          >
            {candidate.justification}
          </p>
        </div>

        {/* Score gauge + expand toggle */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <ScoreGauge score={candidate.rating_10} size={88} />
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.22 }}
            style={{ color: 'var(--color-text-muted)' }}
          >
            <ChevronDown size={16} />
          </motion.div>
        </div>
      </div>

      {/* Expandable detail panel */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            key="details"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <div
              style={{
                borderTop: '1px solid var(--color-border)',
                padding: '20px 24px',
                display: 'flex',
                flexDirection: 'column',
                gap: 20,
              }}
            >
              {/* Structured sections */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: 20,
                }}
              >
                <StructuredSection
                  icon={<Cpu size={12} />}
                  title="Skills"
                  text={candidate.skills}
                />
                <StructuredSection
                  icon={<Briefcase size={12} />}
                  title="Experience"
                  text={candidate.experience}
                />
                <StructuredSection
                  icon={<GraduationCap size={12} />}
                  title="Education"
                  text={candidate.education}
                />
              </div>

              {/* Download */}
              {candidate.file_bytes_b64 && (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      downloadFile(candidate);
                    }}
                    aria-label={`Download ${candidate.id}`}
                  >
                    <Download size={14} />
                    Download Resume
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}
