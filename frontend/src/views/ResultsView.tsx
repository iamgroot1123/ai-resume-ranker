import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Download,
  SortAsc,
  SortDesc,
  Trophy,
  Users,
  Zap,
  BarChart2,
} from 'lucide-react';
import CandidateCard from '../components/CandidateCard';
import type { RankResponse } from '../types';

interface ResultsViewProps {
  response: RankResponse;
  onBack: () => void;
}

type SortKey = 'rating_10' | 'similarity' | 'id';

function exportCsv(response: RankResponse) {
  const headers = [
    'Rank', 'File', 'Email', 'Fit Score', 'SBERT Similarity',
    'Key Matches', 'Justification', 'Skills', 'Experience', 'Education',
  ];
  const rows = response.candidates.map((c, i) => [
    i + 1,
    c.id,
    c.email,
    c.rating_10.toFixed(1),
    (c.similarity * 100).toFixed(1) + '%',
    c.key_matches,
    `"${c.justification.replace(/"/g, '""')}"`,
    `"${c.skills.replace(/"/g, '""')}"`,
    `"${c.experience.replace(/"/g, '""')}"`,
    `"${c.education.replace(/"/g, '""')}"`,
  ]);
  const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'candidate_rankings.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function StatPill({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div
      className="glass-card"
      style={{
        padding: '14px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flex: '1 1 160px',
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 'var(--radius-sm)',
          background: 'rgba(124,97,255,0.12)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div>
        <p style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </p>
        <p style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 700 }}>
          {value}
        </p>
      </div>
    </div>
  );
}

export default function ResultsView({ response, onBack }: ResultsViewProps) {
  const [sortKey, setSortKey] = useState<SortKey>('rating_10');
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = useMemo(() => {
    return [...response.candidates].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number);
      return sortAsc ? cmp : -cmp;
    });
  }, [response.candidates, sortKey, sortAsc]);

  const topScore = Math.max(...response.candidates.map((c) => c.rating_10));
  const avgScore =
    response.candidates.reduce((s, c) => s + c.rating_10, 0) / response.candidates.length;

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc((p) => !p);
    else { setSortKey(key); setSortAsc(false); }
  }

  const SortIcon = sortAsc ? SortAsc : SortDesc;

  return (
    <div className="container" style={{ paddingTop: 36, paddingBottom: 64 }}>
      {/* Toolbar */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          marginBottom: 28,
          flexWrap: 'wrap',
        }}
      >
        <button
          className="btn btn--ghost btn--sm"
          onClick={onBack}
          aria-label="Back to upload"
        >
          <ArrowLeft size={15} />
          New Analysis
        </button>

        <div style={{ flex: 1 }} />

        {/* Sort controls */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginRight: 4 }}>Sort:</span>
          {(['rating_10', 'similarity', 'id'] as SortKey[]).map((key) => (
            <button
              key={key}
              className={`btn btn--sm ${sortKey === key ? 'btn--secondary' : 'btn--ghost'}`}
              onClick={() => toggleSort(key)}
              style={{ gap: 4 }}
            >
              {sortKey === key && <SortIcon size={12} />}
              {key === 'rating_10' ? 'Fit Score' : key === 'similarity' ? 'SBERT' : 'Name'}
            </button>
          ))}
        </div>

        <button
          className="btn btn--secondary btn--sm"
          onClick={() => exportCsv(response)}
          aria-label="Export results as CSV"
        >
          <Download size={14} />
          Export CSV
        </button>
      </motion.div>

      {/* Stats row */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.05 }}
        style={{ display: 'flex', gap: 14, marginBottom: 32, flexWrap: 'wrap' }}
      >
        <StatPill
          icon={<Users size={16} color="var(--color-violet-light)" />}
          label="Analyzed"
          value={`${response.total_uploaded} resumes`}
        />
        <StatPill
          icon={<Trophy size={16} color="var(--color-gold)" />}
          label="Top Score"
          value={`${topScore.toFixed(1)} / 10`}
        />
        <StatPill
          icon={<BarChart2 size={16} color="var(--color-teal-light)" />}
          label="Avg Score"
          value={`${avgScore.toFixed(1)} / 10`}
        />
        <StatPill
          icon={<Zap size={16} color={response.scoring_mode === 'llm+sbert' ? 'var(--color-violet-light)' : 'var(--color-teal-light)'} />}
          label="Mode"
          value={response.scoring_mode === 'llm+sbert' ? 'LLM + SBERT' : 'SBERT Semantic'}
        />
      </motion.div>

      {/* Section label */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.12 }}
        style={{ marginBottom: 20 }}
      >
        <p className="section-label">
          Top {response.total_returned} Candidates
        </p>
      </motion.div>

      {/* Candidate cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {sorted.map((candidate, i) => {
          const originalRank = response.candidates.indexOf(candidate) + 1;
          return (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              rank={originalRank}
              index={i}
              scoringMode={response.scoring_mode}
            />
          );
        })}
      </div>

      {/* Bottom export */}
      {response.candidates.length > 3 && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          style={{ marginTop: 32, display: 'flex', justifyContent: 'center' }}
        >
          <button
            className="btn btn--primary"
            onClick={() => exportCsv(response)}
          >
            <Download size={16} />
            Download Full Report (CSV)
          </button>
        </motion.div>
      )}
    </div>
  );
}
