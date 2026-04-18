import { motion } from 'framer-motion';
import { Brain } from 'lucide-react';

interface LoadingScreenProps {
  useLlm?: boolean;
  fileCount?: number;
  modelName?: string;
}

const steps = [
  { label: 'Extracting text from documents…', delay: 0 },
  { label: 'Computing semantic embeddings…', delay: 0.6 },
  { label: 'Calculating cosine similarity…', delay: 1.2 },
  { label: 'Generating candidate scores…', delay: 1.8 },
];

export default function LoadingScreen({ 
  useLlm = false, 
  fileCount = 0,
  modelName = '' 
}: LoadingScreenProps) {
  const getModelLabel = (name: string) => {
    if (!name) return 'LLM';
    if (name.includes('gpt-3.5')) return 'GPT-3.5';
    if (name.includes('gpt-4o')) return 'GPT-4o';
    if (name.includes('gpt-4')) return 'GPT-4';
    if (name.includes('llama')) return 'Llama 3.3';
    return name;
  };

  const modelLabel = getModelLabel(modelName);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        background: 'rgba(6, 11, 26, 0.88)',
        backdropFilter: 'blur(24px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 32,
        padding: 24,
      }}
    >
      {/* Pulsing brain icon */}
      <motion.div
        animate={{
          scale: [1, 1.08, 1],
          boxShadow: [
            '0 0 20px rgba(124,97,255,0.3)',
            '0 0 50px rgba(124,97,255,0.6)',
            '0 0 20px rgba(124,97,255,0.3)',
          ],
        }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          width: 80,
          height: 80,
          borderRadius: 'var(--radius-lg)',
          background: 'linear-gradient(135deg, #7c61ff, #00d4aa)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Brain size={38} color="#fff" />
      </motion.div>

      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <h2
          style={{
            fontFamily: 'var(--font-heading)',
            fontSize: '1.5rem',
            fontWeight: 700,
            marginBottom: 8,
          }}
        >
          Analyzing {fileCount > 0 ? `${fileCount} ` : ''}Candidates
        </h2>
        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.88rem' }}>
          {useLlm ? 'Running LLM + SBERT hybrid pipeline…' : 'Running semantic similarity pipeline…'}
        </p>
      </div>

      {/* Step list */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          width: '100%',
          maxWidth: 380,
        }}
      >
        {steps.map((step, i) => (
          <motion.div
            key={step.label}
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: step.delay, duration: 0.4 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '10px 14px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid var(--color-border)',
            }}
          >
            {/* Spinner dot */}
            <motion.div
              animate={{ scale: [0.8, 1.2, 0.8], opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.3 }}
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: 'var(--color-violet)',
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)' }}>
              {step.label}
            </span>
          </motion.div>
        ))}

        {useLlm && (
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 2.4, duration: 0.4 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '10px 14px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(124,97,255,0.08)',
              border: '1px solid rgba(124,97,255,0.2)',
            }}
          >
            <motion.div
              animate={{ scale: [0.8, 1.2, 0.8], opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.4, repeat: Infinity, delay: 1.2 }}
              style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--color-violet)', flexShrink: 0 }}
            />
            <span style={{ fontSize: '0.82rem', color: 'var(--color-violet-light)' }}>
              Calling {modelLabel} for qualitative scoring…
            </span>
          </motion.div>
        )}
      </div>

      {/* Progress bar */}
      <div
        style={{
          width: '100%',
          maxWidth: 380,
          height: 3,
          background: 'rgba(255,255,255,0.08)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <motion.div
          initial={{ width: '0%' }}
          animate={{ width: '94%' }}
          transition={{ duration: useLlm ? 12 : 5, ease: 'easeOut' }}
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, var(--color-violet), var(--color-teal))',
            borderRadius: 2,
          }}
        />
      </div>
    </motion.div>
  );
}
