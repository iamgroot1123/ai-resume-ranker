import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, User, ChevronDown } from 'lucide-react';
import type { AppMode } from '../types';

interface ModeSwitcherProps {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

const MODES = [
  {
    id: 'recruiter' as AppMode,
    icon: Users,
    label: 'Recruiter',
    desc: 'Rank multiple candidates against a JD',
  },
  {
    id: 'applicant' as AppMode,
    icon: User,
    label: 'Applicant',
    desc: 'Check how well your resume fits a JD',
  },
];

export default function ModeSwitcher({ mode, onModeChange }: ModeSwitcherProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = MODES.find((m) => m.id === mode)!;

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        id="mode-switcher-btn"
        onClick={() => setOpen((p) => !p)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'rgba(124,97,255,0.12)',
          border: '1px solid rgba(124,97,255,0.3)',
          color: 'var(--color-text-primary)',
          cursor: 'pointer',
          fontSize: '0.82rem',
          fontWeight: 600,
          transition: 'background 0.2s',
        }}
        onMouseEnter={(e) =>
          ((e.currentTarget as HTMLElement).style.background = 'rgba(124,97,255,0.22)')
        }
        onMouseLeave={(e) =>
          ((e.currentTarget as HTMLElement).style.background = 'rgba(124,97,255,0.12)')
        }
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <current.icon size={14} color="var(--color-violet-light)" />
        <span>{current.label} Mode</span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={13} color="var(--color-text-muted)" />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            key="dropdown"
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.18 }}
            role="listbox"
            style={{
              position: 'absolute',
              top: 'calc(100% + 8px)',
              left: 0,
              minWidth: 240,
              background: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)',
              boxShadow: '0 16px 48px rgba(0,0,0,0.4)',
              zIndex: 300,
              overflow: 'hidden',
            }}
          >
            {MODES.map((m) => {
              const Icon = m.icon;
              const isActive = m.id === mode;
              return (
                <button
                  key={m.id}
                  role="option"
                  aria-selected={isActive}
                  onClick={() => {
                    onModeChange(m.id);
                    setOpen(false);
                  }}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '12px 16px',
                    background: isActive ? 'rgba(124,97,255,0.12)' : 'none',
                    border: 'none',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'background 0.15s',
                    borderBottom: '1px solid var(--color-border)',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive)
                      (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive)
                      (e.currentTarget as HTMLElement).style.background = 'none';
                  }}
                >
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 'var(--radius-xs)',
                      background: isActive
                        ? 'rgba(124,97,255,0.2)'
                        : 'rgba(255,255,255,0.06)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <Icon
                      size={15}
                      color={isActive ? 'var(--color-violet-light)' : 'var(--color-text-muted)'}
                    />
                  </div>
                  <div>
                    <p
                      style={{
                        fontSize: '0.84rem',
                        fontWeight: 600,
                        color: isActive
                          ? 'var(--color-violet-light)'
                          : 'var(--color-text-primary)',
                      }}
                    >
                      {m.label} Mode
                    </p>
                    <p
                      style={{
                        fontSize: '0.72rem',
                        color: 'var(--color-text-muted)',
                        marginTop: 2,
                      }}
                    >
                      {m.desc}
                    </p>
                  </div>
                  {isActive && (
                    <div
                      style={{
                        marginLeft: 'auto',
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: 'var(--color-violet)',
                        flexShrink: 0,
                      }}
                    />
                  )}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
