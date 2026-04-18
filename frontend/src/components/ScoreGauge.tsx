import { motion, useSpring, useTransform } from 'framer-motion';
import { useEffect } from 'react';

interface ScoreGaugeProps {
  score: number;   // 1 – 10
  size?: number;
  animate?: boolean;
}

function scoreColor(score: number): string {
  if (score >= 8) return '#10b981';   // emerald
  if (score >= 5.5) return '#f59e0b'; // amber
  return '#ef4444';                    // red
}

function scoreLabel(score: number): string {
  if (score >= 8.5) return 'Excellent';
  if (score >= 7) return 'Strong';
  if (score >= 5.5) return 'Good';
  if (score >= 4) return 'Fair';
  return 'Weak';
}

export default function ScoreGauge({ score, size = 96, animate = true }: ScoreGaugeProps) {
  const clampedScore = Math.max(0, Math.min(10, score));
  const radius = (size / 2) - 8;
  const circumference = 2 * Math.PI * radius;
  const targetOffset = circumference - (clampedScore / 10) * circumference;
  const color = scoreColor(clampedScore);

  const springOffset = useSpring(circumference, {
    stiffness: 60,
    damping: 18,
    restDelta: 0.5,
  });

  useEffect(() => {
    if (animate) {
      // Small delay so it triggers visually after card entrance
      const t = setTimeout(() => {
        springOffset.set(targetOffset);
      }, 200);
      return () => clearTimeout(t);
    } else {
      springOffset.set(targetOffset);
    }
  }, [targetOffset, animate, springOffset]);

  // Animated opacity glow ring
  const glowOpacity = useTransform(
    springOffset,
    [circumference, targetOffset],
    [0, 0.6]
  );

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        flexShrink: 0,
      }}
      title={`Fit Score: ${clampedScore.toFixed(1)} / 10 — ${scoreLabel(clampedScore)}`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ transform: 'rotate(-90deg)' }}
      >
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={7}
        />

        {/* Glow duplicate */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={9}
          strokeLinecap="round"
          strokeDasharray={circumference}
          style={{
            strokeDashoffset: springOffset,
            opacity: glowOpacity,
            filter: `blur(4px)`,
          }}
        />

        {/* Score arc */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={7}
          strokeLinecap="round"
          strokeDasharray={circumference}
          style={{ strokeDashoffset: springOffset }}
        />
      </svg>

      {/* Center text */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-heading)',
            fontSize: size < 80 ? '1.1rem' : '1.4rem',
            fontWeight: 800,
            color,
            lineHeight: 1,
          }}
        >
          {clampedScore.toFixed(1)}
        </span>
        <span
          style={{
            fontSize: '0.55rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: 'var(--color-text-muted)',
          }}
        >
          / 10
        </span>
        {size >= 90 && (
          <span
            style={{
              fontSize: '0.6rem',
              fontWeight: 600,
              color,
              opacity: 0.85,
              marginTop: 2,
            }}
          >
            {scoreLabel(clampedScore)}
          </span>
        )}
      </div>
    </div>
  );
}
