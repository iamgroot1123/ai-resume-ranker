import { useState, useCallback, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import UploadView from './views/UploadView';
import ResultsView from './views/ResultsView';
import LoadingScreen from './components/LoadingScreen';
import { rankResumes, checkHealth } from './api/ranker';
import type { AppView, RankRequest, RankResponse } from './types';

// ---- Navbar ----------------------------------------------------------------
function Navbar({ modelLoaded }: { modelLoaded: boolean | null }) {
  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar__inner">
        <a href="/" className="navbar__logo" aria-label="ResumeIQ home">
          <div className="navbar__logo-icon" aria-hidden>🔮</div>
          <span className="navbar__logo-text">ResumeIQ</span>
        </a>
        <div className="navbar__badge">
          <span
            className={`status-dot ${modelLoaded === false ? 'status-dot--error' : ''}`}
            aria-hidden
          />
          {modelLoaded === null
            ? 'Connecting…'
            : modelLoaded
            ? 'Model ready'
            : 'Model not loaded'}
        </div>
      </div>
    </nav>
  );
}

// ---- Page transition variants ----------------------------------------------
const pageVariants = {
  initial: { opacity: 0, y: 18 },
  enter:   { opacity: 1, y: 0, transition: { duration: 0.4 } },
  exit:    { opacity: 0, y: -12, transition: { duration: 0.25 } },
};

// ---- App -------------------------------------------------------------------
export default function App() {
  const [view, setView] = useState<AppView>('upload');
  const [isLoading, setIsLoading] = useState(false);
  const [rankingUseLlm, setRankingUseLlm] = useState(false);
  const [rankingModelName, setRankingModelName] = useState('');
  const [rankingFileCount, setRankingFileCount] = useState(0);
  const [response, setResponse] = useState<RankResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then((h) => setModelLoaded(h.model_loaded))
      .catch(() => setModelLoaded(false));
  }, []);

  const handleSubmit = useCallback(async (req: RankRequest) => {
    setError(null);
    setIsLoading(true);
    setRankingUseLlm(req.useLlm);
    setRankingModelName(req.llmModel);
    setRankingFileCount(req.resumeFiles.length);

    try {
      const result = await rankResumes(req);
      setResponse(result);
      setView('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
      setView('upload');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleBack = useCallback(() => {
    setView('upload');
    setError(null);
  }, []);

  return (
    <>
      {/* Animated background */}
      <div className="app-bg" aria-hidden>
        <div className="grid-overlay" />
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Loading overlay */}
      <AnimatePresence>
        {isLoading && (
          <LoadingScreen 
            useLlm={rankingUseLlm} 
            fileCount={rankingFileCount} 
            modelName={rankingModelName}
          />
        )}
      </AnimatePresence>

      {/* App shell */}
      <div className="app-shell">
        <Navbar modelLoaded={modelLoaded} />

        <main style={{ flex: 1 }}>
          <AnimatePresence mode="wait">
            {view === 'upload' ? (
              <motion.div
                key="upload"
                variants={pageVariants}
                initial="initial"
                animate="enter"
                exit="exit"
              >
                <UploadView
                  onSubmit={handleSubmit}
                  isLoading={isLoading}
                  error={error}
                />
              </motion.div>
            ) : (
              response && (
                <motion.div
                  key="results"
                  variants={pageVariants}
                  initial="initial"
                  animate="enter"
                  exit="exit"
                >
                  <ResultsView response={response} onBack={handleBack} />
                </motion.div>
              )
            )}
          </AnimatePresence>
        </main>

        {/* Footer */}
        <footer
          style={{
            borderTop: '1px solid var(--color-border)',
            padding: '16px 24px',
            textAlign: 'center',
            fontSize: '0.75rem',
            color: 'var(--color-text-muted)',
          }}
        >
          ResumeIQ — AI-Powered Candidate Ranking &nbsp;·&nbsp; SBERT + GPT-3.5 Pipeline
        </footer>
      </div>
    </>
  );
}
