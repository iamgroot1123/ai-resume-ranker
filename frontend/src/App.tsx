import { useState, useCallback, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import UploadView from './views/UploadView';
import ResultsView from './views/ResultsView';
import ApplicantView from './views/ApplicantView';
import ApplicantResultsView from './views/ApplicantResultsView';
import LoadingScreen from './components/LoadingScreen';
import ModeSwitcher from './components/ModeSwitcher';
import { rankResumes, checkHealth, analyzeApplicant } from './api/ranker';
import type { AppView, AppMode, RankRequest, RankResponse, ApplicantRequest, ApplicantResult } from './types';

// ---- Navbar ----------------------------------------------------------------
function Navbar({
  modelLoaded,
  mode,
  onModeChange,
}: {
  modelLoaded: boolean | null;
  mode: AppMode;
  onModeChange: (m: AppMode) => void;
}) {
  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar__inner">
        {/* Left: mode switcher */}
        <ModeSwitcher mode={mode} onModeChange={onModeChange} />

        {/* Center: logo */}
        <a href="/" className="navbar__logo" aria-label="ResumeIQ home" style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
          <div className="navbar__logo-icon" aria-hidden>🔮</div>
          <span className="navbar__logo-text">ResumeIQ</span>
        </a>

        {/* Right: model status */}
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
  const [appMode, setAppMode] = useState<AppMode>(() => {
    const path = typeof window !== 'undefined' ? window.location.pathname : '/recruiter';
    return path === '/applicant' ? 'applicant' : 'recruiter';
  });

  // Recruiter state
  const [view, setView] = useState<AppView>('upload');
  const [isLoading, setIsLoading] = useState(false);
  const [rankingUseLlm, setRankingUseLlm] = useState(false);
  const [rankingModelName, setRankingModelName] = useState('');
  const [rankingFileCount, setRankingFileCount] = useState(0);
  const [response, setResponse] = useState<RankResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Applicant state
  const [applicantResult, setApplicantResult] = useState<ApplicantResult | null>(null);
  const [applicantView, setApplicantView] = useState<'form' | 'results'>('form');
  const [applicantLoading, setApplicantLoading] = useState(false);
  const [applicantError, setApplicantError] = useState<string | null>(null);

  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then((h) => setModelLoaded(h.model_loaded))
      .catch(() => setModelLoaded(false));
  }, []);

  // Handle URL deep-linking on mount & browser back/forward buttons
  useEffect(() => {
    const path = window.location.pathname;
    if (path !== '/applicant' && path !== '/recruiter') {
      window.history.replaceState(null, '', '/recruiter');
    }

    const handlePopState = () => {
      const currentPath = window.location.pathname;
      const newMode: AppMode = currentPath === '/applicant' ? 'applicant' : 'recruiter';
      setAppMode(newMode);
      setView('upload');
      setApplicantView('form');
      setError(null);
      setApplicantError(null);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // When mode changes, reset to default view
  const handleModeChange = useCallback((newMode: AppMode) => {
    setAppMode(newMode);
    setView('upload');
    setApplicantView('form');
    setError(null);
    setApplicantError(null);

    const targetPath = newMode === 'applicant' ? '/applicant' : '/recruiter';
    if (window.location.pathname !== targetPath) {
      window.history.pushState(null, '', targetPath);
    }
  }, []);

  // ---- Recruiter handlers ----
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

  // ---- Applicant handlers ----
  const handleApplicantSubmit = useCallback(async (req: ApplicantRequest) => {
    setApplicantError(null);
    setApplicantLoading(true);

    try {
      const result = await analyzeApplicant(req);
      setApplicantResult(result);
      setApplicantView('results');
    } catch (err) {
      setApplicantError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setApplicantLoading(false);
    }
  }, []);

  const handleApplicantBack = useCallback(() => {
    setApplicantView('form');
    setApplicantError(null);
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

      {/* Loading overlays */}
      <AnimatePresence>
        {isLoading && (
          <LoadingScreen
            useLlm={rankingUseLlm}
            fileCount={rankingFileCount}
            modelName={rankingModelName}
          />
        )}
        {applicantLoading && (
          <LoadingScreen useLlm={true} fileCount={1} modelName="Analyzing your resume…" />
        )}
      </AnimatePresence>

      {/* App shell */}
      <div className="app-shell">
        <Navbar modelLoaded={modelLoaded} mode={appMode} onModeChange={handleModeChange} />

        <main style={{ flex: 1 }}>
          <AnimatePresence mode="wait">

            {/* ---- RECRUITER MODE ---- */}
            {appMode === 'recruiter' && view === 'upload' && (
              <motion.div key="recruiter-upload" variants={pageVariants} initial="initial" animate="enter" exit="exit">
                <UploadView onSubmit={handleSubmit} isLoading={isLoading} error={error} />
              </motion.div>
            )}
            {appMode === 'recruiter' && view === 'results' && response && (
              <motion.div key="recruiter-results" variants={pageVariants} initial="initial" animate="enter" exit="exit">
                <ResultsView response={response} onBack={handleBack} />
              </motion.div>
            )}

            {/* ---- APPLICANT MODE ---- */}
            {appMode === 'applicant' && applicantView === 'form' && (
              <motion.div key="applicant-form" variants={pageVariants} initial="initial" animate="enter" exit="exit">
                <ApplicantView
                  onSubmit={handleApplicantSubmit}
                  isLoading={applicantLoading}
                  error={applicantError}
                />
              </motion.div>
            )}
            {appMode === 'applicant' && applicantView === 'results' && applicantResult && (
              <motion.div key="applicant-results" variants={pageVariants} initial="initial" animate="enter" exit="exit">
                <ApplicantResultsView result={applicantResult} onBack={handleApplicantBack} />
              </motion.div>
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
          ResumeIQ — AI-Powered Resume Analysis &nbsp;·&nbsp; SBERT + LLM Pipeline
        </footer>
      </div>
    </>
  );
}
