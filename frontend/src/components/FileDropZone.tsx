import React, { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, X, FilePlus } from 'lucide-react';

interface FileDropZoneProps {
  files: File[];
  onChange: (files: File[]) => void;
  accept?: string;
  label?: string;
}

const ACCEPTED_EXTS = ['.pdf', '.txt'];

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileDropZone({
  files,
  onChange,
  accept = '.pdf,.txt',
  label = 'Upload Resumes',
}: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback(
    (incoming: FileList | null) => {
      if (!incoming) return;
      const valid = Array.from(incoming).filter((f) =>
        ACCEPTED_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext))
      );
      const merged = [
        ...files,
        ...valid.filter((v) => !files.some((e) => e.name === v.name)),
      ];
      onChange(merged);
    },
    [files, onChange]
  );

  const removeFile = (name: string) => {
    onChange(files.filter((f) => f.name !== name));
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Drop zone */}
      <motion.div
        className={`dropzone ${isDragging ? 'dropzone--active' : ''}`}
        animate={{ scale: isDragging ? 1.015 : 1 }}
        transition={{ duration: 0.18 }}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label={`${label} drop zone`}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? 'var(--color-violet)' : 'var(--color-border)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '36px 24px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          cursor: 'pointer',
          background: isDragging
            ? 'rgba(124,97,255,0.08)'
            : 'rgba(255,255,255,0.025)',
          transition: 'all 0.2s ease',
          boxShadow: isDragging ? 'var(--shadow-glow)' : 'none',
          minHeight: '180px',
          userSelect: 'none',
        }}
      >
        <motion.div
          animate={{ y: isDragging ? -6 : 0 }}
          transition={{ duration: 0.2 }}
          style={{
            width: 52,
            height: 52,
            borderRadius: 'var(--radius-md)',
            background: isDragging ? 'rgba(124,97,255,0.2)' : 'rgba(255,255,255,0.06)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'background 0.2s ease',
          }}
        >
          {isDragging ? (
            <FilePlus size={24} color="var(--color-violet-light)" />
          ) : (
            <Upload size={24} color="var(--color-text-muted)" />
          )}
        </motion.div>

        <div style={{ textAlign: 'center' }}>
          <p style={{ fontWeight: 600, color: isDragging ? 'var(--color-violet-light)' : 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
            {isDragging ? 'Drop to add files' : 'Drag & drop files here'}
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', marginTop: '4px' }}>
            or <span style={{ color: 'var(--color-violet-light)', fontWeight: 600 }}>click to browse</span> — PDF · TXT
          </p>
        </div>

        {files.length > 0 && (
          <div className="chip chip--violet" style={{ marginTop: '4px' }}>
            {files.length} file{files.length !== 1 ? 's' : ''} ready
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          className="sr-only"
          onChange={(e) => addFiles(e.target.files)}
          aria-label="File input"
        />
      </motion.div>

      {/* File list */}
      <AnimatePresence initial={false}>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            style={{ overflow: 'hidden' }}
          >
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '8px',
                maxHeight: '160px',
                overflowY: 'auto',
                padding: '4px 2px',
              }}
            >
              <AnimatePresence>
                {files.map((file) => (
                  <motion.span
                    key={file.name}
                    className="file-chip"
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.15 }}
                    title={`${file.name} (${humanSize(file.size)})`}
                  >
                    <FileText size={12} style={{ flexShrink: 0 }} />
                    <span className="file-chip__name">{file.name}</span>
                    <button
                      className="file-chip__remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.name);
                      }}
                      aria-label={`Remove ${file.name}`}
                    >
                      <X size={10} />
                    </button>
                  </motion.span>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
