/**
 * Load Skill dialog — Center Stage M.
 *
 * Supports three loading modes:
 *   1. Local file path (SKILL.md or simpleskill.json)
 *   2. URL (http/https pointing to a SKILL.md — auto-converts GitHub blob URLs)
 *   3. Directory path (recursive scan for SKILL.md files)
 *
 * Sends `load_skill` command via WebSocket and waits for the server response
 * (skill_loaded or skill_error) before closing.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { CenterStage } from '../../CenterStage';
import { Upload, FileText, Globe, FolderOpen, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

export interface LoadSkillDialogProps {
  isOpen: boolean;
  onLoad: (source: string) => void;
  onClose: () => void;
  /** Last load result — set by parent when skill_loaded or skill_error arrives */
  loadResult?: { success: boolean; message: string } | null;
}

type LoadMode = 'file' | 'url' | 'directory';

const MODES: { key: LoadMode; label: string; icon: React.ReactNode; placeholder: string; description: string }[] = [
  {
    key: 'file',
    label: 'Local File',
    icon: <FileText size={14} />,
    placeholder: '/path/to/SKILL.md',
    description: 'Path to a SKILL.md or simpleskill.json file',
  },
  {
    key: 'url',
    label: 'URL',
    icon: <Globe size={14} />,
    placeholder: 'https://github.com/user/repo/blob/main/SKILL.md',
    description: 'URL to a SKILL.md file (GitHub blob URLs auto-converted)',
  },
  {
    key: 'directory',
    label: 'Directory',
    icon: <FolderOpen size={14} />,
    placeholder: '/path/to/skills/',
    description: 'Directory to recursively scan for SKILL.md files',
  },
];

export const LoadSkillDialog: React.FC<LoadSkillDialogProps> = ({
  isOpen,
  onLoad,
  onClose,
  loadResult,
}) => {
  const [mode, setMode] = useState<LoadMode>('file');
  const [source, setSource] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const currentMode = MODES.find((m) => m.key === mode)!;

  const handleLoad = useCallback(() => {
    const trimmed = source.trim();
    if (!trimmed) {
      setError('Please enter a source path or URL');
      return;
    }
    if (mode === 'url' && !trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      setError('URL must start with http:// or https://');
      return;
    }
    setError(null);
    setIsLoading(true);
    onLoad(trimmed);
  }, [source, mode, onLoad]);

  // Track isLoading via ref to avoid effect cleanup issues
  const isLoadingRef = useRef(false);
  isLoadingRef.current = isLoading;
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // React to server response
  useEffect(() => {
    if (!loadResult || !isLoadingRef.current) return;

    setIsLoading(false);
    if (loadResult.success) {
      setSuccessMessage(loadResult.message);
      closeTimerRef.current = setTimeout(() => onClose(), 1500);
    } else {
      setError(loadResult.message);
    }
    return () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
    };
  }, [loadResult, onClose]);

  // Reset on open
  useEffect(() => {
    if (isOpen) {
      setSource('');
      setError(null);
      setSuccessMessage(null);
      setIsLoading(false);
    }
  }, [isOpen]);

  return (
    <CenterStage
      isOpen={isOpen}
      onClose={onClose}
      size="M"
      title="Load Skill"
    >
      <div className="space-y-5" data-testid="load-skill-dialog">
        {/* Mode selector */}
        <div>
          <label className="block text-[11px] font-medium text-gray-400 mb-2 uppercase tracking-wide">
            Source Type
          </label>
          <div className="grid grid-cols-3 gap-2">
            {MODES.map((m) => (
              <button
                key={m.key}
                type="button"
                disabled={isLoading}
                onClick={() => {
                  setMode(m.key);
                  setSource('');
                  setError(null);
                }}
                className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border transition-colors ${
                  mode === m.key
                    ? 'border-blue-500/50 bg-blue-900/20 text-blue-400'
                    : 'border-gray-700 bg-gray-900/30 text-gray-400 hover:border-gray-600'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                data-testid={`load-mode-${m.key}`}
              >
                {m.icon}
                <span className="text-[11px] font-medium">{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Source input */}
        <div>
          <label className="block text-[11px] font-medium text-gray-400 mb-1 uppercase tracking-wide">
            {currentMode.label} Path
          </label>
          <p className="text-[10px] text-gray-600 mb-2">{currentMode.description}</p>
          <input
            type="text"
            value={source}
            onChange={(e) => { setSource(e.target.value); setError(null); }}
            placeholder={currentMode.placeholder}
            disabled={isLoading}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-gray-200 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none font-mono disabled:opacity-50"
            data-testid="load-skill-source-input"
            onKeyDown={(e) => { if (e.key === 'Enter' && !isLoading) handleLoad(); }}
            autoFocus
          />
          {error && (
            <div className="mt-1.5 flex items-start gap-1.5 text-[11px] text-red-400">
              <AlertCircle size={12} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
          {successMessage && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-green-400" data-testid="load-skill-success">
              <CheckCircle size={12} />
              <span>{successMessage}</span>
            </div>
          )}
        </div>

        {/* Info box */}
        <div className="text-[11px] text-gray-500 bg-gray-900/40 rounded-lg p-3 border border-gray-700/50">
          <strong className="text-gray-400">Supported formats:</strong>
          <ul className="mt-1 space-y-0.5 list-disc list-inside">
            <li>SKILL.md — YAML frontmatter + Markdown body (Claude / OpenClaw compatible)</li>
            <li>simpleskill.json — JSON manifest with full schema</li>
            <li>GitHub blob URLs — auto-converted to raw content links</li>
            <li>Directory — recursively scans for all SKILL.md files</li>
          </ul>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-2 border-t border-gray-700">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm text-gray-300 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleLoad}
            disabled={!source.trim() || isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
            data-testid="load-skill-submit"
          >
            {isLoading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Upload size={14} />
            )}
            {isLoading ? 'Loading...' : 'Load Skill'}
          </button>
        </div>
      </div>
    </CenterStage>
  );
};
