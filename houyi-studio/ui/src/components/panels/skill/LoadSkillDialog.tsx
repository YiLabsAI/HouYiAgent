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
  onLoad: (source: string, installStrategy?: 'copy' | 'symlink') => void;
  onClose: () => void;
  /** Last load result — set by parent when skill_loaded or skill_error arrives */
  loadResult?: { success: boolean; message: string } | null;
}

type LoadMode = 'file' | 'url' | 'directory';

interface InstallLifecyclePlan {
  installCommands: string[];
  verifyCommand: string;
  updateCommand: string;
  uninstallCommand: string;
}

interface GitHubInstallContext {
  owner: string;
  repo: string;
  linkAlias: string;
  linkTargetRelativePath: string;
}

const GITHUB_BLOB_OR_TREE_RE = /^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/(blob|tree)\/[^/]+\/(.+)$/i;
const GITHUB_REPO_RE = /^https?:\/\/github\.com\/([^/]+)\/([^/]+?)(?:\.git)?(?:\/)?$/i;
const GITHUB_RAW_RE = /^https?:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/[^/]+\/(.+)$/i;

function resolveFileSystemPath(file: File): string {
  const directPath = (file as File & { path?: unknown }).path;
  if (typeof directPath === 'string' && directPath.trim()) {
    return directPath;
  }
  if (file.webkitRelativePath?.trim()) {
    return file.webkitRelativePath;
  }
  return file.name;
}

function resolveDirectoryPath(files: FileList | File[]): string | null {
  const list = Array.from(files);
  if (list.length === 0) {
    return null;
  }
  const first = list[0];
  const directPath = (first as File & { path?: unknown }).path;
  if (typeof directPath === 'string' && directPath.trim()) {
    const relative = first.webkitRelativePath;
    if (!relative) {
      return directPath;
    }
    const slashIndex = relative.indexOf('/');
    const rootName = slashIndex > 0 ? relative.slice(0, slashIndex) : relative;
    if (!rootName) {
      return directPath;
    }
    const marker = `/${rootName}/`;
    if (directPath.includes(marker)) {
      return directPath.split(marker)[0] + `/${rootName}`;
    }
    return directPath;
  }
  if (first.webkitRelativePath) {
    // Browser-only fallback may only expose relative folder names (e.g. "crawl").
    // Returning it here causes false "source not found" for local directory loading.
    return null;
  }
  return null;
}

function isLikelyAbsolutePath(path: string): boolean {
  if (!path.trim()) {
    return false;
  }
  return path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path) || path.startsWith('\\\\');
}

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

function buildInstallLifecyclePlan(mode: LoadMode, source: string): InstallLifecyclePlan | null {
  if (mode !== 'url') {
    return null;
  }
  const trimmed = source.trim();
  if (!trimmed) {
    return null;
  }

  const installContext = resolveGitHubInstallContext(trimmed);
  if (!installContext) {
    return null;
  }

  const { owner, repo, linkAlias, linkTargetRelativePath } = installContext;
  const cloneUrl = `https://github.com/${owner}/${repo}.git`;
  const sourceRoot = `~/.houyi/sources/github.com/${owner}/${repo}`;
  const managedLink = `~/.houyi/skills/${linkAlias}`;
  const linkTarget = `${sourceRoot}/${linkTargetRelativePath}`;

  return {
    installCommands: [
      `git clone ${cloneUrl} ${sourceRoot}`,
      'mkdir -p ~/.houyi/skills',
      `ln -s ${linkTarget} ${managedLink}`,
    ],
    verifyCommand: `ls -la ${managedLink}`,
    updateCommand: `git -C ${sourceRoot} pull`,
    uninstallCommand: `rm ${managedLink}`,
  };
}

function resolveGitHubInstallContext(source: string): GitHubInstallContext | null {
  const deriveBinding = (repoPath: string | null, repoName: string): Pick<GitHubInstallContext, 'linkAlias' | 'linkTargetRelativePath'> => {
    if (!repoPath) {
      return { linkAlias: repoName, linkTargetRelativePath: 'skills' };
    }
    const parts = repoPath.split('/').filter(Boolean);
    const tail = parts[parts.length - 1]?.toLowerCase() ?? '';
    if ((tail === 'skill.md' || tail === 'simpleskill.json') && parts.length >= 2) {
      const linkAlias = parts[parts.length - 2];
      const linkTargetRelativePath = parts.slice(0, -1).join('/');
      return { linkAlias, linkTargetRelativePath };
    }
    return { linkAlias: repoName, linkTargetRelativePath: 'skills' };
  };

  const blobOrTree = source.match(GITHUB_BLOB_OR_TREE_RE);
  if (blobOrTree) {
    const owner = blobOrTree[1];
    const repo = blobOrTree[2].replace(/\.git$/i, '');
    const binding = deriveBinding(blobOrTree[4], repo);
    return { owner, repo, ...binding };
  }

  const raw = source.match(GITHUB_RAW_RE);
  if (raw) {
    const owner = raw[1];
    const repo = raw[2].replace(/\.git$/i, '');
    const binding = deriveBinding(raw[3], repo);
    return { owner, repo, ...binding };
  }

  const repo = source.match(GITHUB_REPO_RE);
  if (repo) {
    return {
      owner: repo[1],
      repo: repo[2].replace(/\.git$/i, ''),
      linkAlias: repo[2].replace(/\.git$/i, ''),
      linkTargetRelativePath: 'skills',
    };
  }

  return null;
}

export const LoadSkillDialog: React.FC<LoadSkillDialogProps> = ({
  isOpen,
  onLoad,
  onClose,
  loadResult,
}) => {
  const [mode, setMode] = useState<LoadMode>('file');
  const [source, setSource] = useState('');
  const [installStrategy, setInstallStrategy] = useState<'copy' | 'symlink'>('symlink');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDraggingPath, setIsDraggingPath] = useState(false);
  const installLifecyclePlan = buildInstallLifecyclePlan(mode, source);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const installStrategyHint = installStrategy === 'copy'
    ? 'copy snapshots the directory into managed local sources before linking into skills.'
    : 'symlink keeps managed local sources pointing to your original directory and links that into skills.';

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
    if (mode === 'directory' && !isLikelyAbsolutePath(trimmed)) {
      setError('Directory path must be absolute (e.g. /Users/name/skills). Please paste or drag an absolute path.');
      return;
    }
    setError(null);
    setIsLoading(true);
    if (mode === 'directory') {
      onLoad(trimmed, installStrategy);
      return;
    }
    if (mode === 'file') {
      onLoad(trimmed, 'copy');
      return;
    }
    onLoad(trimmed);
  }, [source, mode, installStrategy, onLoad]);

  // Track isLoading via ref to avoid effect cleanup issues
  const isLoadingRef = useRef(false);
  isLoadingRef.current = isLoading;
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleBrowseFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFilePicked = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (!picked) {
      return;
    }
    setSource(resolveFileSystemPath(picked));
    setError(null);
    e.target.value = '';
  }, []);

  const handleDragOverPath = useCallback((e: React.DragEvent) => {
    if (mode === 'url') {
      return;
    }
    e.preventDefault();
    setIsDraggingPath(true);
  }, [mode]);

  const handleDragLeavePath = useCallback((e: React.DragEvent) => {
    if (mode === 'url') {
      return;
    }
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return;
    }
    setIsDraggingPath(false);
  }, [mode]);

  const handleDropPath = useCallback((e: React.DragEvent) => {
    if (mode === 'url') {
      return;
    }
    e.preventDefault();
    setIsDraggingPath(false);
    const files = e.dataTransfer.files;
    const text = e.dataTransfer.getData('text/plain').trim();

    if (mode === 'directory') {
      if (text && isLikelyAbsolutePath(text)) {
        setSource(text);
        setError(null);
        return;
      }

      const dirPath = resolveDirectoryPath(files);
      if (dirPath) {
        setSource(dirPath);
        setError(null);
        return;
      }

      if (files && files.length > 0) {
        setError('Dropped folder did not provide an absolute path. Please drop or paste a full directory path.');
        return;
      }

      if (text && !isLikelyAbsolutePath(text)) {
        setError('Directory path must be absolute (e.g. /Users/name/skills). Please paste or drag an absolute path.');
      }
      return;
    }

    if (files && files.length > 0) {
      setSource(resolveFileSystemPath(files[0]));
      setError(null);
      return;
    }

    if (text) {
      setSource(text);
      setError(null);
    }
  }, [mode]);

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
      setInstallStrategy('symlink');
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
                  setSuccessMessage(null);
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

        {mode === 'directory' && (
          <div>
            <label className="block text-[11px] font-medium text-gray-400 mb-2 uppercase tracking-wide">
              Install Strategy
            </label>
            <div className="grid grid-cols-2 gap-2" data-testid="directory-install-strategy">
              <button
                type="button"
                disabled={isLoading}
                onClick={() => setInstallStrategy('copy')}
                className={`px-3 py-2 rounded-lg border text-[11px] transition-colors ${
                  installStrategy === 'copy'
                    ? 'border-blue-500/50 bg-blue-900/20 text-blue-300'
                    : 'border-gray-700 bg-gray-900/30 text-gray-400 hover:border-gray-600'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                data-testid="install-strategy-copy"
              >
                copy
              </button>
              <button
                type="button"
                disabled={isLoading}
                onClick={() => setInstallStrategy('symlink')}
                className={`px-3 py-2 rounded-lg border text-[11px] transition-colors ${
                  installStrategy === 'symlink'
                    ? 'border-blue-500/50 bg-blue-900/20 text-blue-300'
                    : 'border-gray-700 bg-gray-900/30 text-gray-400 hover:border-gray-600'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                data-testid="install-strategy-symlink"
              >
                symlink
              </button>
            </div>
            <p className="mt-1 text-[10px] text-gray-500" data-testid="install-strategy-hint">
              {installStrategyHint}
            </p>
          </div>
        )}

        {/* Source input */}
        <div>
          <label className="block text-[11px] font-medium text-gray-400 mb-1 uppercase tracking-wide">
            {currentMode.label} Path
          </label>
          <p className="text-[10px] text-gray-600 mb-2">{currentMode.description}</p>
          <div
            onDragOver={handleDragOverPath}
            onDragLeave={handleDragLeavePath}
            onDrop={handleDropPath}
            className={`rounded-lg border transition-colors ${
              isDraggingPath ? 'border-blue-500 bg-blue-500/10' : 'border-transparent'
            }`}
            data-testid="load-skill-dropzone"
          >
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
          </div>
          {mode === 'file' && (
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                onClick={handleBrowseFile}
                disabled={isLoading}
                className="px-3 py-1.5 rounded border border-gray-700 bg-gray-900/40 text-[11px] text-gray-300 hover:border-gray-600 disabled:opacity-50"
                data-testid="load-skill-browse-file"
              >
                Browse File
              </button>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,.json"
            onChange={handleFilePicked}
            className="hidden"
            data-testid="load-skill-file-picker"
          />
          {mode !== 'url' && (
            <div className="mt-1 text-[10px] text-gray-600">Tip: drag and drop local file/folder path into the input.</div>
          )}
          {mode === 'directory' && (
            <div className="mt-1 text-[10px] text-amber-400/90" data-testid="directory-path-web-note">
              Web mode cannot read absolute directory paths from folder picker; paste an absolute path directly.
            </div>
          )}
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

        {installLifecyclePlan && (
          <div
            className="text-[11px] text-gray-300 bg-gray-900/40 rounded-lg p-3 border border-gray-700/50 space-y-2"
            data-testid="install-lifecycle-plan"
          >
            <strong className="text-gray-200">Codex-style Install Lifecycle</strong>
            <div className="space-y-1">
              {installLifecyclePlan.installCommands.map((command, index) => (
                <code key={`${command}-${index}`} className="block font-mono text-[10px] text-blue-300">
                  {command}
                </code>
              ))}
            </div>
            <div className="space-y-1 border-t border-gray-700/60 pt-2">
              <code className="block font-mono text-[10px] text-emerald-300"># verify: {installLifecyclePlan.verifyCommand}</code>
              <code className="block font-mono text-[10px] text-amber-300"># update: {installLifecyclePlan.updateCommand}</code>
              <code className="block font-mono text-[10px] text-rose-300"># uninstall: {installLifecyclePlan.uninstallCommand}</code>
            </div>
          </div>
        )}

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
