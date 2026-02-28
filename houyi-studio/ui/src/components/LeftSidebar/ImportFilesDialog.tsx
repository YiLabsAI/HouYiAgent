/**
 * Import Files Dialog - For importing files into a knowledge library
 * Supports path input, drag-and-drop file upload, and import history
 */
import React, { useState, useRef, useEffect } from 'react';
import { X, FolderOpen, File, AlertCircle, StopCircle, Upload, Clock, Loader2 } from 'lucide-react';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';

const IMPORT_HISTORY_KEY = 'houyi_import_history';
const MAX_HISTORY_ITEMS = 10;

interface ImportHistoryEntry {
  paths: string[];
  libraryId: string;  // Library ID for filtering
  libraryName: string;
  timestamp: string;
  success: boolean;  // Only show successful imports
}

function loadImportHistory(): ImportHistoryEntry[] {
  try {
    const stored = localStorage.getItem(IMPORT_HISTORY_KEY);
    const history: ImportHistoryEntry[] = stored ? JSON.parse(stored) : [];
    // Only return entries where success is explicitly true (filter out old/failed entries)
    return history.filter(h => h.success === true);
  } catch {
    return [];
  }
}

function saveImportHistory(history: ImportHistoryEntry[]): void {
  try {
    localStorage.setItem(IMPORT_HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS)));
  } catch {
    // Ignore localStorage errors
  }
}

// Export for use in store when import succeeds
export function addSuccessfulImport(paths: string[], libraryId: string, libraryName: string): void {
  try {
    const history = loadImportHistory();
    const entry: ImportHistoryEntry = {
      paths,
      libraryId,
      libraryName,
      timestamp: new Date().toISOString(),
      success: true,
    };
    // Filter out entries with same paths AND same library
    const updated = [entry, ...history.filter(
      (h) => !(JSON.stringify(h.paths) === JSON.stringify(paths) && h.libraryId === libraryId)
    )].slice(0, MAX_HISTORY_ITEMS);
    saveImportHistory(updated);
  } catch {
    // Ignore errors
  }
}

interface ImportFilesDialogProps {
  isOpen: boolean;
  libraryName: string;
  libraryId: string;
  onImport: (paths: string[]) => void;
  onCancel: () => void;
}

export const ImportFilesDialog: React.FC<ImportFilesDialogProps> = ({
  isOpen,
  libraryName,
  libraryId,
  onImport,
  onCancel,
}) => {
  const [paths, setPaths] = useState<string>('');
  const [importType, setImportType] = useState<'directory' | 'files'>('directory');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [importHistory, setImportHistory] = useState<ImportHistoryEntry[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingImportPaths, setPendingImportPaths] = useState<string[]>([]);
  const dragCounter = useRef(0);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Reset state when dialog opens
      setPaths('');
      setUploadError(null);
      setConfirmOpen(false);
      setPendingImportPaths([]);
      // Filter history to only show imports for the current library
      const allHistory = loadImportHistory();
      const filtered = allHistory.filter((h) => {
        // Match by libraryId (new records)
        if (h.libraryId === libraryId) return true;
        // Match by libraryId extracted from path (old records without libraryId)
        // Path format: .../lib_xxx/uploads/...
        const pathMatch = h.paths[0]?.match(/\/lib_([a-f0-9]+)\//);
        if (pathMatch && `lib_${pathMatch[1]}` === libraryId) return true;
        // Fallback: match by libraryName (only if libraryId not available)
        if (!h.libraryId && h.libraryName === libraryName) return true;
        return false;
      });
      setImportHistory(filtered);
    }
  }, [isOpen, libraryId, libraryName]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const pathList = paths
      .split('\n')
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    if (pathList.length > 0) {
      setPendingImportPaths(pathList);
      setConfirmOpen(true);
    }
  };

  const handleConfirmImport = () => {
    if (pendingImportPaths.length === 0) {
      setConfirmOpen(false);
      return;
    }
    // Don't save to history here - only save on successful completion
    // The store will call addSuccessfulImport() when ingest succeeds
    onImport(pendingImportPaths);
    setConfirmOpen(false);
    setPendingImportPaths([]);
  };

  const handleCancelConfirm = () => {
    setConfirmOpen(false);
  };

  const handleHistorySelect = (entry: ImportHistoryEntry) => {
    setPaths(entry.paths.join('\n'));
    setImportType(entry.paths.length === 1 && !entry.paths[0].includes('.') ? 'directory' : 'files');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onCancel();
    }
  };

  // Drag and drop handlers
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const appendPaths = (incoming: string[]) => {
    const normalized = incoming
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    if (normalized.length === 0) {
      return;
    }
    const existing = paths
      .split('\n')
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    const merged = [...existing];
    for (const p of normalized) {
      if (!merged.includes(p)) {
        merged.push(p);
      }
    }
    setPaths(merged.join('\n'));
  };

  // Helper to read directory entries recursively
  const readDirectoryEntries = async (entry: FileSystemDirectoryEntry): Promise<File[]> => {
    const files: File[] = [];
    const reader = entry.createReader();

    const readEntries = (): Promise<FileSystemEntry[]> => {
      return new Promise((resolve, reject) => {
        reader.readEntries(resolve, reject);
      });
    };

    const getFile = (fileEntry: FileSystemFileEntry): Promise<File> => {
      return new Promise((resolve, reject) => {
        fileEntry.file(resolve, reject);
      });
    };

    // Read all entries (readEntries may need to be called multiple times)
    let entries: FileSystemEntry[] = [];
    let batch: FileSystemEntry[];
    do {
      batch = await readEntries();
      entries = entries.concat(batch);
    } while (batch.length > 0);

    for (const e of entries) {
      if (e.isFile) {
        const file = await getFile(e as FileSystemFileEntry);
        files.push(file);
      } else if (e.isDirectory) {
        const subFiles = await readDirectoryEntries(e as FileSystemDirectoryEntry);
        files.push(...subFiles);
      }
    }
    return files;
  };

  const uploadDroppedOrPickedFiles = async (files: File[]) => {
    if (files.length === 0) {
      setUploadError('No files found. If you dropped a directory, it may be empty.');
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      const response = await fetch(`/api/knowledge/${libraryId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `Upload failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.errors && result.errors.length > 0) {
        setUploadError(`Some files failed: ${result.errors.join(', ')}`);
      }

      if (result.uploaded_paths && result.uploaded_paths.length > 0) {
        appendPaths(result.uploaded_paths);
        setImportType('files');
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleBrowseFile = () => {
    fileInputRef.current?.click();
  };

  const handleBrowseFolder = () => {
    folderInputRef.current?.click();
  };

  const handleFilePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (!selected || selected.length === 0) {
      return;
    }
    setUploadError(null);
    await uploadDroppedOrPickedFiles(Array.from(selected));
    e.target.value = '';
  };

  const handleFolderPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (!selected || selected.length === 0) {
      return;
    }
    setUploadError(null);
    await uploadDroppedOrPickedFiles(Array.from(selected));
    e.target.value = '';
  };

  // Get all files from dataTransfer, including directory contents
  const getFilesFromDataTransfer = async (dataTransfer: DataTransfer): Promise<File[]> => {
    const files: File[] = [];
    const items = Array.from(dataTransfer.items);

    for (const item of items) {
      if (item.kind !== 'file') continue;

      // Try webkitGetAsEntry for directory support
      const entry = item.webkitGetAsEntry?.();
      if (entry) {
        if (entry.isDirectory) {
          const dirFiles = await readDirectoryEntries(entry as FileSystemDirectoryEntry);
          files.push(...dirFiles);
        } else if (entry.isFile) {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      } else {
        // Fallback to getAsFile
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    return files;
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;
    setUploadError(null);

    const droppedText = e.dataTransfer.getData('text/plain').trim();
    if (droppedText) {
      appendPaths([droppedText]);
      setImportType('directory');
      return;
    }

    // Get files including directory contents
    const files = await getFilesFromDataTransfer(e.dataTransfer);
    await uploadDroppedOrPickedFiles(files);
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onKeyDown={handleKeyDown}
    >
      <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h3 className="text-sm font-semibold text-gray-50">Import Files</h3>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-gray-50 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Library info */}
          <div className="text-xs text-gray-400">
            Importing into: <span className="text-gray-50">{libraryName}</span>
          </div>

          {/* Import type selection */}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setImportType('directory')}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs transition-colors ${
                importType === 'directory'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <FolderOpen size={14} />
              Directory
            </button>
            <button
              type="button"
              onClick={() => setImportType('files')}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs transition-colors ${
                importType === 'files'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <File size={14} />
              Files
            </button>
          </div>

          <div className="flex gap-2">
            {importType === 'files' ? (
              <button
                type="button"
                onClick={handleBrowseFile}
                disabled={isUploading}
                className="flex-1 px-3 py-2 rounded border border-gray-700 bg-gray-900/40 text-xs text-gray-300 hover:border-gray-600 disabled:opacity-50"
                data-testid="import-browse-file"
              >
                Browse Files
              </button>
            ) : (
              <button
                type="button"
                onClick={handleBrowseFolder}
                disabled={isUploading}
                className="flex-1 px-3 py-2 rounded border border-gray-700 bg-gray-900/40 text-xs text-gray-300 hover:border-gray-600 disabled:opacity-50"
                data-testid="import-browse-folder"
              >
                Browse Folder
              </button>
            )}
          </div>

          {/* Drop zone + Path input */}
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-lg transition-colors ${
              isDragging
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 bg-gray-900'
            }`}
          >
            {isDragging ? (
              <div className="flex flex-col items-center justify-center py-8 text-blue-400">
                <Upload size={24} className="mb-2" />
                <span className="text-xs">Drop files here to upload</span>
              </div>
            ) : isUploading ? (
              <div className="flex flex-col items-center justify-center py-8 text-blue-400">
                <Loader2 size={24} className="mb-2 animate-spin" />
                <span className="text-xs">Uploading files...</span>
              </div>
            ) : (
              <div>
                <label className="block text-xs text-gray-400 px-3 pt-2 mb-1">
                  {importType === 'directory'
                    ? 'Directory path (one per line)'
                    : 'File paths (one per line)'}
                </label>
                <textarea
                  value={paths}
                  onChange={(e) => setPaths(e.target.value)}
                  placeholder={
                    importType === 'directory'
                      ? './knowledge\n./docs/guides'
                      : './knowledge/doc1.md\n./knowledge/doc2.txt'
                  }
                  className="w-full h-24 px-3 py-2 bg-transparent text-xs text-gray-50 placeholder-gray-500 focus:outline-none resize-none"
                  autoFocus
                />
                <div className="px-3 pb-2 text-[10px] text-gray-600">
                  drag &amp; drop local files/folders to upload, or drop a plain-text path
                </div>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFilePicked}
            className="hidden"
            data-testid="import-file-picker"
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore - webkitdirectory is supported in Chromium-based runtimes.
            webkitdirectory=""
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore - directory is non-standard but supported by some environments.
            directory=""
            onChange={handleFolderPicked}
            className="hidden"
            data-testid="import-folder-picker"
          />

          {/* Upload error */}
          {uploadError && (
            <div className="flex items-start gap-2 p-2 bg-red-900/30 border border-red-700 rounded">
              <AlertCircle size={14} className="text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-[10px] text-red-300">{uploadError}</div>
            </div>
          )}

          {/* Import history */}
          {importHistory.length > 0 && !paths.trim() && !isUploading && (
            <div>
              <div className="flex items-center gap-1 text-[10px] text-gray-500 mb-1">
                <Clock size={10} />
                <span>Recent imports</span>
              </div>
              <div className="space-y-1 max-h-24 overflow-y-auto">
                {importHistory.slice(0, 4).map((entry, i) => {
                  // Show only filenames, not full paths
                  const displayPaths = entry.paths.map((p) => {
                    const parts = p.split('/');
                    return parts[parts.length - 1] || p;
                  }).join(', ');
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => handleHistorySelect(entry)}
                      className="w-full text-left px-2 py-1 rounded text-[10px] text-gray-400 hover:bg-gray-700 hover:text-gray-50 transition-colors truncate"
                      title={entry.paths.join('\n')}
                    >
                      {displayPaths}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Supported formats info */}
          <div className="flex items-start gap-2 p-2 bg-gray-900 rounded">
            <AlertCircle size={14} className="text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="text-[10px] text-gray-400">
              Supported formats: .md, .txt, .pdf, .json, .csv, .html/.htm, .xlsx/.xlsm/.xls, .doc/.docx, .pptx, .epub
              <br />
              Drag &amp; drop files to upload, or enter server paths manually
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-xs text-gray-300 hover:text-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!paths.trim() || isUploading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-gray-50 text-xs rounded transition-colors"
            >
              Import
            </button>
          </div>
        </form>
      </div>

      <ConfirmDialog
        isOpen={confirmOpen}
        title="Confirm Import"
        message={`Start knowledge import for ${pendingImportPaths.length} path(s)?`}
        itemName={libraryName}
        confirmText="Start Import"
        cancelText="Back"
        variant="default"
        onConfirm={handleConfirmImport}
        onCancel={handleCancelConfirm}
      />
    </div>
  );
};

interface IngestProgressProps {
  isIngesting: boolean;
  progress: number;
  currentFile: string;
  filesProcessed: number;
  totalFiles: number;
  onCancel?: () => void;
}

export const IngestProgress: React.FC<IngestProgressProps> = ({
  isIngesting,
  progress,
  currentFile,
  filesProcessed,
  totalFiles,
  onCancel,
}) => {
  if (!isIngesting) return null;

  return (
    <div className="p-3 bg-gray-900 border border-gray-700 rounded">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-300">Importing files...</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">
            {filesProcessed} / {totalFiles}
          </span>
          {onCancel && (
            <button
              onClick={onCancel}
              className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-red-400 transition-colors"
              title="Cancel import"
            >
              <StopCircle size={12} />
            </button>
          )}
        </div>
      </div>
      <div className="w-full h-1.5 bg-gray-700 rounded overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      {currentFile && (
        <div className="mt-1 text-[10px] text-gray-500 truncate">
          {currentFile}
        </div>
      )}
    </div>
  );
};
