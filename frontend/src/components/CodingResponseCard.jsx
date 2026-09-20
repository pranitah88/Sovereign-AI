import React, { useState, useMemo } from 'react';
import { parseCodingResponse } from '../lib/codingParser';

/**
 * Lightweight pure-React Python syntax tokenizer.
 * Escapes cleanly as React virtual DOM nodes, preserving 100% of indentation and whitespace.
 * Keywords and tokens are rendered as inline span elements within the code line,
 * never broken into separate Markdown paragraph elements.
 */
function highlightPythonLine(line) {
  if (!line) return '\u00A0';

  // Comprehensive Python token regex:
  // 1: Comments (#...)
  // 2: Strings ("..." or '...' or multi-line quotes)
  // 3: Keywords (def, class, import, return, etc.)
  // 4: Constants / Booleans (True, False, None, self, cls)
  // 5: Built-in functions (print, len, range, int, str, etc.)
  // 6: Numbers (int, float, hex)
  // 7: Function call names (ident followed by '(')
  const TOKEN_REGEX = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b(?:def|class|return|if|elif|else|for|while|try|except|finally|with|as|import|from|in|is|not|and|or|pass|break|continue|lambda|yield|raise|global|nonlocal|assert|async|await)\b)|(\b(?:True|False|None|self|cls)\b)|(\b(?:print|range|len|int|str|float|list|dict|set|tuple|bool|enumerate|zip|min|max|sum|sorted|map|filter|any|all|isinstance|open|type|abs|round|id|input)\b)|(\b\d+(?:\.\d+)?\b)|([a-zA-Z_]\w*(?=\s*\())/g;

  const elements = [];
  let lastIndex = 0;
  let match;

  while ((match = TOKEN_REGEX.exec(line)) !== null) {
    if (match.index > lastIndex) {
      elements.push(line.substring(lastIndex, match.index));
    }

    const [full, comment, string, keyword, constant, builtin, number, funcCall] = match;

    if (comment) {
      elements.push(<span key={match.index} className="tok-comment">{comment}</span>);
    } else if (string) {
      elements.push(<span key={match.index} className="tok-string">{string}</span>);
    } else if (keyword) {
      elements.push(<span key={match.index} className="tok-keyword">{keyword}</span>);
    } else if (constant) {
      elements.push(<span key={match.index} className="tok-constant">{constant}</span>);
    } else if (builtin) {
      elements.push(<span key={match.index} className="tok-builtin">{builtin}</span>);
    } else if (number) {
      elements.push(<span key={match.index} className="tok-number">{number}</span>);
    } else if (funcCall) {
      elements.push(<span key={match.index} className="tok-function">{funcCall}</span>);
    } else {
      elements.push(full);
    }

    lastIndex = match.index + full.length;
  }

  if (lastIndex < line.length) {
    elements.push(line.substring(lastIndex));
  }

  return elements.length > 0 ? elements : '\u00A0';
}

/**
 * Dedicated Coding Response Presentation Component
 * 
 * Order of presentation:
 * 1. Compact Task Metadata (CODING • model • VERIFIED)
 * 2. Dedicated Code Block / Card (CODE, Python, Copy, Download -> line numbers + code)
 * 3. Dedicated Execution Output Card (OUTPUT -> stdout)
 * 4. Dedicated Error Card if stderr exists (ERROR -> stderr)
 * 5. Collapsible Execution Details (Real Docker Sandbox properties)
 */
export default function CodingResponseCard({ msg }) {
  const parsed = useMemo(() => parseCodingResponse(msg), [msg]);
  const lines = useMemo(() => (parsed?.code ? parsed.code.split('\n') : []), [parsed]);
  const [copied, setCopied] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  if (!parsed) {
    return null;
  }

  // Copy button: Copies strictly ONLY the Python code
  const handleCopyCode = async (e) => {
    e.stopPropagation();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(parsed.code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
        return;
      }
    } catch (err) {
      console.warn('Clipboard write failed, using fallback:', err);
    }

    try {
      const ta = document.createElement('textarea');
      ta.value = parsed.code;
      ta.style.position = 'fixed';
      ta.style.left = '-999999px';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (fallbackErr) {
      console.error('Fallback copy failed:', fallbackErr);
    }
  };

  // Download button: Downloads strictly ONLY the Python code file
  const handleDownloadCode = (e) => {
    e.stopPropagation();
    const blob = new Blob([parsed.code], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mrpl_script.py';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="coding-response-container">
      {/* 1. Optional Conversational Intro (if present and clean) */}
      {parsed.remainingText && (
        <div className="coding-intro-text">{parsed.remainingText}</div>
      )}

      {/* 2. Compact Execution Metadata Row */}
      <div className="coding-metadata-bar" aria-label="Task Metadata">
        <span className="coding-meta-tag task-type">CODING</span>
        <span className="coding-meta-dot">•</span>
        <span className="coding-meta-tag model-name">{parsed.model}</span>
        <span className="coding-meta-dot">•</span>
        <span className={`coding-meta-tag status-tag ${parsed.isVerified ? 'verified' : parsed.isBlocked ? 'blocked' : 'failed'}`}>
          {parsed.isVerified ? 'VERIFIED' : parsed.isBlocked ? 'BLOCKED' : 'FAILED'}
        </span>

        {parsed.attempts && parsed.attempts > 1 && (
          <>
            <span className="coding-meta-dot">•</span>
            <span className="coding-meta-tag attempts">
              Attempts: {parsed.attempts}
            </span>
          </>
        )}

        {parsed.sandboxInfo && (
          <>
            <span className="coding-meta-dot">•</span>
            <span className="coding-meta-tag sandbox-badge">
              Sandbox: {parsed.sandboxInfo.sandbox}
            </span>
            {parsed.sandboxInfo.network && (
              <>
                <span className="coding-meta-dot">•</span>
                <span className="coding-meta-tag network-badge">
                  Network: {parsed.sandboxInfo.network}
                </span>
              </>
            )}
            {parsed.exitCode !== null && (
              <>
                <span className="coding-meta-dot">•</span>
                <span className="coding-meta-tag exit-code-badge">
                  Exit Code: {parsed.exitCode}
                </span>
              </>
            )}
          </>
        )}
      </div>

      {/* 3. Dedicated Code Block / Card */}
      <div className="coding-card code-card" aria-label="Generated Python Code">
        <div className="coding-card-header code-header">
          <div className="code-header-title">
            <span className="code-label">CODE</span>
            <span className="code-lang-badge">Python</span>
          </div>

          <div className="code-header-actions">
            <button
              type="button"
              className={`btn-code-action btn-copy ${copied ? 'copied' : ''}`}
              onClick={handleCopyCode}
              title="Copy code to clipboard"
              aria-label="Copy Python code"
            >
              <span className="btn-icon" aria-hidden="true">
                {copied ? (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" focusable="false" role="img">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                )}
              </span>
              <span className="btn-text">{copied ? 'Copied!' : 'Copy'}</span>
            </button>

            <button
              type="button"
              className="btn-code-action btn-download"
              onClick={handleDownloadCode}
              title="Download Python script"
              aria-label="Download Python code"
            >
              <span className="btn-icon" aria-hidden="true">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" focusable="false" role="img">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
              </span>
              <span className="btn-text">Download</span>
            </button>
          </div>
        </div>

        {/* Code Body with Line Numbers and Monospace Indentation */}
        <div className="code-card-body">
          <div className="code-line-numbers" aria-hidden="true">
            {lines.map((_, i) => (
              <span key={i} className="line-num">{i + 1}</span>
            ))}
          </div>

          <pre className="code-scroll-content">
            <code className="code-text-block">
              {lines.map((line, i) => (
                <span key={i} className="code-editor-line">
                  {highlightPythonLine(line)}
                </span>
              ))}
            </code>
          </pre>
        </div>
      </div>

      {/* 4. Dedicated Execution Output Card (stdout) */}
      {parsed.stdout && (
        <div className="coding-card output-card output-verified" aria-label="Execution Output Console">
          <div className="coding-card-header output-header">
            <span className="output-label">OUTPUT</span>

            <div className="output-header-badge">
              <span className="output-status-pill status-verified">
                <span className="pill-icon" aria-hidden="true">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </span>
                <span className="pill-text">✓ Execution Verified</span>
              </span>
            </div>
          </div>

          <div className="output-console-body">
            <pre className="console-text">{parsed.stdout}</pre>
          </div>
        </div>
      )}

      {/* 5. Dedicated Execution Error Card (stderr) */}
      {parsed.stderr && (
        <div className="coding-card output-card error-card output-failed" aria-label="Execution Error Console">
          <div className="coding-card-header error-header">
            <span className="error-label">ERROR</span>

            <div className="output-header-badge">
              <span className="output-status-pill status-failed">
                <span className="pill-icon" aria-hidden="true">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </span>
                <span className="pill-text">✕ Execution Failed</span>
              </span>
            </div>
          </div>

          <div className="output-console-body">
            <pre className="console-text console-error">{parsed.stderr}</pre>
          </div>
        </div>
      )}

      {/* 6. Execution Notice Banner (if any) */}
      {parsed.notice && !parsed.stdout && !parsed.stderr && (
        <div className="coding-notice-banner" role="alert">
          <span className="notice-icon" aria-hidden="true">⚠️</span>
          <span className="notice-text">{parsed.notice}</span>
        </div>
      )}

      {/* 7. Collapsible Execution Details (Real Docker Sandbox Parameters) */}
      {parsed.sandboxInfo && (
        <div className="execution-details-wrapper">
          <button
            type="button"
            className="execution-details-toggle"
            onClick={() => setShowDetails(!showDetails)}
            aria-expanded={showDetails}
          >
            <span className="details-chevron" aria-hidden="true">{showDetails ? '▼' : '▶'}</span>
            <span className="details-title">Execution Details</span>
          </button>

          {showDetails && (
            <div className="execution-details-content">
              <div className="details-grid">
                {parsed.sandboxInfo.sandbox && (
                  <div className="detail-item">
                    <span className="detail-label">Sandbox:</span>
                    <span className="detail-value">{parsed.sandboxInfo.sandbox}</span>
                  </div>
                )}
                {parsed.sandboxInfo.network && (
                  <div className="detail-item">
                    <span className="detail-label">Network:</span>
                    <span className="detail-value">{parsed.sandboxInfo.network}</span>
                  </div>
                )}
                {parsed.sandboxInfo.filesystem && (
                  <div className="detail-item">
                    <span className="detail-label">Filesystem:</span>
                    <span className="detail-value">{parsed.sandboxInfo.filesystem}</span>
                  </div>
                )}
                {parsed.sandboxInfo.user && (
                  <div className="detail-item">
                    <span className="detail-label">User:</span>
                    <span className="detail-value">{parsed.sandboxInfo.user}</span>
                  </div>
                )}
                {parsed.exitCode !== null && (
                  <div className="detail-item">
                    <span className="detail-label">Exit Code:</span>
                    <span className="detail-value">{parsed.exitCode}</span>
                  </div>
                )}
                {parsed.sandboxInfo.status && (
                  <div className="detail-item">
                    <span className="detail-label">Status:</span>
                    <span className="detail-value">{parsed.sandboxInfo.status}</span>
                  </div>
                )}
                {parsed.attempts && (
                  <div className="detail-item">
                    <span className="detail-label">Correction Attempts:</span>
                    <span className="detail-value">{parsed.attempts}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
