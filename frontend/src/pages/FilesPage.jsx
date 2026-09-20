import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { documents } from '../api/client';

function getFileTypeLabel(filename) {
  const ext = filename?.split('.').pop()?.toLowerCase();
  if (['docx', 'doc'].includes(ext)) return 'DOCX';
  if (['xlsx', 'xls'].includes(ext)) return 'XLSX';
  if (ext === 'csv') return 'CSV';
  if (['pptx', 'ppt'].includes(ext)) return 'PPTX';
  if (ext === 'pdf') return 'PDF';
  if (['png', 'jpg', 'jpeg'].includes(ext)) return 'IMG';
  if (ext === 'txt') return 'TXT';
  if (ext === 'md') return 'MD';
  return ext?.toUpperCase() || 'FILE';
}

function getFileTypeIcon(type) {
  switch (type) {
    case 'PDF': return '📄';
    case 'DOCX': return '📝';
    case 'XLSX':
    case 'CSV': return '📊';
    case 'PPTX': return '📽️';
    default: return '📁';
  }
}

export default function FilesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get('tab') === 'uploaded' ? 'uploaded' : 'generated';
  const [generatedFiles, setGeneratedFiles] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const handleTabChange = (newTab) => {
    setSearchParams({ tab: newTab });
  };

  const loadFiles = async () => {
    setLoading(true);
    setError(null);
    try {
      const [genRes, upRes] = await Promise.all([
        documents.listGenerated().catch(() => []),
        documents.list().catch(() => []),
      ]);
      setGeneratedFiles(Array.isArray(genRes) ? genRes : []);
      setUploadedFiles(Array.isArray(upRes) ? upRes : []);
    } catch (err) {
      setError(err.message || 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFiles();
  }, []);

  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (isoString) => {
    if (!isoString) return '—';
    try {
      return new Date(isoString).toLocaleString();
    } catch {
      return isoString;
    }
  };

  const filteredGenerated = generatedFiles.filter((f) =>
    (f.original_name || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredUploaded = uploadedFiles.filter((f) =>
    (f.original_name || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="page-content">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">
            {activeTab === 'generated' ? 'Deliverables & Generated Outputs' : 'Uploaded Repository Documents'}
          </h1>
          <p className="page-subtitle">
            Enterprise document repository, sandboxed calculation outputs, and verified deliverables.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={loadFiles} disabled={loading}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Tabs & Search controls */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', background: '#ffffff', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '2px' }}>
          <button
            className={`btn btn-sm ${activeTab === 'generated' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => handleTabChange('generated')}
            style={{ borderRadius: 'var(--radius-xs)' }}
          >
            Generated Outputs ({generatedFiles.length})
          </button>
          <button
            className={`btn btn-sm ${activeTab === 'uploaded' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => handleTabChange('uploaded')}
            style={{ borderRadius: 'var(--radius-xs)' }}
          >
            Repository Documents ({uploadedFiles.length})
          </button>
        </div>

        <div style={{ flex: 1, maxWidth: '320px', marginLeft: 'auto' }}>
          <input
            type="text"
            className="input"
            placeholder="Search files by name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Filter files"
          />
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-16" role="alert">{error}</div>
      )}

      {loading ? (
        <div className="empty-state">
          <div className="spinner spinner-lg" />
        </div>
      ) : activeTab === 'generated' ? (
        filteredGenerated.length === 0 ? (
          <div className="empty-state card">
            <div className="empty-state-title">No generated outputs yet</div>
            <div className="empty-state-description">
              Ask the Chat Agent to synthesize reports (.docx), create compliance PDFs, or generate structured spreadsheets.
            </div>
          </div>
        ) : (
          <div className="card" style={{ overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Output Document</th>
                  <th>Format</th>
                  <th>Size</th>
                  <th>Created</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredGenerated.map((file) => {
                  const type = file.file_type?.toUpperCase() || getFileTypeLabel(file.original_name);
                  const isPdf = type === 'PDF' || file.original_name?.toLowerCase().endsWith('.pdf');
                  const token = localStorage.getItem('mrpl_token');
                  const previewUrl = `${documents.downloadGeneratedUrl(file.id)}?inline=true${token ? `&token=${encodeURIComponent(token)}` : ''}`;

                  return (
                    <tr key={file.id}>
                      <td>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span aria-hidden="true">{getFileTypeIcon(type)}</span>
                          <span>{file.original_name}</span>
                        </div>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                          Deliverable ID #{file.id} · Generated on-premise
                        </div>
                      </td>
                      <td>
                        <span className="badge badge-info">{type}</span>
                      </td>
                      <td>{formatBytes(file.file_size_bytes)}</td>
                      <td className="text-sm">{formatDate(file.created_at)}</td>
                      <td style={{ textAlign: 'right' }}>
                        <div className="flex gap-6" style={{ justifyContent: 'flex-end' }}>
                          {isPdf && (
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => window.open(previewUrl, '_blank')}
                              title="Preview PDF"
                              aria-label={`Preview ${file.original_name}`}
                            >
                              Preview
                            </button>
                          )}
                          <a
                            href={documents.downloadGeneratedUrl(file.id)}
                            download
                            className="btn btn-primary btn-sm"
                            style={{ textDecoration: 'none' }}
                            title={`Download ${file.original_name}`}
                            aria-label={`Download ${file.original_name}`}
                          >
                            Download
                          </a>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )
      ) : filteredUploaded.length === 0 ? (
        <div className="empty-state card">
          <div className="empty-state-title">No uploaded documents</div>
          <div className="empty-state-description">
            Upload refinery manuals, specifications, and vigilance policies to index them for local RAG retrieval.
          </div>
        </div>
      ) : (
        <div className="card" style={{ overflow: 'hidden' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Document</th>
                <th>Collection</th>
                <th>Classification</th>
                <th>Size</th>
                <th>Status</th>
                <th>Uploaded</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredUploaded.map((file) => {
                const type = getFileTypeLabel(file.original_name);
                return (
                  <tr key={file.id}>
                    <td>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span aria-hidden="true">{getFileTypeIcon(type)}</span>
                        <span>{file.original_name}</span>
                      </div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                        {type} · ID #{file.id}
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-neutral">{file.department || file.category || 'GENERAL'}</span>
                    </td>
                    <td>
                      <span className="badge badge-clearance-internal">
                        {(file.classification || 'INTERNAL').toUpperCase()}
                      </span>
                    </td>
                    <td>{formatBytes(file.file_size_bytes)}</td>
                    <td>
                      <span
                        className={`badge ${
                          file.index_status === 'indexed' || file.index_status === 'completed'
                            ? 'badge-success'
                            : file.index_status === 'indexing'
                            ? 'badge-warning'
                            : file.index_status === 'failed'
                            ? 'badge-error'
                            : 'badge-neutral'
                        }`}
                      >
                        {file.index_status || 'pending'}
                      </span>
                    </td>
                    <td className="text-sm">{formatDate(file.uploaded_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <a
                        href={documents.downloadUrl(file.id)}
                        download
                        className="btn btn-secondary btn-sm"
                        style={{ textDecoration: 'none' }}
                        title={`Download ${file.original_name}`}
                        aria-label={`Download ${file.original_name}`}
                      >
                        Download
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
