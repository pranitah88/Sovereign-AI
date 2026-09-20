import { useState, useEffect, useRef, useMemo } from 'react';
import { documents } from '../api/client';
import KnowledgeExplorerOverlay from '../components/KnowledgeExplorerOverlay';

export default function DocumentsPage() {
  const [activeTab, setActiveTab] = useState('map'); // 'collections' | 'documents' | 'map'
  const [showExplorer, setShowExplorer] = useState(false);
  const [kmData, setKmData] = useState(null);
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);

  // Filters & Search
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('ALL');
  const [filterClassification, setFilterClassification] = useState('ALL');
  const [filterDocType, setFilterDocType] = useState('ALL');
  const [filterUnit, setFilterUnit] = useState('ALL');

  // Interactive selection in Knowledge Map
  const [selectedCategoryCode, setSelectedCategoryCode] = useState(null);
  const [inspectedDoc, setInspectedDoc] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [mapResponse, uploadList] = await Promise.all([
        documents.getKnowledgeMap(),
        documents.list().catch(() => []),
      ]);
      setKmData(mapResponse);
      setUploadedDocs(Array.isArray(uploadList) ? uploadList : []);
    } catch (err) {
      setError(err.message || 'Failed to load Knowledge Base & Map');
      console.error('Error loading knowledge data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await documents.upload(file);
      await loadAllData();
    } catch (err) {
      setError('Upload failed: ' + err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDelete = async (id, name) => {
    if (!confirm(`Delete document "${name}" from the Knowledge Base?`)) return;
    try {
      await documents.delete(id);
      await loadAllData();
      if (inspectedDoc && inspectedDoc.file_id === id) {
        setInspectedDoc(null);
      }
    } catch (err) {
      setError('Delete failed: ' + err.message);
    }
  };

  const handleReindex = async (id) => {
    try {
      await documents.reindex(id);
      await loadAllData();
    } catch (err) {
      setError('Reindex failed: ' + err.message);
    }
  };

  // Flatten all documents across categories for global listing and filtering
  const allCorpusDocs = useMemo(() => {
    if (!kmData?.categories) return [];
    const list = [];
    for (const cat of kmData.categories) {
      for (const doc of cat.documents || []) {
        list.push({
          ...doc,
          category_code: cat.category_code,
          category_display_name: cat.name,
        });
      }
    }
    return list;
  }, [kmData]);

  // Document types list for filter dropdown
  const docTypes = useMemo(() => {
    const types = new Set();
    allCorpusDocs.forEach(d => {
      if (d.document_type) types.add(d.document_type.toUpperCase());
    });
    return Array.from(types).sort();
  }, [allCorpusDocs]);

  // Available categories for filter dropdown
  const categoriesList = useMemo(() => {
    return kmData?.categories || [];
  }, [kmData]);

  // Filtered documents matching current filters
  const filteredDocs = useMemo(() => {
    return allCorpusDocs.filter(doc => {
      // Category filter
      if (filterCategory !== 'ALL' && doc.category !== filterCategory && doc.category_code !== filterCategory) {
        return false;
      }
      // Classification filter
      if (filterClassification !== 'ALL') {
        const docCls = (doc.classification || 'INTERNAL').toUpperCase();
        if (docCls !== filterClassification) return false;
      }
      // Document type filter
      if (filterDocType !== 'ALL') {
        const type = (doc.document_type || '').toUpperCase();
        if (type !== filterDocType) return false;
      }
      // Unit filter (CDU, HCU, PFCCU, P&ID, DESAL)
      if (filterUnit !== 'ALL') {
        const units = doc.units_detected || [];
        if (!units.includes(filterUnit)) return false;
      }
      // Search term
      if (searchTerm.trim()) {
        const query = searchTerm.toLowerCase();
        const matchTitle = (doc.title || '').toLowerCase().includes(query);
        const matchFilename = (doc.filename || '').toLowerCase().includes(query);
        const matchCategory = (doc.category_name || '').toLowerCase().includes(query);
        const matchDept = (doc.department || '').toLowerCase().includes(query);
        const matchUnits = (doc.units_detected || []).some(u => u.toLowerCase().includes(query));
        if (!matchTitle && !matchFilename && !matchCategory && !matchDept && !matchUnits) {
          return false;
        }
      }
      return true;
    });
  }, [allCorpusDocs, filterCategory, filterClassification, filterDocType, filterUnit, searchTerm]);

  // Active category in Knowledge Map view (if selected)
  const activeCategory = useMemo(() => {
    if (!selectedCategoryCode || !kmData?.categories) return null;
    return kmData.categories.find(c => c.category_code === selectedCategoryCode) || null;
  }, [selectedCategoryCode, kmData]);

  // Classification Badge Helper
  const renderClassificationBadge = (cls) => {
    const clean = (cls || 'INTERNAL').toUpperCase();
    if (clean === 'CONFIDENTIAL' || clean === 'HIGHLY_CONFIDENTIAL') {
      return <span className="badge badge-clearance-confidential">CONFIDENTIAL</span>;
    }
    if (clean === 'RESTRICTED') {
      return <span className="badge badge-clearance-restricted">RESTRICTED</span>;
    }
    if (clean === 'PUBLIC') {
      return <span className="badge badge-clearance-public">PUBLIC</span>;
    }
    return <span className="badge badge-clearance-internal">INTERNAL</span>;
  };

  // Status Badge Helper
  const renderStatusBadge = (status = 'indexed') => {
    const clean = (status || 'indexed').toLowerCase();
    if (clean === 'indexed' || clean === 'completed') {
      return <span className="badge badge-success">INDEXED</span>;
    }
    if (clean === 'indexing') {
      return <span className="badge badge-warning">INDEXING</span>;
    }
    if (clean === 'failed') {
      return <span className="badge badge-error">FAILED</span>;
    }
    return <span className="badge badge-neutral">PENDING</span>;
  };

  // Document format badge helper
  const renderTypeBadge = (docType) => {
    const clean = (docType || 'FILE').toUpperCase();
    return (
      <span
        style={{
          fontSize: '0.66rem',
          fontWeight: 700,
          fontFamily: 'var(--font-mono)',
          padding: '1px 5px',
          background: 'var(--bg-tertiary)',
          color: 'var(--text-secondary)',
          border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-xs)',
        }}
      >
        {clean}
      </span>
    );
  };

  const _formatSize = (bytes) => {
    if (!bytes) return null;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
      {/* ── Page Header ────────────────────────────────────────────── */}
      <div className="page-header" style={{ marginBottom: '16px' }}>
        <div>
          <h1 className="page-title">Enterprise Knowledge Base</h1>
          <p className="page-subtitle">
            Authoritative on-premise refinery knowledge corpus indexed in ChromaDB vector store with RBAC access control
          </p>
        </div>

        <div className="flex gap-12 items-center">
          {/* Main View Tabs: [ Collections ] [ Documents ] [ Knowledge Map ] */}
          <div className="kb-tab-group" role="tablist" aria-label="Knowledge Base views">
            <button
              role="tab"
              aria-selected={activeTab === 'collections'}
              className={`kb-tab-btn ${activeTab === 'collections' ? 'active' : ''}`}
              onClick={() => setActiveTab('collections')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
              </svg>
              Collections
            </button>

            <button
              role="tab"
              aria-selected={activeTab === 'documents'}
              className={`kb-tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => setActiveTab('documents')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              Documents ({allCorpusDocs.length || 0})
            </button>

            <button
              role="tab"
              aria-selected={activeTab === 'map'}
              className={`kb-tab-btn ${activeTab === 'map' ? 'active' : ''}`}
              onClick={() => setActiveTab('map')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="5" r="3" />
                <circle cx="5" cy="19" r="3" />
                <circle cx="19" cy="19" r="3" />
                <path d="M12 8v4" />
                <path d="M5 16l4.5-4" />
                <path d="M19 16l-4.5-4" />
              </svg>
              Knowledge Map
            </button>
          </div>

          {/* ✦ Explore Knowledge Interactive Overlay Button */}
          <button
            className="kb-explore-btn"
            onClick={() => setShowExplorer(true)}
            title="Open interactive progressive Knowledge Explorer"
          >
            <span className="sparkle-icon">✦</span>
            Explore Knowledge
          </button>

          {/* Upload Button */}
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleUpload}
            style={{ display: 'none' }}
            accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.pptx,.ppt,.txt,.md"
            aria-label="Upload document to Knowledge Base"
          />
          <button
            className="btn btn-primary"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            title="Upload industrial document to knowledge base"
          >
            {uploading ? (
              <>
                <div className="spinner" /> Indexing Document...
              </>
            ) : (
              <>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Upload Document
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-16" role="alert">
          {error}
        </div>
      )}

      {/* ── Search & Filter Controls Bar ────────────────────────────── */}
      <div
        className="card"
        style={{
          padding: '12px 16px',
          marginBottom: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
        }}
      >
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Search Input */}
          <div style={{ flex: 2, minWidth: '260px' }}>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                className="input"
                placeholder="Search knowledge by document name, process unit, or topic..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{ paddingLeft: '32px' }}
                aria-label="Search knowledge base"
              />
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--text-tertiary)"
                strokeWidth="2"
                style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }}
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </div>
          </div>

          {/* Category Filter */}
          <div style={{ minWidth: '180px' }}>
            <select
              className="input"
              value={filterCategory}
              onChange={(e) => {
                setFilterCategory(e.target.value);
                if (e.target.value !== 'ALL') {
                  setSelectedCategoryCode(e.target.value);
                }
              }}
              aria-label="Filter by category"
            >
              <option value="ALL">All Categories ({kmData?.categories?.length || 0})</option>
              {categoriesList.map((cat) => (
                <option key={cat.category_code} value={cat.category_code}>
                  {cat.name} ({cat.document_count})
                </option>
              ))}
            </select>
          </div>

          {/* Classification Filter */}
          <div style={{ minWidth: '150px' }}>
            <select
              className="input"
              value={filterClassification}
              onChange={(e) => setFilterClassification(e.target.value)}
              aria-label="Filter by classification"
            >
              <option value="ALL">All Classifications</option>
              <option value="PUBLIC">PUBLIC</option>
              <option value="INTERNAL">INTERNAL</option>
              <option value="RESTRICTED">RESTRICTED</option>
              <option value="CONFIDENTIAL">CONFIDENTIAL</option>
            </select>
          </div>

          {/* Doc Type Filter */}
          <div style={{ minWidth: '120px' }}>
            <select
              className="input"
              value={filterDocType}
              onChange={(e) => setFilterDocType(e.target.value)}
              aria-label="Filter by document type"
            >
              <option value="ALL">All Formats</option>
              {docTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>

          {/* Clear Filters Button */}
          {(searchTerm || filterCategory !== 'ALL' || filterClassification !== 'ALL' || filterDocType !== 'ALL' || filterUnit !== 'ALL') && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => {
                setSearchTerm('');
                setFilterCategory('ALL');
                setFilterClassification('ALL');
                setFilterDocType('ALL');
                setFilterUnit('ALL');
                setSelectedCategoryCode(null);
              }}
              style={{ fontSize: '0.75rem' }}
            >
              Reset Filters
            </button>
          )}
        </div>

        {/* Quick Unit Filter Tags */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', paddingTop: '4px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>
            Refinery Process Units:
          </span>
          {['ALL', 'CDU', 'HCU', 'PFCCU', 'P&ID', 'DESAL'].map((unit) => (
            <button
              key={unit}
              className={`btn btn-sm ${filterUnit === unit ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setFilterUnit(unit)}
              style={{
                fontSize: '0.7rem',
                padding: '2px 8px',
                height: '24px',
                fontFamily: unit === 'ALL' ? 'inherit' : 'var(--font-mono)',
              }}
            >
              {unit === 'ALL' ? 'All Units' : unit}
            </button>
          ))}
          <span style={{ marginLeft: 'auto', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            Showing <strong>{filteredDocs.length}</strong> of {kmData?.root?.total_documents || 0} indexed documents
          </span>
        </div>
      </div>

      {/* Loading state */}
      {loading ? (
        <div className="empty-state" style={{ padding: '60px 0' }}>
          <div className="spinner spinner-lg" />
          <div style={{ marginTop: '14px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Connecting to ChromaDB Vector Store & Knowledge Base...
          </div>
        </div>
      ) : (
        <>
          {/* ════════════════════════════════════════════════════════════
              TAB 1: KNOWLEDGE MAP INTERACTIVE VISUALIZATION
              ════════════════════════════════════════════════════════════ */}
          {activeTab === 'map' && (
            <div className="km-canvas">
              {/* Root Node: MRPL Sovereign Knowledge Base */}
              <div className="km-root-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ fontSize: '1.2rem' }}>🏛️</span>
                  <div className="km-root-title">{kmData?.root?.name || 'MRPL Sovereign Knowledge Base'}</div>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)', maxWidth: '520px' }}>
                  Authoritative enterprise repository of Mangalore Refinery operational, engineering, and compliance data
                </div>

                <div className="km-root-stats">
                  <div className="km-stat-chip">
                    Total Documents: <strong>{kmData?.root?.total_documents || 0}</strong>
                  </div>
                  <div className="km-stat-chip">
                    Vector Chunks: <strong>{(kmData?.root?.total_chunks || 0).toLocaleString()}</strong>
                  </div>
                  <div className="km-stat-chip">
                    Collections: <strong>{kmData?.root?.total_categories || 0}</strong>
                  </div>
                  <div className="km-stat-chip" style={{ color: 'var(--success)', borderColor: 'var(--success-border)' }}>
                    ● {kmData?.root?.indexed_status || 'Fully Indexed'}
                  </div>
                  <div className="km-stat-chip">
                    Clearance: <strong style={{ color: 'var(--brand-green)' }}>{kmData?.root?.user_clearance || 'INTERNAL'}</strong>
                  </div>
                </div>
              </div>

              {/* Tree Connecting Trunk */}
              <div className="km-tree-trunk" />

              {/* Level 1: Category / Collection Nodes */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--brand-olive)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Enterprise Knowledge Collections ({kmData?.categories?.length || 0})
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                    Click a collection to filter documents below
                  </div>
                </div>

                <div className="km-categories-container">
                  {(kmData?.categories || []).map((cat) => {
                    const isSelected = selectedCategoryCode === cat.category_code;
                    return (
                      <div
                        key={cat.category_code}
                        className={`km-category-card ${isSelected ? 'selected' : ''}`}
                        onClick={() => {
                          if (isSelected) {
                            setSelectedCategoryCode(null);
                            setFilterCategory('ALL');
                          } else {
                            setSelectedCategoryCode(cat.category_code);
                            setFilterCategory(cat.category_code);
                          }
                        }}
                      >
                        <div className="km-category-header">
                          <div className="km-category-title">{cat.name}</div>
                          {renderClassificationBadge(cat.classification)}
                        </div>

                        <div className="km-category-desc">{cat.description}</div>

                        <div className="km-category-footer">
                          <span>
                            <strong>{cat.document_count}</strong> {cat.document_count === 1 ? 'doc' : 'docs'} ·{' '}
                            <strong>{cat.total_chunks.toLocaleString()}</strong> chunks
                          </span>
                          <span style={{ color: isSelected ? 'var(--brand-green)' : 'var(--text-tertiary)', fontWeight: 600 }}>
                            {isSelected ? '● Active' : 'Filter →'}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Level 2: Document Nodes Section */}
              <div className="km-docs-section">
                <div className="km-docs-header">
                  <div>
                    <h2 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--brand-olive)' }}>
                      {activeCategory ? activeCategory.name : 'Indexed Knowledge Documents'}
                    </h2>
                    <p style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                      {activeCategory
                        ? `Showing all documents in ${activeCategory.name} (${filteredDocs.length} items)`
                        : `Displaying all authorized knowledge assets (${filteredDocs.length} items)`}
                    </p>
                  </div>

                  {selectedCategoryCode && (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setSelectedCategoryCode(null);
                        setFilterCategory('ALL');
                      }}
                      style={{ fontSize: '0.74rem' }}
                    >
                      Show All Categories
                    </button>
                  )}
                </div>

                {filteredDocs.length === 0 ? (
                  <div className="empty-state" style={{ padding: '30px 0' }}>
                    <div style={{ fontSize: '0.86rem', color: 'var(--text-secondary)' }}>
                      No documents match the current filter criteria.
                    </div>
                  </div>
                ) : (
                  <div className="km-docs-grid">
                    {filteredDocs.map((doc) => {
                      const isInspected = inspectedDoc?.id === doc.id;
                      return (
                        <div
                          key={doc.id}
                          className={`km-doc-node ${isInspected ? 'active' : ''}`}
                          onClick={() => setInspectedDoc(doc)}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
                            <div className="km-doc-title" title={doc.title}>
                              {doc.title}
                            </div>
                            {renderTypeBadge(doc.document_type)}
                          </div>

                          <div className="km-doc-raw-name" title={doc.filename}>
                            {doc.filename}
                          </div>

                          <div className="km-doc-meta">
                            {renderClassificationBadge(doc.classification)}
                            {renderStatusBadge(doc.indexed_status)}
                            <span style={{ color: 'var(--text-tertiary)' }}>
                              {doc.chunks_count} {doc.chunks_count === 1 ? 'chunk' : 'chunks'}
                            </span>
                            {doc.page_count && (
                              <span style={{ color: 'var(--text-tertiary)' }}>
                                · {doc.page_count} {doc.page_count === 1 ? 'pg' : 'pgs'}
                              </span>
                            )}
                          </div>

                          {doc.units_detected && doc.units_detected.length > 0 && (
                            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginTop: '2px' }}>
                              {doc.units_detected.map((u) => (
                                <span key={u} className="unit-tag">
                                  {u}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════════════════
              TAB 2: COLLECTIONS OVERVIEW VIEW
              ════════════════════════════════════════════════════════════ */}
          {activeTab === 'collections' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="dashboard-grid">
                <div className="stat-card">
                  <div className="stat-card-header">
                    <span className="stat-card-label">Total Corpus Documents</span>
                    <div className="stat-card-icon">DOC</div>
                  </div>
                  <div className="stat-card-value">{kmData?.root?.total_documents || 0}</div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    Authoritative on-premise knowledge
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-header">
                    <span className="stat-card-label">Vector Store Chunks</span>
                    <div className="stat-card-icon" style={{ background: 'var(--tech-blue-light)', color: 'var(--tech-blue)' }}>VEC</div>
                  </div>
                  <div className="stat-card-value">{(kmData?.root?.total_chunks || 0).toLocaleString()}</div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    ChromaDB collection: <code>mrpl_knowledge_base</code>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-header">
                    <span className="stat-card-label">Clearance Scope</span>
                    <div className="stat-card-icon" style={{ background: 'var(--brand-orange-light)', color: 'var(--brand-orange)' }}>SEC</div>
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--brand-olive)', marginTop: '4px' }}>
                    {kmData?.root?.user_clearance || 'INTERNAL'}
                  </div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    Level {kmData?.root?.max_clearance_level || 2} · RBAC Enforced
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-card-header">
                    <span className="stat-card-label">Vector Model</span>
                    <div className="stat-card-icon">EMB</div>
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--brand-olive)', marginTop: '4px' }}>
                    nomic-embed-text
                  </div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                    Local 768-dim · Zero Data Leakage
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '16px' }}>
                {(kmData?.categories || []).map((cat) => (
                  <div key={cat.category_code} className="card" style={{ padding: '18px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                      <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--brand-olive)' }}>{cat.name}</h3>
                      {renderClassificationBadge(cat.classification)}
                    </div>

                    <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '14px', flex: 1 }}>
                      {cat.description}
                    </p>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)' }}>
                      <span style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)' }}>
                        <strong>{cat.document_count}</strong> {cat.document_count === 1 ? 'document' : 'documents'} ·{' '}
                        <strong>{cat.total_chunks.toLocaleString()}</strong> chunks
                      </span>

                      <div className="flex gap-6">
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => {
                            setFilterCategory(cat.category_code);
                            setActiveTab('documents');
                          }}
                          style={{ fontSize: '0.72rem' }}
                        >
                          View Docs
                        </button>
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => {
                            setSelectedCategoryCode(cat.category_code);
                            setFilterCategory(cat.category_code);
                            setActiveTab('map');
                          }}
                          style={{ fontSize: '0.72rem' }}
                        >
                          Open in Map
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════════════════
              TAB 3: AUTHORITATIVE DOCUMENTS TABLE VIEW
              ════════════════════════════════════════════════════════════ */}
          {activeTab === 'documents' && (
            <div className="card" style={{ overflow: 'hidden' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Document Name & Title</th>
                    <th>Classification</th>
                    <th>Collection / Category</th>
                    <th>Chunks</th>
                    <th>Pages / Format</th>
                    <th>Process Units</th>
                    <th>Index Status</th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDocs.map((doc) => {
                    const uploadedMatch = uploadedDocs.find(
                      (u) => u.id === doc.file_id || u.original_name === doc.filename
                    );
                    return (
                      <tr key={doc.id}>
                        <td>
                          <div
                            style={{
                              fontWeight: 600,
                              color: 'var(--text-primary)',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                            }}
                            onClick={() => setInspectedDoc(doc)}
                            title="Click to inspect document metadata"
                          >
                            <span aria-hidden="true">📄</span>
                            <span>{doc.title}</span>
                          </div>
                          <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                            {doc.filename} {doc.year ? `· FY ${doc.year}` : ''}
                          </div>
                        </td>
                        <td>{renderClassificationBadge(doc.classification)}</td>
                        <td>
                          <span className="badge badge-neutral" style={{ fontSize: '0.68rem' }}>
                            {doc.category_name || doc.category}
                          </span>
                        </td>
                        <td>
                          <span style={{ fontWeight: 600, color: 'var(--brand-olive)' }}>
                            {doc.chunks_count}
                          </span>
                        </td>
                        <td>
                          <div className="flex gap-4 items-center">
                            {renderTypeBadge(doc.document_type)}
                            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                              {doc.page_count ? `${doc.page_count} pgs` : '1 pg'}
                            </span>
                          </div>
                        </td>
                        <td>
                          {doc.units_detected && doc.units_detected.length > 0 ? (
                            <div className="flex gap-4" style={{ flexWrap: 'wrap' }}>
                              {doc.units_detected.map((u) => (
                                <span key={u} className="unit-tag">
                                  {u}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem' }}>—</span>
                          )}
                        </td>
                        <td>{renderStatusBadge(doc.indexed_status)}</td>
                        <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                          <div className="flex gap-6" style={{ justifyContent: 'flex-end' }}>
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => setInspectedDoc(doc)}
                              title="Inspect node metadata"
                              style={{ fontSize: '0.7rem' }}
                            >
                              Inspect
                            </button>

                            {uploadedMatch && (
                              <>
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => handleReindex(uploadedMatch.id)}
                                  title="Re-run RAG chunking and vector indexing"
                                  style={{ fontSize: '0.7rem' }}
                                >
                                  Re-index
                                </button>
                                <button
                                  className="btn btn-danger btn-sm"
                                  onClick={() => handleDelete(uploadedMatch.id, uploadedMatch.original_name)}
                                  title="Delete from knowledge base"
                                  style={{ fontSize: '0.7rem' }}
                                >
                                  Delete
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* ── Document Details Inspector Drawer (Right Side) ─────────── */}
      {inspectedDoc && (
        <div className="km-inspector-overlay" onClick={() => setInspectedDoc(null)}>
          <div className="km-inspector-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="km-inspector-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '1.1rem' }}>📄</span>
                <div>
                  <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--brand-olive)' }}>
                    Document Node Inspector
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
                    Authoritative metadata from ChromaDB vector store
                  </div>
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setInspectedDoc(null)}
                aria-label="Close document inspector"
              >
                ✕
              </button>
            </div>

            <div className="km-inspector-body">
              <div className="km-field-group">
                <span className="km-field-label">Document Title</span>
                <span className="km-field-value" style={{ fontWeight: 600 }}>
                  {inspectedDoc.title}
                </span>
              </div>

              <div className="km-field-group">
                <span className="km-field-label">Source Filename</span>
                <span className="km-field-value font-mono" style={{ fontSize: '0.76rem', wordBreak: 'break-all' }}>
                  {inspectedDoc.filename}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="km-field-group">
                  <span className="km-field-label">Data Classification</span>
                  <div>{renderClassificationBadge(inspectedDoc.classification)}</div>
                </div>

                <div className="km-field-group">
                  <span className="km-field-label">Access Clearance Status</span>
                  <div>
                    <span className="badge badge-success">
                      ✓ ACCESSIBLE
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="km-field-group">
                  <span className="km-field-label">Collection / Category</span>
                  <span className="km-field-value" style={{ fontSize: '0.8rem' }}>
                    {inspectedDoc.category_name || inspectedDoc.category}
                  </span>
                </div>

                <div className="km-field-group">
                  <span className="km-field-label">Department</span>
                  <span className="km-field-value" style={{ fontSize: '0.8rem' }}>
                    {inspectedDoc.department || 'GENERAL'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="km-field-group">
                  <span className="km-field-label">ChromaDB Chunks</span>
                  <span className="km-field-value" style={{ fontWeight: 700, color: 'var(--brand-olive)' }}>
                    {inspectedDoc.chunks_count} chunks
                  </span>
                </div>

                <div className="km-field-group">
                  <span className="km-field-label">Page Count</span>
                  <span className="km-field-value">
                    {inspectedDoc.page_count ? `${inspectedDoc.page_count} pages` : '1 page'}
                  </span>
                </div>
              </div>

              {inspectedDoc.units_detected && inspectedDoc.units_detected.length > 0 && (
                <div className="km-field-group">
                  <span className="km-field-label">Detected Process Units</span>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '4px' }}>
                    {inspectedDoc.units_detected.map((unit) => (
                      <span key={unit} className="unit-tag" style={{ fontSize: '0.74rem', padding: '2px 8px' }}>
                        {unit}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {inspectedDoc.year && (
                <div className="km-field-group">
                  <span className="km-field-label">Fiscal / Reporting Year</span>
                  <span className="km-field-value font-mono">
                    FY {inspectedDoc.year}
                  </span>
                </div>
              )}

              <div className="km-field-group" style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-default)' }}>
                <span className="km-field-label">RAG Retrieval Pipeline</span>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.4 }}>
                  ✓ Vectorized with <code>nomic-embed-text</code> (768 dimensions)
                  <br />
                  ✓ Indexed for BM25 hybrid lexical retrieval
                  <br />
                  ✓ Filtered by caller RBAC clearance at retrieval time
                </div>
              </div>

              {inspectedDoc.file_id && (
                <div style={{ display: 'flex', gap: '8px', marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleReindex(inspectedDoc.file_id)}
                    style={{ flex: 1 }}
                  >
                    Re-index Chunks
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDelete(inspectedDoc.file_id, inspectedDoc.filename)}
                    style={{ flex: 1 }}
                  >
                    Delete File
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Interactive Progressive Knowledge Explorer Overlay ─── */}
      <KnowledgeExplorerOverlay
        isOpen={showExplorer}
        onClose={() => setShowExplorer(false)}
        kmData={kmData}
        userClearance={localStorage.getItem('mrpl_clearance') || 'CONFIDENTIAL'}
        onSelectDoc={(doc) => {
          setActiveTab('documents');
          setSearchTerm(doc.title || doc.filename);
          setFilterCategory('ALL');
          setFilterClassification('ALL');
          setFilterDocType('ALL');
          setFilterUnit('ALL');
        }}
      />
    </div>
  );
}
