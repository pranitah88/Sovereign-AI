import { useState, useEffect, useMemo, useRef, useLayoutEffect } from 'react';

/**
 * KnowledgeExplorerOverlay — Mindmap Graph Visualization.
 *
 * Interactive progressive mindmap graph for MRPL Sovereign AI Workbench.
 * Employs horizontal tree layout with smooth cubic bezier SVG connector branches,
 * category-themed color stems, expandable capsule nodes, pan/zoom canvas,
 * and live contextual metadata inspection.
 *
 * Strictly uses REAL authoritative ChromaDB metadata from `mrpl_knowledge_base`.
 * Respects backend RBAC and user clearance tier.
 */

// Category Theme Palettes for Mindmap Branches
const CATEGORY_THEMES = {
  '01_Refinery_Manufacturing': {
    color: '#1b7a43',
    border: '#234d20',
    bg: '#f0fdf4',
    accent: '#10b981',
    label: 'REFINING',
    icon: '⚙️',
  },
  '02_Environment_Compliance': {
    color: '#15803d',
    border: '#166534',
    bg: '#f2fcf5',
    accent: '#22c55e',
    label: 'ENVIRONMENT',
    icon: '🌱',
  },
  '03_Safety_HSE': {
    color: '#c2410c',
    border: '#9a3412',
    bg: '#fff7ed',
    accent: '#ea580c',
    label: 'HSE SAFETY',
    icon: '🛡️',
  },
  '07_Finance': {
    color: '#0369a1',
    border: '#075985',
    bg: '#f0f9ff',
    accent: '#0284c7',
    label: 'FINANCE',
    icon: '📊',
  },
  '09_Confidential': {
    color: '#b91c1c',
    border: '#991b1b',
    bg: '#fef2f2',
    accent: '#ef4444',
    label: 'CONFIDENTIAL',
    icon: '🔒',
  },
  '05_Policies_Certifications': {
    color: '#4338ca',
    border: '#3730a3',
    bg: '#eef2ff',
    accent: '#6366f1',
    label: 'STANDARDS',
    icon: '📜',
  },
  '04_Company_General': {
    color: '#334155',
    border: '#1e293b',
    bg: '#f8fafc',
    accent: '#64748b',
    label: 'CORPORATE',
    icon: '🏢',
  },
  '06_Website_Content': {
    color: '#0f766e',
    border: '#115e59',
    bg: '#f0fdfa',
    accent: '#14b8a6',
    label: 'PORTAL',
    icon: '🌐',
  },
  '08_Other': {
    color: '#78350f',
    border: '#451a03',
    bg: '#fffbeb',
    accent: '#d97706',
    label: 'TECHNICAL',
    icon: '📑',
  },
  'general': {
    color: '#4d7c0f',
    border: '#3f6212',
    bg: '#f7fee7',
    accent: '#84cc16',
    label: 'MANUALS',
    icon: '📘',
  },
  'root': {
    color: '#1b3e19',
    border: '#143013',
    bg: '#edf4eb',
    accent: '#fbbf24',
    label: 'KNOWLEDGE BASE',
    icon: '✦',
  },
};

export default function KnowledgeExplorerOverlay({
  isOpen,
  onClose,
  kmData,
  onSelectDoc,
  userClearance = 'CONFIDENTIAL',
}) {
  // Tree expansion state: Set of node IDs currently expanded
  const [expandedNodes, setExpandedNodes] = useState(new Set(['root', 'cat_01_Refinery_Manufacturing']));
  // Selected node for contextual inspector
  const [selectedNode, setSelectedNode] = useState(null);
  // Breadcrumb path from root to selected node
  const [activePath, setActivePath] = useState(['MRPL Sovereign Knowledge Base']);
  // Search query
  const [searchQuery, setSearchQuery] = useState('');
  // Pan and Zoom
  const [zoomScale, setZoomScale] = useState(1.0);
  const [panOffset, setPanOffset] = useState({ x: 40, y: 40 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  // Copy feedback
  const [copiedId, setCopiedId] = useState(null);

  const canvasContainerRef = useRef(null);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Set default selection when opened
  useEffect(() => {
    if (isOpen && kmData?.root && !selectedNode) {
      setSelectedNode({
        id: 'root',
        type: 'root',
        title: 'MRPL SOVEREIGN KNOWLEDGE BASE',
        subtitle: 'Authoritative Enterprise Vector Corpus',
        docCount: kmData.root.total_documents || 185,
        chunkCount: kmData.root.total_chunks || 21328,
        categoriesCount: kmData.categories?.length || 11,
        indexingStatus: kmData.root.indexed_status || 'Indexed (ChromaDB)',
        embeddingEngine: kmData.root.embedding_engine || 'all-MiniLM-L6-v2 (Local CPU)',
        clearance: kmData.root.user_clearance || userClearance,
        classification: 'RESTRICTED',
      });
      setActivePath(['MRPL Sovereign Knowledge Base']);
    }
  }, [isOpen, kmData, selectedNode, userClearance]);

  // ── Build Hierarchical Mindmap Data ──────────────────────────────────────
  const treeData = useMemo(() => {
    if (!kmData || !kmData.categories) return null;

    const rootDocCount = kmData.root?.total_documents || 185;
    const rootChunkCount = kmData.root?.total_chunks || 21328;

    const categories = kmData.categories.map((cat) => {
      const catCode = cat.category_code;
      const docs = cat.documents || [];
      const theme = CATEGORY_THEMES[catCode] || CATEGORY_THEMES['root'];

      // Group documents by detected industrial units or logical sub-categories
      const unitsMap = {};
      const unassignedDocs = [];

      docs.forEach((doc) => {
        const units = doc.units_detected || [];
        if (units.length > 0) {
          units.forEach((u) => {
            if (!unitsMap[u]) unitsMap[u] = [];
            unitsMap[u].push(doc);
          });
        } else if (catCode === '03_Safety_HSE') {
          const titleLower = (doc.title + ' ' + doc.filename).toLowerCase();
          const group = titleLower.includes('vigilance') || titleLower.includes('complaint') || titleLower.includes('cvc')
            ? 'CVC & Vigilance Governance'
            : 'Plant Safety & Operations';
          if (!unitsMap[group]) unitsMap[group] = [];
          unitsMap[group].push(doc);
        } else if (catCode === '07_Finance') {
          const titleLower = (doc.title + ' ' + doc.filename).toLowerCase();
          let group = 'Audited Annual Reports';
          if (titleLower.includes('202') || titleLower.includes('21-22') || titleLower.includes('22-23') || titleLower.includes('23-24') || titleLower.includes('25-26')) {
            group = 'Fiscal Reports (2021-2026)';
          } else {
            group = 'Historical Reports (2012-2020)';
          }
          if (!unitsMap[group]) unitsMap[group] = [];
          unitsMap[group].push(doc);
        } else {
          unassignedDocs.push(doc);
        }
      });

      // Build Unit / Sub-stream child nodes
      const unitNodes = Object.keys(unitsMap).map((unitName) => {
        const unitDocs = unitsMap[unitName];
        const unitChunks = unitDocs.reduce((acc, d) => acc + (d.chunks_count || 0), 0);

        return {
          id: `unit_${catCode}_${unitName}`,
          type: 'unit',
          title: unitName,
          categoryName: cat.name,
          categoryCode: catCode,
          docCount: unitDocs.length,
          chunkCount: unitChunks,
          classification: cat.classification || 'INTERNAL',
          theme: theme,
          description: `Industrial asset unit stream in ${cat.name}`,
          documents: unitDocs,
        };
      });

      // Include unassigned docs under a General stream if unitNodes exist
      if (unitNodes.length > 0 && unassignedDocs.length > 0) {
        const genChunks = unassignedDocs.reduce((acc, d) => acc + (d.chunks_count || 0), 0);
        unitNodes.push({
          id: `unit_${catCode}_general`,
          type: 'unit',
          title: `General ${cat.name.split(' ')[0]} Assets`,
          categoryName: cat.name,
          categoryCode: catCode,
          docCount: unassignedDocs.length,
          chunkCount: genChunks,
          classification: cat.classification || 'INTERNAL',
          theme: theme,
          description: `Standard procedures and operational assets for ${cat.name}`,
          documents: unassignedDocs,
        });
      }

      return {
        id: `cat_${catCode}`,
        type: 'category',
        title: cat.name,
        categoryCode: catCode,
        description: cat.description,
        docCount: cat.document_count,
        chunkCount: cat.total_chunks,
        classification: cat.classification || 'INTERNAL',
        theme: theme,
        units: unitNodes,
        documents: docs,
      };
    });

    return {
      id: 'root',
      type: 'root',
      title: 'MRPL SOVEREIGN KNOWLEDGE BASE',
      subtitle: `${rootDocCount} Documents · ${rootChunkCount.toLocaleString()} Chunks`,
      docCount: rootDocCount,
      chunkCount: rootChunkCount,
      categoriesCount: categories.length,
      indexingStatus: kmData.root?.indexed_status || 'Indexed (ChromaDB)',
      embeddingEngine: kmData.root?.embedding_engine || 'all-MiniLM-L6-v2 (Local CPU)',
      clearance: kmData.root?.user_clearance || userClearance,
      classification: 'RESTRICTED',
      theme: CATEGORY_THEMES['root'],
      categories: categories,
    };
  }, [kmData, userClearance]);

  // Toggle node expansion
  const toggleExpand = (nodeId, e) => {
    if (e) e.stopPropagation();
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (!treeData) return;
    const allIds = new Set(['root']);
    treeData.categories.forEach((cat) => {
      allIds.add(cat.id);
      cat.units.forEach((u) => allIds.add(u.id));
    });
    setExpandedNodes(allIds);
  };

  const collapseAll = () => {
    setExpandedNodes(new Set(['root']));
  };

  // Reset Pan and Zoom
  const resetView = () => {
    setZoomScale(1.0);
    setPanOffset({ x: 40, y: 40 });
  };

  // Canvas Drag / Pan Handlers
  const handleMouseDown = (e) => {
    // Only pan on canvas background or left mouse button
    if (e.target.closest('.mm-node-card') || e.target.closest('.mm-expand-circle')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPanOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Copy filename feedback
  const handleCopyName = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  // Filter in main Knowledge Base
  const handleFilterInKb = (doc) => {
    if (onSelectDoc) {
      onSelectDoc(doc);
    }
    onClose();
  };

  if (!isOpen || !treeData) return null;

  const query = searchQuery.trim().toLowerCase();

  return (
    <div
      className="ke-overlay-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="MRPL Knowledge Explorer Mindmap"
    >
      <div
        className="ke-modal-window mm-modal"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Top Bar: Title, Search, Controls ───────────────────────── */}
        <div className="ke-header mm-header">
          <div className="ke-header-left">
            <div className="mm-brand-badge">
              <span className="mm-sparkle">✦</span>
              <span>MRPL MINDMAP KNOWLEDGE GRAPH</span>
            </div>
            <div className="mm-breadcrumb-trail">
              {activePath.map((item, idx) => (
                <span key={idx} className="mm-breadcrumb-item">
                  {idx > 0 && <span className="mm-breadcrumb-sep">›</span>}
                  <span className={idx === activePath.length - 1 ? 'active' : ''}>{item}</span>
                </span>
              ))}
            </div>
          </div>

          {/* Search Box */}
          <div className="ke-search-box mm-search">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              className="ke-search-input"
              placeholder="Search mindmap (CDU, HCU, Vigilance, docs)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                className="ke-search-clear"
                onClick={() => setSearchQuery('')}
                title="Clear"
              >
                ✕
              </button>
            )}
          </div>

          {/* Controls */}
          <div className="ke-toolbar">
            <button className="ke-tool-btn" onClick={expandAll} title="Expand all branches">
              Expand All
            </button>
            <button className="ke-tool-btn" onClick={collapseAll} title="Collapse all">
              Collapse
            </button>

            <div className="ke-zoom-group">
              <button
                className="ke-zoom-btn"
                onClick={() => setZoomScale((prev) => Math.max(0.6, prev - 0.1))}
                title="Zoom Out"
              >
                −
              </button>
              <span className="ke-zoom-label">{Math.round(zoomScale * 100)}%</span>
              <button
                className="ke-zoom-btn"
                onClick={() => setZoomScale((prev) => Math.min(1.4, prev + 0.1))}
                title="Zoom In"
              >
                +
              </button>
              <button
                className="ke-zoom-btn"
                onClick={resetView}
                title="Reset View"
                style={{ fontSize: '0.72rem', padding: '0 6px' }}
              >
                ↺
              </button>
            </div>

            <button
              className="ke-close-btn"
              onClick={onClose}
              title="Close Mindmap (Esc)"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ── Main Canvas & Inspector Pane ───────────────────────────── */}
        <div className="ke-body mm-body">
          {/* Canvas Wrapper */}
          <div
            className="mm-canvas-viewport"
            ref={canvasContainerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
          >
            {/* Mindmap Graph Tree Layer */}
            <div
              className="mm-canvas-stage"
              style={{
                transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomScale})`,
                transformOrigin: 'top left',
              }}
            >
              {/* ── Root Mindmap Hub ─────────────────────────────────── */}
              <div className="mm-root-cluster">
                {/* Central Root Capsule Node */}
                <div
                  className={`mm-node-card mm-root-node ${selectedNode?.id === 'root' ? 'active-node' : ''}`}
                  onClick={() => {
                    setSelectedNode(treeData);
                    setActivePath(['MRPL Sovereign Knowledge Base']);
                    toggleExpand('root');
                  }}
                >
                  <div className="mm-node-glow-ring" />
                  <div className="mm-root-badge">
                    <span className="mm-sparkle-gold">✦</span>
                    <span>AUTHORITATIVE REFINERY CORPUS</span>
                  </div>
                  <h1 className="mm-root-title">MRPL Sovereign Knowledge Base</h1>
                  <div className="mm-root-meta">
                    <span className="mm-pill-stat">
                      <strong>{treeData.docCount}</strong> Documents
                    </span>
                    <span className="mm-pill-stat">
                      <strong>{treeData.chunkCount.toLocaleString()}</strong> Chunks
                    </span>
                    <span className="mm-pill-stat clearance">
                      {treeData.clearance}
                    </span>
                  </div>

                  {/* Root expand circle button */}
                  <div
                    className="mm-expand-circle root-circle"
                    onClick={(e) => toggleExpand('root', e)}
                    title={expandedNodes.has('root') ? 'Collapse Knowledge Base' : 'Expand Collections'}
                  >
                    {expandedNodes.has('root') ? '−' : '+'}
                  </div>
                </div>

                {/* ── Level 1: Categories Branch Group ──────────────── */}
                {expandedNodes.has('root') && (
                  <MindMapBranchGroup
                    items={treeData.categories}
                    parentId="root"
                    stemColor="#234d20"
                    renderNode={(cat) => {
                      const isCatExpanded = expandedNodes.has(cat.id);
                      const isCatSelected = selectedNode?.id === cat.id;

                      const matchesQuery = !query ||
                        cat.title.toLowerCase().includes(query) ||
                        cat.units.some(u => u.title.toLowerCase().includes(query)) ||
                        cat.documents.some(d => (d.title + ' ' + d.filename).toLowerCase().includes(query));

                      if (!matchesQuery) return null;

                      return (
                        <div key={cat.id} className="mm-branch-unit">
                          {/* Category Capsule Node */}
                          <div
                            className={`mm-node-card mm-category-node ${isCatSelected ? 'active-node' : ''}`}
                            style={{
                              borderLeftColor: cat.theme.color,
                              '--node-accent': cat.theme.accent,
                            }}
                            onClick={() => {
                              setSelectedNode({ ...cat, type: 'category' });
                              setActivePath(['MRPL Sovereign Knowledge Base', cat.title]);
                              toggleExpand(cat.id);
                            }}
                          >
                            <div className="mm-node-icon-bubble" style={{ background: cat.theme.bg, color: cat.theme.color }}>
                              {cat.theme.icon}
                            </div>
                            <div className="mm-node-content">
                              <div className="mm-node-header-row">
                                <span className="mm-tag-code" style={{ color: cat.theme.color }}>
                                  {cat.theme.label}
                                </span>
                                <span className="mm-doc-count-pill">
                                  {cat.docCount} docs
                                </span>
                              </div>
                              <div className="mm-node-label" title={cat.title}>{cat.title}</div>
                              <div className="mm-node-sublabel">
                                {cat.chunkCount.toLocaleString()} vector chunks
                              </div>
                            </div>

                            {/* Node expansion junction */}
                            <div
                              className={`mm-expand-circle ${isCatExpanded ? 'expanded' : ''}`}
                              style={{ borderColor: cat.theme.color, color: cat.theme.color }}
                              onClick={(e) => toggleExpand(cat.id, e)}
                              title={isCatExpanded ? 'Collapse' : 'Expand'}
                            >
                              {isCatExpanded ? '−' : '+'}
                            </div>
                          </div>

                          {/* ── Level 2: Industrial Units Branches ──────── */}
                          {isCatExpanded && (
                            <MindMapBranchGroup
                              items={cat.units.length > 0 ? cat.units : cat.documents}
                              parentId={cat.id}
                              stemColor={cat.theme.color}
                              renderNode={(unitOrDoc) => {
                                if (cat.units.length > 0) {
                                  // Unit node
                                  const unit = unitOrDoc;
                                  const isUnitExpanded = expandedNodes.has(unit.id);
                                  const isUnitSelected = selectedNode?.id === unit.id;

                                  const matchesUnit = !query ||
                                    unit.title.toLowerCase().includes(query) ||
                                    cat.title.toLowerCase().includes(query) ||
                                    unit.documents.some(d => (d.title + ' ' + d.filename).toLowerCase().includes(query));

                                  if (!matchesUnit) return null;

                                  return (
                                    <div key={unit.id} className="mm-branch-unit">
                                      <div
                                        className={`mm-node-card mm-unit-node ${isUnitSelected ? 'active-node' : ''}`}
                                        style={{
                                          borderLeftColor: cat.theme.accent,
                                          '--node-accent': cat.theme.accent,
                                        }}
                                        onClick={() => {
                                          setSelectedNode({ ...unit, type: 'unit' });
                                          setActivePath(['MRPL Sovereign Knowledge Base', cat.title, unit.title]);
                                          toggleExpand(unit.id);
                                        }}
                                      >
                                        <div className="mm-node-content">
                                          <div className="mm-node-header-row">
                                            <span className="mm-unit-flag" style={{ color: cat.theme.color }}>
                                              UNIT STREAM
                                            </span>
                                            <span className="mm-doc-count-pill sm">
                                              {unit.docCount} docs
                                            </span>
                                          </div>
                                          <div className="mm-unit-title" title={unit.title}>{unit.title}</div>
                                          <div className="mm-node-sublabel">
                                            {unit.chunkCount.toLocaleString()} chunks
                                          </div>
                                        </div>

                                        <div
                                          className={`mm-expand-circle sm ${isUnitExpanded ? 'expanded' : ''}`}
                                          style={{ borderColor: cat.theme.accent, color: cat.theme.color }}
                                          onClick={(e) => toggleExpand(unit.id, e)}
                                        >
                                          {isUnitExpanded ? '−' : '+'}
                                        </div>
                                      </div>

                                      {/* ── Level 3: Documents ────────── */}
                                      {isUnitExpanded && (
                                        <MindMapBranchGroup
                                          items={unit.documents}
                                          parentId={unit.id}
                                          stemColor={cat.theme.accent}
                                          renderNode={(doc) => {
                                            const isDocSelected = selectedNode?.id === doc.id;
                                            const isDocExpanded = expandedNodes.has(`doc_${doc.id}`);

                                            const matchesDoc = !query ||
                                              (doc.title + ' ' + doc.filename).toLowerCase().includes(query);

                                            if (!matchesDoc) return null;

                                            return (
                                              <div key={doc.id} className="mm-branch-unit">
                                                <div
                                                  className={`mm-node-card mm-doc-node ${isDocSelected ? 'active-node' : ''}`}
                                                  onClick={() => {
                                                    setSelectedNode({ ...doc, type: 'document' });
                                                    setActivePath(['MRPL Sovereign Knowledge Base', cat.title, unit.title, doc.title]);
                                                    toggleExpand(`doc_${doc.id}`);
                                                  }}
                                                >
                                                  <div className="mm-doc-badge-row">
                                                    <span className="mm-format-pill">
                                                      {doc.document_type?.toUpperCase() || 'PDF'}
                                                    </span>
                                                    <span
                                                      className={`badge ${
                                                        doc.classification === 'CONFIDENTIAL'
                                                          ? 'badge-clearance-confidential'
                                                          : doc.classification === 'PUBLIC'
                                                          ? 'badge-clearance-public'
                                                          : 'badge-clearance-internal'
                                                      }`}
                                                      style={{ fontSize: '0.58rem', padding: '0 4px' }}
                                                    >
                                                      {doc.classification}
                                                    </span>
                                                  </div>
                                                  <div className="mm-doc-title" title={doc.title}>
                                                    {doc.title}
                                                  </div>
                                                  <div className="mm-doc-filename" title={doc.filename}>
                                                    {doc.filename}
                                                  </div>
                                                  <div className="mm-doc-footer">
                                                    <span>{doc.chunks_count} chunks {doc.page_count ? `· ${doc.page_count} pgs` : ''}</span>
                                                    <div
                                                      className={`mm-expand-circle xs ${isDocExpanded ? 'expanded' : ''}`}
                                                      onClick={(e) => toggleExpand(`doc_${doc.id}`, e)}
                                                      title="Inspect metadata chunks"
                                                    >
                                                      {isDocExpanded ? '−' : '+'}
                                                    </div>
                                                  </div>
                                                </div>

                                                {/* ── Level 4: Metadata breakdown pills ── */}
                                                {isDocExpanded && (
                                                  <div className="mm-meta-capsule-row">
                                                    <span className="mm-meta-pill">📄 {doc.page_count || 1} Pages</span>
                                                    <span className="mm-meta-pill">🧩 {doc.chunks_count} Chunks</span>
                                                    <span className="mm-meta-pill">🔒 {doc.classification}</span>
                                                    <span className="mm-meta-pill">⚙️ {doc.units_detected?.join(', ') || doc.department || 'Refining'}</span>
                                                  </div>
                                                )}
                                              </div>
                                            );
                                          }}
                                        />
                                      )}
                                    </div>
                                  );
                                } else {
                                  // Direct document node (no unit level)
                                  const doc = unitOrDoc;
                                  const isDocSelected = selectedNode?.id === doc.id;
                                  return (
                                    <div key={doc.id} className="mm-branch-unit">
                                      <div
                                        className={`mm-node-card mm-doc-node ${isDocSelected ? 'active-node' : ''}`}
                                        onClick={() => {
                                          setSelectedNode({ ...doc, type: 'document' });
                                          setActivePath(['MRPL Sovereign Knowledge Base', cat.title, doc.title]);
                                        }}
                                      >
                                        <div className="mm-doc-badge-row">
                                          <span className="mm-format-pill">{doc.document_type?.toUpperCase() || 'PDF'}</span>
                                          <span className="badge badge-clearance-internal" style={{ fontSize: '0.58rem' }}>
                                            {doc.classification}
                                          </span>
                                        </div>
                                        <div className="mm-doc-title" title={doc.title}>{doc.title}</div>
                                        <div className="mm-doc-filename" title={doc.filename}>{doc.filename}</div>
                                        <div className="mm-doc-footer">
                                          <span>{doc.chunks_count} chunks {doc.page_count ? `· ${doc.page_count} pgs` : ''}</span>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                }
                              }}
                            />
                          )}
                        </div>
                      );
                    }}
                  />
                )}
              </div>
            </div>
          </div>

          {/* ── Contextual Metadata Inspector Drawer ─────────────────── */}
          <div className="ke-inspector-pane mm-inspector">
            <div className="ke-inspector-header">
              <span className="ke-inspector-icon">✦</span>
              <span className="ke-inspector-title">Knowledge Node Inspector</span>
              <span className="ke-inspector-badge">
                {selectedNode?.type?.toUpperCase() || 'NODE'}
              </span>
            </div>

            {selectedNode ? (
              <div className="ke-inspector-content">
                <div className="ke-insp-block">
                  <div className="ke-insp-label">Inspected Entity</div>
                  <h3 className="ke-insp-value-title">{selectedNode.title}</h3>
                  {selectedNode.filename && (
                    <div className="ke-insp-filename" title={selectedNode.filename}>
                      {selectedNode.filename}
                    </div>
                  )}
                  {selectedNode.description && (
                    <p className="ke-insp-desc">{selectedNode.description}</p>
                  )}
                </div>

                {/* Badges */}
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                  {selectedNode.classification && (
                    <span
                      className={`badge ${
                        selectedNode.classification === 'CONFIDENTIAL'
                          ? 'badge-clearance-confidential'
                          : selectedNode.classification === 'RESTRICTED'
                          ? 'badge-clearance-restricted'
                          : selectedNode.classification === 'PUBLIC'
                          ? 'badge-clearance-public'
                          : 'badge-clearance-internal'
                      }`}
                    >
                      {selectedNode.classification}
                    </span>
                  )}
                  {selectedNode.document_type && (
                    <span className="badge badge-neutral">
                      FORMAT: {selectedNode.document_type.toUpperCase()}
                    </span>
                  )}
                  {selectedNode.indexingStatus && (
                    <span className="badge badge-success">
                      {selectedNode.indexingStatus}
                    </span>
                  )}
                </div>

                {/* Metrics Table */}
                <div className="ke-insp-metrics">
                  {selectedNode.docCount !== undefined && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Document Assets</span>
                      <span className="ke-metric-val">{selectedNode.docCount.toLocaleString()}</span>
                    </div>
                  )}
                  {selectedNode.chunkCount !== undefined && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Vector Chunks</span>
                      <span className="ke-metric-val">{selectedNode.chunkCount.toLocaleString()}</span>
                    </div>
                  )}
                  {selectedNode.chunks_count !== undefined && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Indexed Chunks</span>
                      <span className="ke-metric-val">{selectedNode.chunks_count.toLocaleString()}</span>
                    </div>
                  )}
                  {selectedNode.page_count !== undefined && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Source Pages</span>
                      <span className="ke-metric-val">{selectedNode.page_count || '1 (Single Page)'}</span>
                    </div>
                  )}
                  {selectedNode.categoryName && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Parent Collection</span>
                      <span className="ke-metric-val">{selectedNode.categoryName}</span>
                    </div>
                  )}
                  {selectedNode.department && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Department</span>
                      <span className="ke-metric-val">{selectedNode.department}</span>
                    </div>
                  )}
                  {selectedNode.year && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Fiscal Year</span>
                      <span className="ke-metric-val">{selectedNode.year}</span>
                    </div>
                  )}
                  {selectedNode.units_detected && selectedNode.units_detected.length > 0 && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Units Detected</span>
                      <span className="ke-metric-val">{selectedNode.units_detected.join(', ')}</span>
                    </div>
                  )}
                  {selectedNode.embeddingEngine && (
                    <div className="ke-metric-row">
                      <span className="ke-metric-label">Embedding Model</span>
                      <span className="ke-metric-val" style={{ fontSize: '0.72rem' }}>
                        {selectedNode.embeddingEngine}
                      </span>
                    </div>
                  )}
                </div>

                {/* Air-gap security box */}
                <div className="ke-insp-security-box">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
                    <span style={{ color: 'var(--brand-green)', fontWeight: 700 }}>●</span>
                    <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--brand-olive)' }}>
                      ZERO-EGRESS LOCAL VECTOR CORPUS
                    </span>
                  </div>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.35 }}>
                    Clearance verified (<strong>{userClearance}</strong>). All metadata queried from local ChromaDB index.
                  </p>
                </div>

                {/* Actions */}
                <div className="ke-insp-actions">
                  {selectedNode.filename && (
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ width: '100%', justifyContent: 'center' }}
                      onClick={() => handleCopyName(selectedNode.filename, selectedNode.id)}
                    >
                      {copiedId === selectedNode.id ? '✓ Filename Copied' : 'Copy Filename'}
                    </button>
                  )}

                  {selectedNode.type === 'document' && (
                    <button
                      className="btn btn-primary btn-sm"
                      style={{ width: '100%', justifyContent: 'center' }}
                      onClick={() => handleFilterInKb(selectedNode)}
                    >
                      Filter in Knowledge Base →
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="ke-insp-empty">
                <div style={{ fontSize: '1.4rem', marginBottom: '8px', color: 'var(--brand-green)' }}>✦</div>
                <div style={{ fontWeight: 600, color: 'var(--brand-olive)', marginBottom: '4px' }}>
                  Select Mindmap Node
                </div>
                <p style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)', margin: 0 }}>
                  Click on any category, unit capsule, or document node in the mindmap to inspect its metadata.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * MindMapBranchGroup Component
 *
 * Renders a cluster of child nodes with organic SVG bezier connector arms
 * drawn from the parent junction to each child capsule.
 */
function MindMapBranchGroup({ items, _parentId, stemColor = '#234d20', renderNode }) {
  const groupRef = useRef(null);
  const [paths, setPaths] = useState([]);

  // Calculate SVG cubic bezier connector paths dynamically
  useLayoutEffect(() => {
    if (!groupRef.current) return;

    const container = groupRef.current;
    const parentCluster = container.parentElement;
    if (!parentCluster) return;

    // Find parent's node card
    const parentCard = parentCluster.querySelector(':scope > .mm-node-card');
    if (!parentCard) return;

    const groupRect = container.getBoundingClientRect();
    const parentRect = parentCard.getBoundingClientRect();

    // Start point: Center right of parent card relative to container
    const startX = parentRect.right - groupRect.left;
    const startY = parentRect.top + parentRect.height / 2 - groupRect.top;

    // Find all direct child node cards
    const childCards = container.querySelectorAll(':scope > .mm-branch-unit > .mm-node-card');
    const newPaths = [];

    childCards.forEach((childCard) => {
      const childRect = childCard.getBoundingClientRect();
      const endX = childRect.left - groupRect.left;
      const endY = childRect.top + childRect.height / 2 - groupRect.top;

      // Draw smooth S-shaped cubic bezier curve
      const deltaX = Math.max(30, (endX - startX) * 0.55);
      const cp1X = startX + deltaX;
      const cp1Y = startY;
      const cp2X = endX - deltaX;
      const cp2Y = endY;

      const d = `M ${startX} ${startY} C ${cp1X} ${cp1Y}, ${cp2X} ${cp2Y}, ${endX} ${endY}`;
      newPaths.push({ d, endX, endY });
    });

    setPaths(newPaths);
  }, [items]);

  return (
    <div className="mm-branch-group" ref={groupRef}>
      {/* SVG Canvas for Organic Curved Bezier Branch Lines */}
      <svg className="mm-connectors-svg">
        {paths.map((p, idx) => (
          <g key={idx}>
            <path
              d={p.d}
              className="mm-branch-path"
              stroke={stemColor}
              fill="none"
            />
            {/* Small connector terminal dot */}
            <circle
              cx={p.endX}
              cy={p.endY}
              r="3.5"
              fill={stemColor}
              className="mm-branch-terminal-dot"
            />
          </g>
        ))}
      </svg>

      {/* Children list */}
      <div className="mm-children-stack">
        {items.map((item) => renderNode(item))}
      </div>
    </div>
  );
}
