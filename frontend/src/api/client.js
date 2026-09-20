/**
 * API client for the MRPL Sovereign AI Workbench backend.
 * Single source of truth for all HTTP calls.
 */

const API_BASE = '';

function getToken() {
  return localStorage.getItem('mrpl_token');
}

async function request(endpoint, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...options.headers,
  };

  // Remove Content-Type for FormData
  if (options.body instanceof FormData) {
    delete headers['Content-Type'];
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
    credentials: 'include',
  });

  if (response.status === 401) {
    localStorage.removeItem('mrpl_token');
    localStorage.removeItem('mrpl_user');
    window.location.href = '/login';
    throw new Error('Session expired');
  }

  if (response.status === 204) {
    return null;
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// ── Auth ────────────────────────────────────────────────────────────────

export const auth = {
  login: (username, password) =>
    request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),

  logout: () =>
    request('/api/auth/logout', { method: 'POST' }),

  me: () =>
    request('/api/auth/me'),
};

// ── Chat ────────────────────────────────────────────────────────────────

export const chat = {
  createSession: (title = 'New Chat') =>
    request('/api/chat/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),

  listSessions: () =>
    request('/api/chat/sessions'),

  getSession: (id) =>
    request(`/api/chat/sessions/${id}`),

  sendMessage: (sessionId, content) =>
    request(`/api/chat/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    }),

  renameSession: (id, title) =>
    request(`/api/chat/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  deleteSession: (id) =>
    request(`/api/chat/sessions/${id}`, { method: 'DELETE' }),

  analyzeImage: (sessionId, file, query = '') => {
    const formData = new FormData();
    formData.append('file', file);
    if (query) {
      formData.append('query', query);
    }
    return request(`/api/chat/sessions/${sessionId}/image-analysis`, {
      method: 'POST',
      body: formData,
    });
  },
};

// ── Agent ───────────────────────────────────────────────────────────────

export const agent = {
  invoke: (query, sessionId = null) =>
    request('/api/agent/invoke', {
      method: 'POST',
      body: JSON.stringify({ query, session_id: sessionId }),
    }),

  getTask: (taskId) =>
    request(`/api/agent/tasks/${taskId}`),
};

// ── Documents ───────────────────────────────────────────────────────────

export const documents = {
  upload: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/api/documents/upload', {
      method: 'POST',
      body: formData,
    });
  },

  list: (source = null) =>
    request(source ? `/api/documents?source=${encodeURIComponent(source)}` : '/api/documents'),

  getKnowledgeMap: () =>
    request('/api/documents/knowledge-map'),

  getCorpus: () =>
    request('/api/documents/corpus'),

  delete: (id) =>
    request(`/api/documents/${id}`, { method: 'DELETE' }),

  reindex: (id) =>
    request(`/api/documents/${id}/reindex`, { method: 'POST' }),

  downloadUrl: (id) => `${API_BASE}/api/documents/${id}/download`,

  listGenerated: () =>
    request('/api/documents/generated/list'),

  downloadGeneratedUrl: (id) => `${API_BASE}/api/documents/generated/${id}/download`,
};

// ── RAG ─────────────────────────────────────────────────────────────────

export const rag = {
  query: (query, topK = null) =>
    request('/api/rag/query', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),
};

// ── Models ──────────────────────────────────────────────────────────────

export const models = {
  list: () =>
    request('/api/models'),

  get: (id) =>
    request(`/api/models/${id}`),

  toggle: (id, enabled) =>
    request(`/api/models/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    }),

  status: (id) =>
    request(`/api/models/${id}/status`),
};

// ── Admin ───────────────────────────────────────────────────────────────

export const admin = {
  listUsers: () =>
    request('/api/admin/users'),

  createUser: (data) =>
    request('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateUser: (id, data) =>
    request(`/api/admin/users/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  deactivateUser: (id) =>
    request(`/api/admin/users/${id}`, { method: 'DELETE' }),
};

// ── Audit ───────────────────────────────────────────────────────────────

export const audit = {
  getLogs: (params = {}) => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        searchParams.append(key, value);
      }
    });
    return request(`/api/audit/logs?${searchParams.toString()}`);
  },
};

// ── Status ──────────────────────────────────────────────────────────────

export const status = {
  system: () => request('/api/status/system'),
  network: () => request('/api/status/network'),
  services: () => request('/api/status/services'),
  features: () => request('/api/status/features'),
  security: () => request('/api/status/security'),
  seal: () => request('/api/status/seal'),
  diagnostics: () => request('/api/status/diagnostics'),
};

// ── Approvals (Human-in-the-Loop) ───────────────────────────────────────

export const approvals = {
  listPending: () => request('/api/approvals/pending'),
  approve: (id, reason = '') =>
    request(`/api/approvals/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  reject: (id, reason = '') =>
    request(`/api/approvals/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  propose: (data) =>
    request('/api/approvals/propose', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};

// ── Sandbox ─────────────────────────────────────────────────────────────

export const sandbox = {
  execute: (code, language = 'python', timeoutSeconds = 30) =>
    request('/api/sandbox/execute', {
      method: 'POST',
      body: JSON.stringify({ code, language, timeout_seconds: timeoutSeconds }),
    }),
};

// ── Voice ───────────────────────────────────────────────────────────────

export const voice = {
  getStatus: () => request('/api/voice/status'),

  transcribe: (audioBlob, language = null) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'speech.webm');
    if (language && language !== 'auto') {
      formData.append('language', language);
    }
    return request('/api/voice/transcribe', {
      method: 'POST',
      body: formData,
    });
  },

  process: (audioBlob, sessionId = null, language = null, confirmedQuery = null, clarificationContext = null) => {
    const formData = new FormData();
    if (audioBlob) {
      formData.append('file', audioBlob, 'speech.webm');
    }
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    if (language && language !== 'auto') {
      formData.append('language', language);
    }
    if (confirmedQuery) {
      formData.append('confirmed_query', confirmedQuery);
    }
    if (clarificationContext) {
      formData.append(
        'clarification_context',
        typeof clarificationContext === 'string'
          ? clarificationContext
          : JSON.stringify(clarificationContext)
      );
    }
    return request('/api/voice/process', {
      method: 'POST',
      body: formData,
    });
  },

  tts: (text, language = 'en') =>
    request('/api/voice/tts', {
      method: 'POST',
      body: JSON.stringify({ text, language }),
    }),

  partialTranscribe: (audioBlob, language = null, signal = null) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'speech_chunk.webm');
    if (language && language !== 'auto') {
      formData.append('language', language);
    }
    return request('/api/voice/partial_transcribe', {
      method: 'POST',
      body: formData,
      signal,
    });
  },

  greeting: (language = 'en') => {
    return request(`/api/voice/greeting?language=${encodeURIComponent(language || 'en')}`);
  },

  interrupt: (sessionId = null) => {
    const formData = new FormData();
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    return request('/api/voice/interrupt', {
      method: 'POST',
      body: formData,
    });
  },

  processStream: async (
    audioBlob,
    sessionId = null,
    language = null,
    confirmedQuery = null,
    clarificationContext = null,
    onEvent = null,
    signal = null,
    turnId = null
  ) => {
    const formData = new FormData();
    if (audioBlob) {
      formData.append('file', audioBlob, 'speech.webm');
    }
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    if (language && language !== 'auto') {
      formData.append('language', language);
    }
    if (confirmedQuery) {
      formData.append('confirmed_query', confirmedQuery);
    }
    if (turnId !== null && turnId !== undefined) {
      formData.append('turn_id', turnId);
    }
    if (clarificationContext) {
      formData.append(
        'clarification_context',
        typeof clarificationContext === 'string'
          ? clarificationContext
          : JSON.stringify(clarificationContext)
      );
    }

    const token = localStorage.getItem('token');
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch('/api/voice/process_stream', {
      method: 'POST',
      headers,
      body: formData,
      signal,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errJson.detail || 'Voice streaming request failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop(); // keep remainder
      for (const part of parts) {
        const trimmed = part.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            if (onEvent) onEvent(data);
          } catch (e) {
            console.warn('Could not parse SSE event:', trimmed, e);
          }
        }
      }
    }

    if (buffer && buffer.trim().startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.trim().slice(6));
        if (onEvent) onEvent(data);
      } catch (e) {
        // ignore
      }
    }
  },
};

// ── Health ──────────────────────────────────────────────────────────────

export const health = {
  check: () => request('/health'),
};

