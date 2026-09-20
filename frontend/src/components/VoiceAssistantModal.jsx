import React, { useEffect, useRef } from 'react';
import VoiceOrb from './VoiceOrb';

/**
 * VoiceAssistantModal
 * 
 * Floating, animated conversational voice assistant window.
 * Displays real-time audio visualizer, partial transcription, 
 * pipeline states (Listening, Understanding, Searching, Thinking, Speaking),
 * and barge-in interruption controls.
 */
export default function VoiceAssistantModal({
  isOpen,
  onClose,
  voiceState,
  voiceLanguage,
  onLanguageChange,
  partialTranscript,
  rawTranscript,
  normalizedTranscript,
  isNormalized,
  voiceStatusMsg,
  onStopListening,
  onInterrupt,
  onRetry,
  analyser,
  activeSpokenText,
  activeClarificationPrompt,
}) {
  const canvasRef = useRef(null);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Real-time audio waveform canvas rendering
  useEffect(() => {
    if (!isOpen || !canvasRef.current) return;
    let animationId;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const numBars = 28;

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const hasAnalyser = analyser && (
        voiceState === 'LISTENING' ||
        voiceState === 'WAITING_FOR_USER_REPLY' ||
        voiceState === 'SPEAKING' ||
        voiceState === 'SPEAKING_CLARIFICATION' ||
        voiceState === 'CLARIFYING'
      );

      if (hasAnalyser) {
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        const barWidth = Math.floor((width - (numBars * 3)) / numBars);
        let x = 4;

        for (let i = 0; i < numBars; i++) {
          const index = Math.floor((i / numBars) * (bufferLength * 0.45));
          const value = dataArray[index] || 0;
          const barHeight = Math.max(4, (value / 255) * (height - 12));

          const gradient = ctx.createLinearGradient(0, (height - barHeight) / 2, 0, (height + barHeight) / 2);
          if (voiceState === 'LISTENING' || voiceState === 'WAITING_FOR_USER_REPLY' || voiceState === 'USER_SPEAKING') {
            gradient.addColorStop(0, '#c3d9c0');
            gradient.addColorStop(0.5, '#234d20');
            gradient.addColorStop(1, '#143013');
          } else if (voiceState.includes('CLARIF')) {
            gradient.addColorStop(0, '#e1ede0');
            gradient.addColorStop(0.5, '#3a6634');
            gradient.addColorStop(1, '#1b3e19');
          } else if (voiceState === 'ERROR') {
            gradient.addColorStop(0, '#fcd34d');
            gradient.addColorStop(0.5, '#d96b27');
            gradient.addColorStop(1, '#92400e');
          } else {
            gradient.addColorStop(0, '#e1ede0');
            gradient.addColorStop(0.5, '#2d6129');
            gradient.addColorStop(1, '#143013');
          }

          ctx.fillStyle = gradient;
          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(x, (height - barHeight) / 2, barWidth, barHeight, 3);
          } else {
            ctx.rect(x, (height - barHeight) / 2, barWidth, barHeight);
          }
          ctx.fill();
          x += barWidth + 3;
        }
      } else {
        // Fluid animated breathing wave for idle / processing phases using MRPL Corporate Palette
        const time = Date.now() * 0.004;
        const barWidth = Math.floor((width - (numBars * 3)) / numBars);
        let x = 4;

        for (let i = 0; i < numBars; i++) {
          let barHeight = 4;
          let fillStyle = '#234d20';

          if (voiceState === 'INITIALIZING') {
            barHeight = 6 + Math.sin(time + i * 0.3) * 5 + 4;
            fillStyle = '#d96b27';
          } else if (voiceState === 'USER_SPEAKING') {
            barHeight = 9 + Math.sin(time * 2.0 + i * 0.4) * 8 + 6;
            fillStyle = '#3a6634';
          } else if (voiceState === 'UNDERSTANDING') {
            barHeight = 8 + Math.sin(time * 1.5 + i * 0.4) * 6 + 5;
            fillStyle = '#234d20';
          } else if (voiceState === 'SEARCHING') {
            barHeight = 9 + Math.sin(time * 1.8 + i * 0.35) * 7 + 6;
            fillStyle = '#1b3e19';
          } else if (voiceState === 'THINKING') {
            barHeight = 10 + Math.sin(time * 2.2 + i * 0.5) * 8 + 6;
            fillStyle = '#2c4d28';
          } else if (voiceState === 'SPEAKING' || voiceState === 'SPEAKING_CLARIFICATION' || voiceState === 'CLARIFYING' || voiceState === 'ASKING_CLARIFICATION') {
            barHeight = 12 + Math.sin(time * 2.5 + i * 0.45) * 10 + 8;
            fillStyle = '#2d6129';
          } else if (voiceState === 'ERROR') {
            barHeight = 4;
            fillStyle = '#d96b27';
          }

          ctx.fillStyle = fillStyle;
          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(x, (height - barHeight) / 2, barWidth, barHeight, 3);
          } else {
            ctx.rect(x, (height - barHeight) / 2, barWidth, barHeight);
          }
          ctx.fill();
          x += barWidth + 3;
        }
      }
    };

    draw();
    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [isOpen, analyser, voiceState]);

  if (!isOpen) return null;

  const getStateMeta = () => {
    switch (voiceState) {
      case 'INITIALIZING':
      case 'STARTING':
        return {
          title: 'Starting Voice Session...',
          badgeClass: 'badge-init',
          description: 'Initializing sovereign conversational audio capture.',
          icon: '🎙️',
        };
      case 'GREETING':
        return {
          title: 'Listening Soon...',
          badgeClass: 'badge-speaking',
          description: activeSpokenText || 'Hi! What can I help you with?',
          icon: '👋',
        };
      case 'LISTENING':
        return {
          title: activeClarificationPrompt ? 'Listening for Your Answer...' : 'Listening...',
          badgeClass: 'badge-listening',
          description: activeClarificationPrompt
            ? `Answering: "${activeClarificationPrompt}"`
            : 'Speak your question in English, Hindi, or Marathi.',
          icon: '🔴',
        };
      case 'PROCESSING':
        return {
          title: 'Processing Speech...',
          badgeClass: 'badge-processing',
          description: 'Evaluating acoustic confidence and routing governance.',
          icon: '⚡',
        };
      case 'CLARIFYING':
      case 'ASKING_CLARIFICATION':
        return {
          title: 'Asking Clarification...',
          badgeClass: 'badge-clarifying',
          description: activeClarificationPrompt || 'Assistant asking clarification question.',
          icon: '💬',
        };
      case 'SPEAKING_CLARIFICATION':
        return {
          title: 'Speaking Clarification...',
          badgeClass: 'badge-speaking-clarification',
          description: activeClarificationPrompt || 'Assistant speaking clarification...',
          icon: '🗣️',
        };
      case 'WAITING_FOR_USER_REPLY':
        return {
          title: 'Listening for Reply...',
          badgeClass: 'badge-waiting-reply',
          description: activeClarificationPrompt ? `Clarifying: "${activeClarificationPrompt}"` : 'Listening for your spoken response...',
          icon: '👂',
        };
      case 'UNDERSTANDING':
        return {
          title: 'Understanding Speech...',
          badgeClass: 'badge-understanding',
          description: 'Transcribing speech & checking security clearance.',
          icon: '⚙️',
        };
      case 'SEARCHING':
        return {
          title: 'Searching Knowledge Base...',
          badgeClass: 'badge-searching',
          description: 'Querying technical refinery documents & records.',
          icon: '🔍',
        };
      case 'THINKING':
        return {
          title: 'Thinking...',
          badgeClass: 'badge-thinking',
          description: 'Streaming sovereign local Ollama reasoning.',
          icon: '🧠',
        };
      case 'SPEAKING':
        return {
          title: 'Speaking Response...',
          badgeClass: 'badge-speaking',
          description: 'Playing synthesized natural response.',
          icon: '🔊',
        };
      case 'ERROR':
        return {
          title: 'Microphone Notice',
          badgeClass: 'badge-error',
          description: voiceStatusMsg || 'Microphone capture encountered an issue.',
          icon: '⚠️',
        };
      default:
        return {
          title: 'Voice Assistant Ready',
          badgeClass: 'badge-ready',
          description: 'Click microphone or speak to start.',
          icon: '🎙️',
        };
    }
  };

  const meta = getStateMeta();

  return (
    <div className="voice-modal-backdrop" onClick={onClose}>
      <div className="voice-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="voice-modal-header">
          <div className="voice-modal-brand">
            <span className="voice-brand-logo">MRPL</span>
            <div className="voice-brand-text">
              <h4>Sovereign Voice Assistant</h4>
              <span className="voice-offline-tag">100% On-Premise · Zero Cloud Egress</span>
            </div>
          </div>

          <div className="voice-modal-actions">
            {/* Language Selector */}
            <select
              className="voice-modal-lang-select"
              value={voiceLanguage}
              onChange={(e) => onLanguageChange(e.target.value)}
              disabled={voiceState === 'LISTENING' || voiceState === 'UNDERSTANDING'}
              title="Change recognition language"
            >
              <option value="auto">🌐 Auto-Detect</option>
              <option value="hi">🇮🇳 Hindi</option>
              <option value="mr">🇮🇳 Marathi</option>
              <option value="en">🇬🇧 English</option>
            </select>

            <button
              type="button"
              className="voice-modal-close-btn"
              onClick={onClose}
              title="Close Voice Assistant (Esc)"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Central Visualizer Section */}
        <div className="voice-modal-body">
          {/* Animated State Badge */}
          <div className={`voice-state-banner ${meta.badgeClass}`}>
            <span className="voice-state-icon">{meta.icon}</span>
            <span className="voice-state-title">{meta.title}</span>
          </div>

          {/* Central Animated Voice Orb (Strict Single Mic Pipeline - Uses Parent Analyser) */}
          <div className="voice-modal-orb-container" style={{ display: 'flex', justifyContent: 'center', margin: '4px 0 10px 0' }}>
            <VoiceOrb
              voiceState={voiceState}
              analyser={analyser}
              size={110}
            />
          </div>

          {/* Audio Canvas Spectrum */}
          <div className="voice-visualizer-container">
            <canvas
              ref={canvasRef}
              width={340}
              height={55}
              className="voice-visualizer-canvas"
            />
            {(voiceState === 'LISTENING' || voiceState === 'WAITING_FOR_USER_REPLY') && (
              <div className="voice-mic-glow-ring"></div>
            )}
          </div>

          <p className="voice-state-subtext">{meta.description}</p>

          {/* Real-Time Live Transcription: Authoritative ASR and Deterministic Normalization */}
          {(partialTranscript || rawTranscript || normalizedTranscript) ? (
            <div className="voice-modal-transcript-box">
              <span className="transcript-header-label">LIVE TRANSCRIPTION</span>
              {isNormalized && rawTranscript ? (
                <div className="transcript-diff-view">
                  <div className="transcript-row">
                    <span className="transcript-tag heard-tag">HEARD:</span>
                    <span className="transcript-text">“{rawTranscript}”</span>
                  </div>
                  <div className="transcript-row">
                    <span className="transcript-tag norm-tag">NORMALIZED:</span>
                    <span className="transcript-text highlight">“{normalizedTranscript}”</span>
                  </div>
                </div>
              ) : (
                <div className="transcript-row">
                  <span className="transcript-tag heard-tag">HEARD:</span>
                  <span className="transcript-text">“{rawTranscript || partialTranscript}”</span>
                </div>
              )}
            </div>
          ) : (voiceState === 'LISTENING' || voiceState === 'WAITING_FOR_USER_REPLY') ? (
            <div className="voice-modal-transcript-placeholder">
              <span>(Listening for speech... Deterministic formatting preserves tags like CDU, 11-P-101A)</span>
            </div>
          ) : null}

          {/* Spoken Text Preview during GREETING, SPEAKING, or CLARIFYING */}
          {(voiceState === 'GREETING' || voiceState === 'SPEAKING' || voiceState === 'SPEAKING_CLARIFICATION' || voiceState === 'CLARIFYING' || voiceState === 'ASKING_CLARIFICATION') && (activeSpokenText || activeClarificationPrompt) && (
            <div className="voice-modal-speaking-box">
              <span className="speaking-label">
                {voiceState === 'GREETING' ? 'Greeting:' : voiceState.includes('CLARIF') ? 'Clarification:' : 'Assistant:'}
              </span>
              <p className="speaking-text">
                {voiceState.includes('CLARIF') ? activeClarificationPrompt : activeSpokenText}
              </p>
            </div>
          )}

          {/* Error Details */}
          {voiceState === 'ERROR' && (
            <div className="voice-modal-error-box">
              <p className="error-title">⚠️ {voiceStatusMsg || 'Microphone permission required'}</p>
              <p className="error-hint">
                Please ensure your microphone is plugged in and allowed in your browser address bar permissions.
              </p>
              <button
                type="button"
                className="btn-voice-retry"
                onClick={onRetry}
              >
                🔄 Try Again
              </button>
            </div>
          )}
        </div>

        {/* Interactive Controls Footer */}
        <div className="voice-modal-footer">
          {(voiceState === 'LISTENING' || voiceState === 'WAITING_FOR_USER_REPLY') && (
            <div className="voice-footer-controls">
              <button
                type="button"
                className="btn-voice-done"
                onClick={onStopListening}
                title="Click to process immediately without waiting for silence"
              >
                ✓ Done Speaking
              </button>
              <button
                type="button"
                className="btn-voice-cancel"
                onClick={onClose}
              >
                Cancel
              </button>
            </div>
          )}

          {(voiceState === 'GREETING' || voiceState === 'SPEAKING' || voiceState === 'SPEAKING_CLARIFICATION' || voiceState === 'CLARIFYING' || voiceState === 'ASKING_CLARIFICATION') && (
            <div className="voice-footer-controls">
              <button
                type="button"
                className="btn-voice-interrupt"
                onClick={onInterrupt}
                title="Interrupt assistant immediately and speak new question"
              >
                ✋ Interrupt (Barge-In)
              </button>
              <span className="voice-interrupt-tip">Or simply start speaking to barge-in</span>
            </div>
          )}

          {(voiceState === 'STARTING' || voiceState === 'PROCESSING' || voiceState === 'UNDERSTANDING' || voiceState === 'SEARCHING' || voiceState === 'THINKING') && (
            <div className="voice-footer-processing">
              <span className="voice-spinner">◌</span>
            </div>
          )}

          {voiceState === 'IDLE' && (
            <div className="voice-footer-controls">
              <button
                type="button"
                className="btn-voice-start"
                onClick={onRetry}
              >
                🎙️ Speak Question
              </button>
            </div>
          )}

          <div className="voice-footer-note">
            <span>● Hands-Free VAD Active: Automatically submits after 800ms silence</span>
          </div>
        </div>
      </div>
    </div>
  );
}
