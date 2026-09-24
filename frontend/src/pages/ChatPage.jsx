import { useState, useEffect, useRef } from 'react';
import { chat, agent, voice } from '../api/client';
import VoiceAssistantModal from '../components/VoiceAssistantModal';
import RobotCorner from '../components/ui/robot-corner';
import CodingResponseCard from '../components/CodingResponseCard';
import { parseCodingResponse, stripRawUiMarkers } from '../lib/codingParser';

export default function ChatPage({ user }) {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [activeTrace, setActiveTrace] = useState(null);
  const [activeTraceId, setActiveTraceId] = useState(null);
  const [activeTraceMeta, setActiveTraceMeta] = useState(null);
  const [showTracePanel, setShowTracePanel] = useState(false);
  const [attachedImage, setAttachedImage] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Voice Assistant states & refs
  const [voiceState, setVoiceState] = useState('IDLE'); // 'IDLE' | 'INITIALIZING' | 'LISTENING' | 'UNDERSTANDING' | 'SEARCHING' | 'THINKING' | 'SPEAKING' | 'CLARIFYING' | 'ERROR'
  const [voiceLanguage, setVoiceLanguage] = useState('auto');
  const [voiceStatusMsg, setVoiceStatusMsg] = useState('');
  const [partialTranscript, setPartialTranscript] = useState('');
  const [activeClarificationPrompt, setActiveClarificationPrompt] = useState(null);
  const [activeAudioPlayingMsgId, setActiveAudioPlayingMsgId] = useState(null);
  const [voiceModalOpen, setVoiceModalOpen] = useState(false);
  const [activeSpokenText, setActiveSpokenText] = useState('');
  const [rawTranscript, setRawTranscript] = useState('');
  const [normalizedTranscript, setNormalizedTranscript] = useState('');
  const [isNormalized, setIsNormalized] = useState(false);
  const [voiceMode, setVoiceMode] = useState(null); // 'NOVA' | 'DICTATION' | null
  const voiceModeRef = useRef(null);
  const voiceSessionIdRef = useRef(null);

  const isDictating = voiceMode === 'DICTATION' || voiceMode === 'VOICE_TO_TEXT';

  const clarificationContextRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const activeAudioRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const abortControllerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const vadIntervalRef = useRef(null);
  const partialTranscribeIntervalRef = useRef(null);
  const speechDetectedRef = useRef(false);
  const lastSpeechTimeRef = useRef(0);
  const interimAbortRef = useRef(null);
  const interimRequestIdRef = useRef(0);
  const activeTurnIdRef = useRef(0);
  const voiceStateRef = useRef('IDLE');
  const sessionListenStartTimeRef = useRef(0);
  const [vadSilenceThresholdMs, setVadSilenceThresholdMs] = useState(750);
  const highpassFilterRef = useRef(null);
  const preprocessedStreamRef = useRef(null);
  const noiseFloorRmsRef = useRef(0.008);
  const speechRmsRef = useRef(0.0);
  const currentRmsRef = useRef(0.0);
  const snrDbRef = useRef(0.0);
  const micDiagnosticsRef = useRef({});
  const vadDiagnosticsRef = useRef({});
  const lastDiagUpdateRef = useRef(0);

  const [vadDiagnostics, setVadDiagnostics] = useState({
    noise_floor_rms: 0.008,
    speech_rms: 0.0,
    current_rms: 0.0,
    snr_db: 0.0,
    vad_confidence: 0.0,
    speech_active: false,
  });
  const [micDiagnostics, setMicDiagnostics] = useState({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    sampleRate: 16000,
    channelCount: 1,
    highpassCutoffHz: 80,
    lowpassCutoffHz: 7500,
  });

  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  useEffect(() => {
    voiceModeRef.current = voiceMode;
  }, [voiceMode]);

  const handleCopy = async (text, id) => {
    if (!text) return;
    let success = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        success = true;
      } catch (e) {
        console.warn('navigator.clipboard failed, using fallback:', e);
      }
    }
    if (!success) {
      try {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        success = document.execCommand('copy');
        document.body.removeChild(textArea);
      } catch (err) {
        console.error('execCommand copy failed:', err);
      }
    }
    if (success) {
      setCopiedId(id);
      setTimeout(() => {
        setCopiedId(null);
      }, 2000);
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadSessions = async () => {
    try {
      const data = await chat.listSessions();
      setSessions(data);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  };

  const selectSession = async (session) => {
    setActiveSession(session);
    try {
      const data = await chat.getSession(session.id);
      const msgs = data.messages || [];
      setMessages(msgs);
      if (msgs.length > 0) {
        const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant' && (m.execution_trace || m.tool_calls));
        if (lastAssistant && lastAssistant.execution_trace) {
          setActiveTrace(lastAssistant.execution_trace);
          setActiveTraceId(lastAssistant.trace_id);
          setActiveTraceMeta({
            task_type: lastAssistant.task_type,
            model_id: lastAssistant.model_id,
            requires_human_review: lastAssistant.requires_human_review,
            approval_id: lastAssistant.approval_id,
          });
        }
      }
    } catch (err) {
      console.error('Failed to load messages:', err);
    }
  };

  const createNewSession = async () => {
    try {
      const session = await chat.createSession();
      setSessions([session, ...sessions]);
      setActiveSession(session);
      setMessages([]);
      setActiveTrace(null);
      setActiveTraceId(null);
      setActiveTraceMeta(null);
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  };

  const handleSelectImage = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const allowedExts = ['jpg', 'jpeg', 'png', 'webp'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
      alert(`Unsupported file format .${ext}. Please upload a .jpg, .jpeg, .png, or .webp image.`);
      e.target.value = '';
      return;
    }
    const maxBytes = 10 * 1024 * 1024; // 10MB
    if (file.size > maxBytes) {
      alert(`Image is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 10 MB.`);
      e.target.value = '';
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    setAttachedImage({
      file,
      name: file.name,
      size: file.size,
      previewUrl,
    });
  };

  const handleRemoveImage = () => {
    if (attachedImage?.previewUrl) {
      URL.revokeObjectURL(attachedImage.previewUrl);
    }
    setAttachedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ── Voice Assistant Handlers (Real-Time Conversational Streaming) ──────
  const cleanupAudioContextAndTracks = () => {
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (partialTranscribeIntervalRef.current) {
      clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = null;
    }
    interimRequestIdRef.current++;
    if (interimAbortRef.current) {
      try { interimAbortRef.current.abort(); } catch (e) {}
      interimAbortRef.current = null;
    }
    if (preprocessedStreamRef.current && preprocessedStreamRef.current !== mediaStreamRef.current) {
      try {
        preprocessedStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) {}
      preprocessedStreamRef.current = null;
    }
    if (mediaStreamRef.current) {
      try {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) {
        // ignore
      }
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close();
      } catch (e) {
        // ignore
      }
      audioContextRef.current = null;
    }
    highpassFilterRef.current = null;
  };

  const handleInterrupt = () => {
    // Immediate barge-in: stop current playback, clear queue, abort backend stream
    if (activeAudioRef.current) {
      try {
        activeAudioRef.current.pause();
        activeAudioRef.current.currentTime = 0;
      } catch (e) {
        // ignore
      }
      activeAudioRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setActiveAudioPlayingMsgId(null);
    setActiveSpokenText('');

    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {
        // ignore
      }
      abortControllerRef.current = null;
    }

    if (activeSession?.id || voiceSessionIdRef.current) {
      voice.interrupt(activeSession?.id, voiceSessionIdRef.current).catch(() => {});
    }

    if (voiceModeRef.current === 'NOVA' && voiceSessionIdRef.current) {
      setVoiceState('LISTENING');
      setTimeout(() => {
        startListening(true);
      }, 40);
    }
  };

  const stopNovaAssistant = () => {
    console.log('Nova Voice Agent: Stop Assistant invoked. Shutting down Nova session completely.');
    const activeVoiceSessionId = voiceSessionIdRef.current;
    voiceSessionIdRef.current = null;
    activeTurnIdRef.current++;

    // Invalidate server-side Nova session & active SSE stream
    if (activeVoiceSessionId || activeSession?.id) {
      voice.stop(activeVoiceSessionId, activeSession?.id).catch(() => {});
    }

    if (abortControllerRef.current) {
      try { abortControllerRef.current.abort(); } catch (e) {}
      abortControllerRef.current = null;
    }
    if (interimAbortRef.current) {
      try { interimAbortRef.current.abort(); } catch (e) {}
      interimAbortRef.current = null;
    }
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (partialTranscribeIntervalRef.current) {
      clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
      } catch (e) {}
      mediaRecorderRef.current = null;
    }
    if (activeAudioRef.current) {
      try {
        activeAudioRef.current.pause();
        activeAudioRef.current.currentTime = 0;
      } catch (e) {}
      activeAudioRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setActiveAudioPlayingMsgId(null);
    setActiveSpokenText('');

    cleanupAudioContextAndTracks();

    clarificationContextRef.current = null;
    setActiveClarificationPrompt(null);
    setVoiceModalOpen(false);
    setVoiceMode(null);
    voiceModeRef.current = null;
    setVoiceState('IDLE');
    setVoiceStatusMsg('NOVA STOPPED');
    setTimeout(() => {
      setVoiceStatusMsg((prev) => (prev === 'NOVA STOPPED' ? '' : prev));
    }, 2500);
  };

  const closeVoiceModal = stopNovaAssistant;

  const acquireMicrophoneStream = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone API not supported on this browser.');
    }
    const supported = (navigator.mediaDevices.getSupportedConstraints && navigator.mediaDevices.getSupportedConstraints()) || {};
    const audioConstraints = {};
    if (supported.echoCancellation !== false) audioConstraints.echoCancellation = true;
    if (supported.noiseSuppression !== false) audioConstraints.noiseSuppression = true;
    if (supported.autoGainControl !== false) audioConstraints.autoGainControl = true;
    if (supported.channelCount !== false) audioConstraints.channelCount = 1;
    if (supported.sampleRate !== false) audioConstraints.sampleRate = 16000;

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    } catch (constraintErr) {
      console.warn('Preferred audio constraints not accepted, falling back to basic audio:', constraintErr);
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }

    try {
      const track = stream.getAudioTracks()[0];
      const trackSettings = track?.getSettings ? track.getSettings() : {};
      const trackConstraints = track?.getConstraints ? track.getConstraints() : {};
      const diag = {
        echoCancellation: trackSettings.echoCancellation ?? trackConstraints.echoCancellation ?? true,
        noiseSuppression: trackSettings.noiseSuppression ?? trackConstraints.noiseSuppression ?? true,
        autoGainControl: trackSettings.autoGainControl ?? trackConstraints.autoGainControl ?? true,
        sampleRate: trackSettings.sampleRate || 16000,
        channelCount: trackSettings.channelCount || 1,
        highpassCutoffHz: 80,
        lowpassCutoffHz: 7500,
      };
      micDiagnosticsRef.current = diag;
      setMicDiagnostics(diag);
    } catch (diagErr) {
      console.warn('Could not extract track diagnostics:', diagErr);
    }
    return stream;
  };

  const setupAudioPreprocessing = async (stream) => {
    try {
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioCtx();
        audioContextRef.current = audioCtx;

        const source = audioCtx.createMediaStreamSource(stream);

        // Web Audio high-pass filter: strips fan/AC rumble < 80 Hz without affecting vocal formants
        const highpass = audioCtx.createBiquadFilter();
        highpass.type = 'highpass';
        highpass.frequency.setValueAtTime(80, audioCtx.currentTime);
        highpass.Q.setValueAtTime(0.707, audioCtx.currentTime);
        highpassFilterRef.current = highpass;

        // Web Audio low-pass filter: cuts high-frequency electrical hiss and fan whine > 7500 Hz
        const lowpass = audioCtx.createBiquadFilter();
        lowpass.type = 'lowpass';
        lowpass.frequency.setValueAtTime(7500, audioCtx.currentTime);
        lowpass.Q.setValueAtTime(0.707, audioCtx.currentTime);

        source.connect(highpass);
        highpass.connect(lowpass);

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        lowpass.connect(analyser);
        analyserRef.current = analyser;

        preprocessedStreamRef.current = stream;
      }
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
    } catch (e) {
      console.warn('AudioContext setup warning:', e);
      preprocessedStreamRef.current = stream;
    }
  };

  const startNovaAssistant = async () => {
    // If voice-to-text / dictation is running, cancel it cleanly
    if (voiceModeRef.current === 'VOICE_TO_TEXT' || voiceModeRef.current === 'DICTATION') {
      cancelVoiceToText();
    }

    // 1. Create a brand-new unique session ID for Nova
    const newSessionId = `nov_ses_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    voiceSessionIdRef.current = newSessionId;
    activeTurnIdRef.current = 1;
    setVoiceMode('NOVA');
    voiceModeRef.current = 'NOVA';

    // Register active session with backend
    if (voice.registerSession) {
      voice.registerSession(newSessionId).catch(() => {});
    }

    setVoiceModalOpen(true);
    setVoiceState('STARTING');
    setVoiceStatusMsg('');
    setPartialTranscript('');
    setActiveSpokenText('');

    setActiveTrace([
      {
        event: 'VOICE_SESSION_STARTED',
        title: 'Nova Voice Assistant session started',
        status: 'allowed',
        timestamp: new Date().toISOString(),
        details: { voice_session_id: newSessionId, voice_mode: 'NOVA' },
      },
      {
        event: 'VOICE_GREETING_STARTED',
        title: 'Playing deterministic Nova greeting',
        status: 'allowed',
        timestamp: new Date().toISOString(),
      },
    ]);

    // Pre-acquire / warm-up microphone so there is zero transition delay when greeting ends
    try {
      if (!mediaStreamRef.current || !mediaStreamRef.current.active) {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await acquireMicrophoneStream();
          mediaStreamRef.current = stream;
          await setupAudioPreprocessing(stream);
        }
      }
    } catch (micErr) {
      console.warn('Microphone pre-warm note:', micErr);
    }

    try {
      setVoiceState('GREETING');
      const greetingData = await voice.greeting(voiceLanguage);
      const greetingText = greetingData?.text || "Hi, I'm Nova. How can I help you today?";
      setActiveSpokenText(greetingText);

      if (greetingData?.audio_base64) {
        const audio = new Audio(`data:audio/wav;base64,${greetingData.audio_base64}`);
        activeAudioRef.current = audio;

        audio.onended = () => {
          activeAudioRef.current = null;
          // Guard: if user stopped Nova while greeting was playing, do not start listening
          if (voiceSessionIdRef.current !== newSessionId || voiceModeRef.current !== 'NOVA') return;
          setActiveTrace((prev) => [
            ...prev,
            {
              event: 'GREETING_COMPLETED',
              title: 'Nova greeting finished speaking',
              status: 'verified',
              timestamp: new Date().toISOString(),
            },
            {
              event: 'LISTENING',
              title: 'Hands-free continuous listening active',
              status: 'allowed',
              timestamp: new Date().toISOString(),
            },
          ]);
          setVoiceState('LISTENING');
          startListening(true);
        };

        audio.onerror = () => {
          activeAudioRef.current = null;
          if (voiceSessionIdRef.current !== newSessionId || voiceModeRef.current !== 'NOVA') return;
          setVoiceState('LISTENING');
          startListening(true);
        };

        audio.play().catch(() => {
          activeAudioRef.current = null;
          if (voiceSessionIdRef.current !== newSessionId || voiceModeRef.current !== 'NOVA') return;
          setVoiceState('LISTENING');
          startListening(true);
        });
      } else {
        setTimeout(() => {
          if (voiceSessionIdRef.current !== newSessionId || voiceModeRef.current !== 'NOVA') return;
          setVoiceState('LISTENING');
          startListening(true);
        }, 800);
      }
    } catch (greetErr) {
      console.warn('Greeting fetch notice:', greetErr);
      if (voiceSessionIdRef.current !== newSessionId || voiceModeRef.current !== 'NOVA') return;
      setVoiceState('LISTENING');
      startListening(true);
    }
  };

  const startVoiceSession = startNovaAssistant;

  const startListening = async (isAutoResume = false) => {
    // Safety guard: only listen when Nova is actively running
    if (voiceModeRef.current !== 'NOVA' || !voiceSessionIdRef.current) {
      return;
    }
    // If speaking or greeting, user speaking is an interruption
    if (!isAutoResume && (voiceState === 'GREETING' || voiceState === 'SPEAKING' || voiceState === 'SPEAKING_CLARIFICATION' || voiceState === 'ASKING_CLARIFICATION' || voiceState === 'CLARIFYING')) {
      handleInterrupt();
      return;
    }

    setVoiceStatusMsg('');
    setPartialTranscript('');
    setVoiceModalOpen(true);
    setVoiceState(clarificationContextRef.current ? 'WAITING_FOR_USER_REPLY' : 'LISTENING');
    sessionListenStartTimeRef.current = Date.now();

    try {
      let stream = mediaStreamRef.current;
      if (!stream || !stream.active) {
        stream = await acquireMicrophoneStream();
        mediaStreamRef.current = stream;
      }
      await setupAudioPreprocessing(stream);

      audioChunksRef.current = [];
      speechDetectedRef.current = false;
      lastSpeechTimeRef.current = Date.now();

      // Setup or resume Web Audio VAD with acoustic barge-in monitoring
      try {
        const analyser = analyserRef.current;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        let noiseFloorRms = noiseFloorRmsRef.current || 0.008;
        const minEnergyFloor = 0.012;
        const snrStartDb = 6.0;
        const snrStopDb = 3.0;
        const silenceThresholdMs = vadSilenceThresholdMs || 750;

        if (vadIntervalRef.current) clearInterval(vadIntervalRef.current);
        vadIntervalRef.current = setInterval(() => {
          if (!analyserRef.current) return;
          analyserRef.current.getByteTimeDomainData(dataArray);

          let sumSquares = 0;
          for (let i = 0; i < dataArray.length; i++) {
            const normalized = (dataArray[i] - 128) / 128;
            sumSquares += normalized * normalized;
          }
          const rms = Math.sqrt(sumSquares / dataArray.length);
          currentRmsRef.current = rms;

          const currentState = voiceStateRef.current;

          // 1. Acoustic Barge-In detection while assistant is speaking or greeting
          if (currentState === 'GREETING' || currentState === 'SPEAKING' || currentState === 'SPEAKING_CLARIFICATION' || currentState === 'ASKING_CLARIFICATION') {
            const bargeInThreshold = Math.max(0.035, noiseFloorRms * 2.5);
            if (rms >= bargeInThreshold) {
              console.log('Voice Agent: Acoustic barge-in detected during playback (RMS:', rms, ', noise floor:', noiseFloorRms, ')');
              handleInterrupt();
              return;
            }
          }

          // 2. Adaptive Turn-taking detection during listening
          if (currentState === 'LISTENING' || currentState === 'WAITING_FOR_USER_REPLY' || currentState === 'USER_SPEAKING') {
            const snrDb = 20 * Math.log10(Math.max(rms, 1e-5) / Math.max(noiseFloorRms, 1e-5));
            snrDbRef.current = snrDb;

            // Dual-threshold hysteresis: start vs stop
            let isSpeechActive = false;
            if (speechDetectedRef.current) {
              isSpeechActive = (snrDb >= snrStopDb && rms >= minEnergyFloor * 0.8) || (rms >= noiseFloorRms * 1.3 && rms >= minEnergyFloor);
            } else {
              isSpeechActive = (rms >= minEnergyFloor) && (snrDb >= snrStartDb || rms >= noiseFloorRms * 2.0);
            }

            if (isSpeechActive) {
              speechDetectedRef.current = true;
              lastSpeechTimeRef.current = Date.now();
              speechRmsRef.current = 0.9 * (speechRmsRef.current || rms) + 0.1 * rms;
              if (currentState !== 'USER_SPEAKING') {
                setVoiceState('USER_SPEAKING');
              }
            } else {
              // Smooth ambient noise floor during non-speech periods
              noiseFloorRms = 0.96 * noiseFloorRms + 0.04 * rms;
              noiseFloorRms = Math.max(Math.min(noiseFloorRms, 0.2), 1e-5);
              noiseFloorRmsRef.current = noiseFloorRms;

              if (speechDetectedRef.current && (Date.now() - lastSpeechTimeRef.current) >= silenceThresholdMs) {
                console.log('Voice Agent: Adaptive endpoint silence detected (SNR:', snrDb.toFixed(1), 'dB, noise floor:', noiseFloorRms.toFixed(4), ') -> finalizing utterance');
                stopListening();
              } else if (!speechDetectedRef.current && sessionListenStartTimeRef.current > 0 && (Date.now() - sessionListenStartTimeRef.current) >= 20000) {
                console.log('Voice Agent: Conversational silence timeout reached.');
                stopNovaAssistant();
              }
            }

            const vadConfidence = isSpeechActive ? Math.min(Math.max((snrDb - snrStopDb) / 12.0, 0.1), 1.0) : 0.0;
            const diagObj = {
              noise_floor_rms: Number(noiseFloorRms.toFixed(4)),
              speech_rms: Number((speechRmsRef.current || 0).toFixed(4)),
              current_rms: Number(rms.toFixed(4)),
              snr_db: Number(snrDb.toFixed(2)),
              vad_confidence: Number(vadConfidence.toFixed(2)),
              speech_active: Boolean(isSpeechActive),
            };
            vadDiagnosticsRef.current = diagObj;

            const now = Date.now();
            if (now - lastDiagUpdateRef.current >= 120) {
              lastDiagUpdateRef.current = now;
              setVadDiagnostics(diagObj);
            }
          }
        }, 35);
      } catch (vadErr) {
        console.warn('Web Audio VAD setup note:', vadErr);
      }

      // Live partial transcription interval: prevents obsolete in-flight requests from blocking final ASR
      if (partialTranscribeIntervalRef.current) clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = setInterval(async () => {
        if (!speechDetectedRef.current || audioChunksRef.current.length === 0) return;
        try {
          if (interimAbortRef.current) {
            try { interimAbortRef.current.abort(); } catch (e) {}
          }
          const abortCtrl = new AbortController();
          interimAbortRef.current = abortCtrl;
          const reqId = ++interimRequestIdRef.current;

          const sliceBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          if (sliceBlob.size > 2000) {
            const interim = await voice.partialTranscribe(sliceBlob, voiceLanguage, abortCtrl.signal);
            if (
              reqId === interimRequestIdRef.current &&
              interim &&
              interim.text &&
              interim.text.trim() &&
              interim.status !== 'repetition_hallucination' &&
              (!interim.quality || interim.quality.valid !== false)
            ) {
              const trimmed = interim.text.trim();
              // Client-side guard against runaway n-gram repetitions
              const words = trimmed.toLowerCase().split(/\s+/);
              let hasPathologicalRepeat = false;
              if (words.length >= 6) {
                for (let n = 2; n <= 4; n++) {
                  const phraseCounts = {};
                  for (let i = 0; i <= words.length - n; i++) {
                    const p = words.slice(i, i + n).join(' ');
                    phraseCounts[p] = (phraseCounts[p] || 0) + 1;
                    if (phraseCounts[p] >= 3) {
                      hasPathologicalRepeat = true;
                      break;
                    }
                  }
                  if (hasPathologicalRepeat) break;
                }
              }

              if (!hasPathologicalRepeat) {
                setPartialTranscript(trimmed);
                if (interim.raw_text) setRawTranscript(interim.raw_text);
                if (interim.normalized_text) setNormalizedTranscript(interim.normalized_text);
                setIsNormalized(Boolean(interim.is_normalized));
              }
            }
          }
        } catch (e) {
          // non-critical interim preview
        }
      }, 1200);

      // Cleanly detach and stop any existing MediaRecorder instance to prevent trailing chunks from polluting the new turn
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state !== 'inactive') {
          try { mediaRecorderRef.current.stop(); } catch (e) {}
        }
        mediaRecorderRef.current = null;
      }

      audioChunksRef.current = [];
      speechDetectedRef.current = false;
      lastSpeechTimeRef.current = Date.now();

      let mimeType = 'audio/webm;codecs=opus';
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported) {
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/webm';
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'audio/ogg;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
              mimeType = '';
            }
          }
        }
      }

      const recordStream = preprocessedStreamRef.current || stream;
      const recorderOptions = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(recordStream, recorderOptions);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (partialTranscribeIntervalRef.current) {
          clearInterval(partialTranscribeIntervalRef.current);
          partialTranscribeIntervalRef.current = null;
        }
        if (interimAbortRef.current) {
          try { interimAbortRef.current.abort(); } catch (e) {}
          interimAbortRef.current = null;
        }

        // If Nova was stopped while recording, do not process and do not restart listening
        if (voiceModeRef.current !== 'NOVA' || !voiceSessionIdRef.current) {
          return;
        }

        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' });
        if (audioBlob.size > 1000) {
          await handleProcessVoiceStream(audioBlob);
        } else {
          setVoiceState('LISTENING');
          setTimeout(() => startListening(true), 60);
        }
      };

      mediaRecorder.start(250); // Slice chunks every 250ms
      setVoiceState(clarificationContextRef.current ? 'WAITING_FOR_USER_REPLY' : 'LISTENING');
    } catch (err) {
      console.error('Microphone access denied:', err);
      cleanupAudioContextAndTracks();
      setVoiceState('ERROR');
      setVoiceStatusMsg(err.name === 'NotAllowedError' ? 'Microphone permission required. Please allow microphone access in your browser address bar.' : 'Microphone unavailable or blocked.');
    }
  };

  const stopListening = () => {
    // 1. Stop scheduling new interim slices
    if (partialTranscribeIntervalRef.current) {
      clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = null;
    }
    // 2. Immediately abort and cancel obsolete in-flight interim transcription work
    interimRequestIdRef.current++;
    if (interimAbortRef.current) {
      try { interimAbortRef.current.abort(); } catch (e) {}
      interimAbortRef.current = null;
    }
    // 3. Immediately finalize current utterance and submit final ASR
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      setVoiceState('UNDERSTANDING');
      mediaRecorderRef.current.stop();
    }
  };

  // ── Normal Voice Input (Dictation) Mode Handlers ───────────────────────
  const startVoiceToText = async () => {
    // If Nova is running, stop it completely
    if (voiceModeRef.current === 'NOVA' || voiceModalOpen) {
      stopNovaAssistant();
    }

    setVoiceMode('DICTATION');
    voiceModeRef.current = 'DICTATION';
    setVoiceModalOpen(false); // Does NOT open modal
    setVoiceState('VOICE_TO_TEXT_LISTENING');
    setVoiceStatusMsg('');
    setPartialTranscript('');

    try {
      let stream = mediaStreamRef.current;
      if (!stream || !stream.active) {
        stream = await acquireMicrophoneStream();
        mediaStreamRef.current = stream;
      }
      await setupAudioPreprocessing(stream);

      audioChunksRef.current = [];
      speechDetectedRef.current = false;
      lastSpeechTimeRef.current = Date.now();
      let firstSpeechTime = 0;
      let consecutiveSpeechFrames = 0;

      // VAD silence detection for Voice-to-Text
      const analyser = analyserRef.current;
      const dataArray = new Uint8Array(analyser ? analyser.frequencyBinCount : 256);
      let noiseFloorRms = noiseFloorRmsRef.current || 0.008;
      const minEnergyFloor = 0.012;
      const snrStartDb = 6.0;
      const snrStopDb = 3.0;
      const silenceThresholdMs = 1500;

      if (vadIntervalRef.current) clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(dataArray);
        let sumSquares = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const normalized = (dataArray[i] - 128) / 128;
          sumSquares += normalized * normalized;
        }
        const rms = Math.sqrt(sumSquares / dataArray.length);
        currentRmsRef.current = rms;

        const snrDb = 20 * Math.log10(Math.max(rms, 1e-5) / Math.max(noiseFloorRms, 1e-5));
        let isSpeechActive = false;
        if (speechDetectedRef.current) {
          isSpeechActive = (snrDb >= snrStopDb && rms >= minEnergyFloor * 0.8) || (rms >= noiseFloorRms * 1.3 && rms >= minEnergyFloor);
        } else {
          isSpeechActive = (rms >= minEnergyFloor) && (snrDb >= snrStartDb || rms >= noiseFloorRms * 2.0);
        }

        if (isSpeechActive) {
          consecutiveSpeechFrames++;
          if (consecutiveSpeechFrames >= 4) {
            if (!speechDetectedRef.current) {
              speechDetectedRef.current = true;
              firstSpeechTime = Date.now();
            }
            lastSpeechTimeRef.current = Date.now();
          }
        } else {
          consecutiveSpeechFrames = 0;
          noiseFloorRms = 0.96 * noiseFloorRms + 0.04 * rms;
          noiseFloorRmsRef.current = noiseFloorRms;
          const speechDuration = firstSpeechTime > 0 ? (Date.now() - firstSpeechTime) : 0;
          if (speechDetectedRef.current && speechDuration >= 400 && (Date.now() - lastSpeechTimeRef.current) >= silenceThresholdMs) {
            stopVoiceToText();
          }
        }
      }, 40);

      // Interim partial transcription
      if (partialTranscribeIntervalRef.current) clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = setInterval(async () => {
        if (!speechDetectedRef.current || audioChunksRef.current.length === 0) return;
        try {
          if (interimAbortRef.current) {
            try { interimAbortRef.current.abort(); } catch (e) {}
          }
          const abortCtrl = new AbortController();
          interimAbortRef.current = abortCtrl;
          const sliceBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          if (sliceBlob.size > 2000) {
            const interim = await voice.partialTranscribe(sliceBlob, voiceLanguage, abortCtrl.signal);
            if (interim && interim.text && interim.text.trim()) {
              setPartialTranscript(interim.text.trim());
            }
          }
        } catch (e) {}
      }, 1000);

      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state !== 'inactive') {
          try { mediaRecorderRef.current.stop(); } catch (e) {}
        }
        mediaRecorderRef.current = null;
      }

      let mimeType = 'audio/webm;codecs=opus';
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported) {
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'audio/webm';
      }

      const recorder = new MediaRecorder(preprocessedStreamRef.current || stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        if (partialTranscribeIntervalRef.current) {
          clearInterval(partialTranscribeIntervalRef.current);
          partialTranscribeIntervalRef.current = null;
        }
        if (vadIntervalRef.current) {
          clearInterval(vadIntervalRef.current);
          vadIntervalRef.current = null;
        }
        const blob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' });
        if (blob.size > 1000) {
          await handleProcessVoiceToText(blob);
        } else {
          setVoiceState('IDLE');
          setVoiceMode(null);
          voiceModeRef.current = null;
          cleanupAudioContextAndTracks();
        }
      };

      recorder.start(250);
    } catch (err) {
      console.error('Voice-to-Text mic access error:', err);
      cleanupAudioContextAndTracks();
      setVoiceState('ERROR');
      setVoiceStatusMsg(err.name === 'NotAllowedError' ? 'Microphone permission denied.' : 'Microphone unavailable.');
      setVoiceMode(null);
      voiceModeRef.current = null;
    }
  };

  const stopVoiceToText = () => {
    if (partialTranscribeIntervalRef.current) {
      clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      setVoiceState('TRANSCRIBING');
      mediaRecorderRef.current.stop();
    }
  };

  const cancelVoiceToText = () => {
    if (partialTranscribeIntervalRef.current) {
      clearInterval(partialTranscribeIntervalRef.current);
      partialTranscribeIntervalRef.current = null;
    }
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      if (mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch (e) {}
      }
      mediaRecorderRef.current = null;
    }
    cleanupAudioContextAndTracks();
    setPartialTranscript('');
    setVoiceState('IDLE');
    setVoiceMode(null);
    voiceModeRef.current = null;
  };

  const handleProcessVoiceToText = async (audioBlob) => {
    setVoiceState('TRANSCRIBING');
    activeTurnIdRef.current++;
    const currentTurn = activeTurnIdRef.current;

    try {
      await voice.processStream(
        audioBlob,
        null,
        voiceLanguage,
        null,
        null,
        (event) => {
          if (activeTurnIdRef.current !== currentTurn) return;
          if (event.event === 'voice_to_text_low_confidence') {
            console.warn('Voice-to-Text low confidence:', event.low_confidence_reason || event.message, event.diagnostics);
            setVoiceStatusMsg(event.message || "Couldn't confidently transcribe that. Please try again.");
            setTimeout(() => setVoiceStatusMsg((prev) => prev.includes('confidently') ? '' : prev), 4500);
          } else if (event.event === 'voice_to_text_completed') {
            const finalNorm = event.normalized_text || event.raw_text || '';
            if (finalNorm) {
              setInput((prev) => (prev && prev.trim() ? `${prev.trim()} ${finalNorm}` : finalNorm));
            }
          } else if (event.event === 'empty') {
            setVoiceStatusMsg(event.message || 'No audible speech detected.');
            setTimeout(() => setVoiceStatusMsg((prev) => prev.includes('speech detected') ? '' : prev), 3500);
          } else if (event.event === 'audio_diagnostics') {
            console.debug('Voice input audio diagnostics:', event);
          }
        },
        null,
        currentTurn,
        'DICTATION',
        null
      );
    } catch (err) {
      console.warn('Voice-to-Text streaming notice:', err);
      try {
        const directResult = await voice.transcribe(audioBlob, voiceLanguage);
        if (directResult && directResult.text) {
          const finalNorm = directResult.normalized_text || directResult.text;
          setInput((prev) => (prev && prev.trim() ? `${prev.trim()} ${finalNorm}` : finalNorm));
        } else if (directResult?.status === 'low_confidence') {
          setVoiceStatusMsg("Couldn't confidently transcribe that. Please try again.");
          setTimeout(() => setVoiceStatusMsg((prev) => prev.includes('confidently') ? '' : prev), 4500);
        }
      } catch (fallbackErr) {
        setVoiceStatusMsg("Couldn't transcribe that. Please try again.");
        setTimeout(() => setVoiceStatusMsg((prev) => prev.includes('transcribe') ? '' : prev), 4500);
      }
    } finally {
      cleanupAudioContextAndTracks();
      setPartialTranscript('');
      setVoiceState('IDLE');
      setVoiceMode(null);
      voiceModeRef.current = null;
    }
  };

  const toggleNormalVoiceInput = () => {
    if (voiceMode === 'NOVA' && voiceState !== 'IDLE') {
      return;
    }
    if (isDictating && (voiceState === 'VOICE_TO_TEXT_LISTENING' || voiceState === 'LISTENING')) {
      stopVoiceToText();
    } else if (isDictating && voiceState === 'TRANSCRIBING') {
      return;
    } else {
      startVoiceToText();
    }
  };

  const playAudioQueue = () => {
    if (isPlayingAudioRef.current) return;
    if (audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingAudioRef.current = true;
    const chunk = audioQueueRef.current.shift();
    if (!chunk || !chunk.audio_base64) {
      isPlayingAudioRef.current = false;
      if (audioQueueRef.current.length > 0) playAudioQueue();
      return;
    }

    if (clarificationContextRef.current) {
      setVoiceState('SPEAKING_CLARIFICATION');
    } else {
      setVoiceState('SPEAKING');
    }

    if (chunk.spoken_text || chunk.text) {
      setActiveSpokenText(chunk.spoken_text || chunk.text);
    }
    try {
      const audio = chunk._preloadedAudio || new Audio(`data:audio/wav;base64,${chunk.audio_base64}`);
      activeAudioRef.current = audio;

      // Preload next queued audio chunk immediately for seamless continuous playback
      if (audioQueueRef.current.length > 0 && !audioQueueRef.current[0]._preloadedAudio) {
        try {
          const nextAudio = new Audio(`data:audio/wav;base64,${audioQueueRef.current[0].audio_base64}`);
          nextAudio.preload = 'auto';
          audioQueueRef.current[0]._preloadedAudio = nextAudio;
        } catch (preloadErr) {}
      }

      audio.onended = () => {
        isPlayingAudioRef.current = false;
        activeAudioRef.current = null;
        // Continuous audio playback: if next chunk is already queued, play immediately without dead gap
        if (audioQueueRef.current.length > 0) {
          playAudioQueue();
        } else if (clarificationContextRef.current) {
          // Clarification question finished speaking! Hands-free auto-listen for reply
          console.log('Voice Agent: Clarification spoken. Listening for user reply hands-free.');
          setVoiceState('WAITING_FOR_USER_REPLY');
          sessionListenStartTimeRef.current = Date.now();
          setTimeout(() => {
            setVoiceState('LISTENING');
            startListening(true);
          }, 60);
        } else {
          // Spoken response finished! Hands-free continuous conversational turn-taking
          console.log('Voice Agent: Spoken response finished. Re-arming hands-free listening.');
          setVoiceState('LISTENING');
          sessionListenStartTimeRef.current = Date.now();
          setTimeout(() => {
            startListening(true);
          }, 60);
        }
      };

      audio.onerror = (err) => {
        console.warn('Audio chunk playback notice:', err);
        isPlayingAudioRef.current = false;
        activeAudioRef.current = null;
        if (audioQueueRef.current.length > 0) {
          playAudioQueue();
        } else {
          setVoiceState('LISTENING');
          sessionListenStartTimeRef.current = Date.now();
          setTimeout(() => {
            startListening(true);
          }, 60);
        }
      };

      audio.play().catch((err) => {
        console.warn('Audio chunk playback notice:', err);
        isPlayingAudioRef.current = false;
        activeAudioRef.current = null;
        if (audioQueueRef.current.length > 0) {
          playAudioQueue();
        } else {
          setVoiceState('LISTENING');
          sessionListenStartTimeRef.current = Date.now();
          setTimeout(() => {
            startListening(true);
          }, 60);
        }
      });
    } catch (e) {
      isPlayingAudioRef.current = false;
      activeAudioRef.current = null;
      if (audioQueueRef.current.length > 0) {
        playAudioQueue();
      } else {
        setVoiceState('LISTENING');
        sessionListenStartTimeRef.current = Date.now();
        setTimeout(() => {
          startListening(true);
        }, 60);
      }
    }
  };

  const handleProcessVoiceStream = async (audioBlob, confirmedText = null) => {
    setVoiceState('UNDERSTANDING');
    let sessionId = activeSession?.id;

    if (!sessionId) {
      try {
        const title = confirmedText ? `[Voice] ${confirmedText.substring(0, 30)}` : '[Voice Interaction]';
        const session = await chat.createSession(title);
        setSessions([session, ...sessions]);
        setActiveSession(session);
        sessionId = session.id;
      } catch (err) {
        console.error('Failed to create session for voice:', err);
        setVoiceState('ERROR');
        setVoiceStatusMsg('Failed to initialize session');
        setTimeout(() => setVoiceState('IDLE'), 4000);
        return;
      }
    }

    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {}
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    activeTurnIdRef.current++;
    const currentTurnId = activeTurnIdRef.current;

    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    if (activeAudioRef.current) {
      try {
        activeAudioRef.current.pause();
        activeAudioRef.current.currentTime = 0;
      } catch (e) {}
      activeAudioRef.current = null;
    }

    const asstMsgId = `vce_${Date.now()}`;
    let streamedContent = '';
    let userMsgAdded = false;

    // Single-turn consumption of clarification context to prevent stale leakage
    const activeClarificationContext = clarificationContextRef.current;
    clarificationContextRef.current = null;
    setActiveTrace([]);
    setActiveTraceId(null);
    setActiveTraceMeta(null);

    try {
      setSending(true);

      await voice.processStream(
        confirmedText ? null : audioBlob,
        sessionId,
        voiceLanguage,
        confirmedText,
        activeClarificationContext,
        (event) => {
          // Guard: ignore stale events from previous turns or stopped Nova sessions
          if (!voiceSessionIdRef.current || voiceModeRef.current !== 'NOVA') {
            return;
          }
          if (event.voice_session_id && event.voice_session_id !== voiceSessionIdRef.current) {
            return;
          }
          if (activeTurnIdRef.current !== currentTurnId || (event.turn_id !== undefined && event.turn_id !== null && event.turn_id !== currentTurnId)) {
            return; // Ignore stale events from previous/aborted turns
          }
          if (event.event === 'language_detected') {
            // Language detected
          } else if (event.event === 'clarification_needed') {
            // Conversational voice clarification: assistant asks clarification question
            clarificationContextRef.current = event.context;
            setActiveClarificationPrompt(event.clarification_text);
            setVoiceState('ASKING_CLARIFICATION');
            setActiveSpokenText(event.clarification_text);
            if (event.raw_text) setRawTranscript(event.raw_text);
            if (event.normalized_text) setNormalizedTranscript(event.normalized_text);
            setIsNormalized(Boolean(event.raw_text && event.normalized_text && event.raw_text !== event.normalized_text));
          } else if (event.event === 'waiting_for_clarification') {
            // Guard against missing audio: if no audio is queued or playing after 800ms, transition directly to LISTENING
            setTimeout(() => {
              if (!isPlayingAudioRef.current && audioQueueRef.current.length === 0) {
                setVoiceState('LISTENING');
                startListening(true);
              }
            }, 800);
          } else if (event.event === 'clarification_resolved') {
            // Clarification answered and resolved into complete query
            clarificationContextRef.current = null;
            setActiveClarificationPrompt(null);
          } else if (event.event === 'asr_final_raw') {
            if (event.raw_text) setRawTranscript(event.raw_text);
          } else if (event.event === 'asr_normalized') {
            if (event.normalized_text) setNormalizedTranscript(event.normalized_text);
            setIsNormalized(Boolean(event.is_normalized));
          } else if (event.event === 'asr_final') {
            const raw = event.raw_text || event.text;
            const norm = event.normalized_text || event.text;
            const isNorm = Boolean(event.is_normalized || (raw !== norm));
            console.log('VOICE_FINAL_TRANSCRIPT:', norm, '(raw:', raw, ')');
            console.log('CHAT_REQUEST:', norm);
            setRawTranscript(raw);
            setNormalizedTranscript(norm);
            setIsNormalized(isNorm);

            const displayLabel = isNorm ? `[${event.language?.name || 'Voice'}] ${norm} (Heard: ${raw})` : `[${event.language?.name || 'Voice'}] ${norm}`;
            // Add user message immediately
            const userMsg = {
              role: 'user',
              content: `🎤 ${displayLabel}`,
              created_at: new Date().toISOString(),
              is_voice: true,
              voice_language: event.language?.name,
              raw_transcript: raw,
              normalized_transcript: norm,
              is_normalized: isNorm,
            };

            // Add placeholder assistant message that will stream tokens
            const placeholderAsst = {
              id: asstMsgId,
              role: 'assistant',
              content: '',
              created_at: new Date().toISOString(),
              is_voice: true,
            };

            setMessages((prev) => [...prev, userMsg, placeholderAsst]);
            userMsgAdded = true;
            setPartialTranscript('');
            setVoiceState('UNDERSTANDING');
          } else if (event.event === 'empty') {
            setSending(false);
            setVoiceStatusMsg(event.message || 'No audible speech detected. Please speak clearly.');
            setVoiceState('LISTENING');
            sessionListenStartTimeRef.current = Date.now();
            setTimeout(() => {
              startListening(true);
            }, 600);
          } else if (event.event === 'routing_decision') {
            if (event.requires_rag) {
              setVoiceState('SEARCHING');
            } else {
              setVoiceState('THINKING');
            }
          } else if (event.event === 'retrieval_started') {
            setVoiceState('SEARCHING');
          } else if (event.event === 'token') {
            // Stream token progressively into UI
            streamedContent += event.token;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === asstMsgId ? { ...m, content: streamedContent } : m
              )
            );
            if (voiceState !== 'SPEAKING' && voiceState !== 'SPEAKING_CLARIFICATION') {
              setVoiceState('THINKING');
            }
          } else if (event.event === 'tts_chunk') {
            // New synthesized audio chunk arrived! Preload and queue for continuous playback
            if (event.audio_base64) {
              const chunkObj = {
                audio_base64: event.audio_base64,
                spoken_text: event.spoken_text || event.text,
                duration_seconds: event.duration_seconds,
              };
              try {
                const preAudio = new Audio(`data:audio/wav;base64,${event.audio_base64}`);
                preAudio.preload = 'auto';
                chunkObj._preloadedAudio = preAudio;
              } catch (e) {}

              audioQueueRef.current.push(chunkObj);
              playAudioQueue();
            }
          } else if (event.event === 'first_audio') {
            setVoiceState('SPEAKING');
          } else if (event.event === 'scope_blocked' || event.event === 'rbac_denied' || event.event === 'insufficient_evidence') {
            if (event.response) {
              streamedContent = event.response;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === asstMsgId ? { ...m, content: streamedContent } : m
                )
              );
            }
          } else if (event.event === 'complete') {
            // Finalize message with complete details, trace, and latencies
            setMessages((prev) =>
              prev.map((m) =>
                m.id === asstMsgId
                  ? {
                      ...m,
                      content: event.response || streamedContent,
                      execution_trace: event.trace,
                      trace_id: event.trace_id,
                      task_type: event.task_type,
                      model_id: event.model_id,
                      latencies: event.latencies,
                      spoken_text: event.spoken_text,
                    }
                  : m
              )
            );

            if (event.trace && event.trace.length > 0) {
              setActiveTrace(event.trace);
              setActiveTraceId(event.trace_id);
              setActiveTraceMeta({
                task_type: event.task_type,
                model_id: event.model_id,
                latencies: event.latencies,
                requires_human_review: event.requires_human_review,
                approval_id: event.approval_id,
                escalation_rule: event.escalation_rule,
              });
            }

            if (audioQueueRef.current.length === 0 && !isPlayingAudioRef.current) {
              setVoiceState('LISTENING');
              sessionListenStartTimeRef.current = Date.now();
              setTimeout(() => {
                startListening(true);
              }, 60);
            }
          } else if (event.event === 'interrupted') {
            // Stream stopped cleanly due to user interruption
            console.log('Voice stream interrupted by user barge-in');
          } else if (event.event === 'error') {
            setVoiceState('ERROR');
            setVoiceStatusMsg(event.error || 'Voice streaming error');
            setTimeout(() => setVoiceState('IDLE'), 5000);
          }
        },
        abortController.signal,
        currentTurnId,
        'NOVA',
        voiceSessionIdRef.current
      );
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Voice streaming request cancelled by barge-in');
      } else {
        console.error('Voice stream failure:', err);
        setVoiceState('ERROR');
        setVoiceStatusMsg(err.message || 'Voice streaming error');
        setTimeout(() => setVoiceState('IDLE'), 4000);
      }
    } finally {
      setSending(false);
      abortControllerRef.current = null;
    }
  };

  const handleReplayVoice = async (msg) => {
    if (msg.audio_base64) {
      audioQueueRef.current = [{ audio_base64: msg.audio_base64, spoken_text: msg.spoken_text }];
      playAudioQueue();
    } else if (msg.content) {
      try {
        setVoiceState('UNDERSTANDING');
        const res = await voice.tts(msg.content, voiceLanguage === 'auto' ? 'en' : voiceLanguage);
        const blob = new Blob([res], { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = reader.result.split(',')[1];
          msg.audio_base64 = b64;
          audioQueueRef.current = [{ audio_base64: b64, spoken_text: msg.content }];
          playAudioQueue();
        };
        reader.readAsDataURL(blob);
      } catch (err) {
        console.error('TTS replay error:', err);
        setVoiceState('IDLE');
      }
    }
  };

  const handleSend = async () => {
    if ((!input.trim() && !attachedImage) || sending) return;

    let sessionId = activeSession?.id;
    const currentQuery = input.trim() || (attachedImage ? 'Please analyze this technical diagram/image in detail and extract all visible equipment tags.' : '');
    const currentImage = attachedImage;

    if (!sessionId) {
      try {
        const title = currentImage ? `[Vision] ${currentImage.name.substring(0, 30)}` : currentQuery.substring(0, 50);
        const session = await chat.createSession(title);
        setSessions([session, ...sessions]);
        setActiveSession(session);
        sessionId = session.id;
      } catch (err) {
        console.error('Failed to create session:', err);
        return;
      }
    }

    const userMsg = {
      role: 'user',
      content: currentQuery,
      created_at: new Date().toISOString(),
      image_preview: currentImage?.previewUrl,
      image_name: currentImage?.name,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setAttachedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setSending(true);
    setActiveTrace([]);
    setActiveTraceId(null);
    setActiveTraceMeta(null);

    try {
      if (currentImage) {
        // Local Sovereign Vision Analysis Flow
        const result = await chat.analyzeImage(sessionId, currentImage.file, currentQuery);
        if (result?.execution_trace && result.execution_trace.length > 0) {
          setActiveTrace(result.execution_trace);
          setActiveTraceId(result.trace_id);
          setActiveTraceMeta({
            task_type: result.task_type || 'VISION',
            model_id: result.model_id,
            requires_human_review: result.requires_human_review,
            approval_id: result.approval_id,
          });
        }
        const data = await chat.getSession(sessionId);
        let msgs = data.messages || [];
        if (msgs.length > 0) {
          const lastMsg = msgs[msgs.length - 1];
          if (lastMsg.role === 'assistant') {
            if (result?.deliverable && !lastMsg.deliverable) lastMsg.deliverable = result.deliverable;
            if (result?.trace_id) lastMsg.trace_id = result.trace_id;
            if (result?.execution_trace) lastMsg.execution_trace = result.execution_trace;
            if (result?.verification) lastMsg.verification = result.verification;
            if (result?.model_id) lastMsg.model_id = result.model_id;
            if (result?.requires_human_review) lastMsg.requires_human_review = result.requires_human_review;
            if (result?.approval_id) lastMsg.approval_id = result.approval_id;
            if (result?.analysis_id) lastMsg.analysis_id = result.analysis_id;
            if (result?.image_hash) lastMsg.image_hash = result.image_hash;
          }
        }
        setMessages(msgs);
      } else {
        // Standard Text Agent Flow
        console.log('CHAT_REQUEST:', currentQuery);
        await chat.sendMessage(sessionId, currentQuery);
        const result = await agent.invoke(currentQuery, sessionId);
        if (result?.execution_trace && result.execution_trace.length > 0) {
          setActiveTrace(result.execution_trace);
          setActiveTraceId(result.trace_id);
          setActiveTraceMeta({
            task_type: result.task_type,
            model_id: result.model_id,
            scope_decision: result.scope_decision,
            confidence_decision: result.confidence_decision,
            citation_validation: result.citation_validation,
            requires_human_review: result.requires_human_review,
            approval_id: result.approval_id,
            escalation_rule: result.escalation_rule,
          });
        }
        const data = await chat.getSession(sessionId);
        let msgs = data.messages || [];

        // Enrich the assistant message with execution details
        if (msgs.length > 0) {
          const lastMsg = msgs[msgs.length - 1];
          if (lastMsg.role === 'assistant') {
            if (result?.deliverable && !lastMsg.deliverable) lastMsg.deliverable = result.deliverable;
            if (result?.task_type) lastMsg.task_type = result.task_type;
            if (result?.model_id) lastMsg.model_id = result.model_id;
            if (result?.sources && (!lastMsg.sources || lastMsg.sources.length === 0)) lastMsg.sources = result.sources;
            if (result?.tool_results) lastMsg.tool_results = result.tool_results;
            if (result?.coding_result) lastMsg.coding_result = result.coding_result;
            if (result?.execution_ms) lastMsg.execution_ms = result.execution_ms;
            if (result?.retrieval_ms) lastMsg.retrieval_ms = result.retrieval_ms;
            lastMsg.trace_id = result?.trace_id;
            lastMsg.execution_trace = result?.execution_trace;
            lastMsg.scope_decision = result?.scope_decision;
            lastMsg.confidence_decision = result?.confidence_decision;
            lastMsg.citation_validation = result?.citation_validation;
            lastMsg.requires_human_review = result?.requires_human_review;
            lastMsg.approval_id = result?.approval_id;
          }
        }
        setMessages(msgs);
      }
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `Error: ${err.message}. Please try again.`,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const deleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await chat.deleteSession(sessionId);
      setSessions(sessions.filter((s) => s.id !== sessionId));
      if (activeSession?.id === sessionId) {
        setActiveSession(null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  };

  const formatSize = (bytes) => {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  const getDeliverable = (msg) => {
    if (msg.deliverable && (msg.deliverable.status === 'success' || msg.deliverable.filename)) {
      return msg.deliverable;
    }
    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        if (tc?.tool?.startsWith('docgen_') && tc.status === 'success' && tc.result) {
          return tc.result;
        }
        if (tc?.result?.type && tc.result?.filename && tc.status === 'success') {
          return tc.result;
        }
      }
    }
    return null;
  };

  const getDeliverableError = (msg) => {
    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        if (tc?.tool?.startsWith('docgen_') && tc.status === 'error') {
          return tc.error || 'Document generation failed';
        }
      }
    }
    return null;
  };

  const handleOpenFile = (deliverable) => {
    const token = localStorage.getItem('mrpl_token');
    const downloadPath = deliverable.download_url || `/api/documents/generated/${deliverable.output_id}/download`;
    const separator = downloadPath.includes('?') ? '&' : '?';
    const url = `${downloadPath}${separator}inline=true&t=${Date.now()}${token ? `&token=${encodeURIComponent(token)}` : ''}`;
    window.open(url, '_blank');
  };

  const handleDownloadFile = (deliverable) => {
    const token = localStorage.getItem('mrpl_token');
    const downloadPath = deliverable.download_url || `/api/documents/generated/${deliverable.output_id}/download`;
    const separator = downloadPath.includes('?') ? '&' : '?';
    const url = `${downloadPath}${separator}inline=false&t=${Date.now()}${token ? `&token=${encodeURIComponent(token)}` : ''}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = deliverable.filename || 'document';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Helper to extract correction attempts
  const getCorrectionAttempts = (msg) => {
    if (Array.isArray(msg.tool_results)) {
      const attempts = msg.tool_results.filter((r) => r.attempt != null);
      if (attempts.length > 0) return attempts.length;
    }
    return null;
  };

  return (
    <div className="chat-container">
      {/* ── Conversations Sidebar ────────────────────────────────────────── */}
      <aside className="chat-sessions" aria-label="Conversation History">
        <div className="chat-sessions-header">
          <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--brand-olive)' }}>
            Conversations
          </span>
          <button
            className="btn btn-primary btn-sm"
            onClick={createNewSession}
            title="Start a new chat session"
            aria-label="New Conversation"
          >
            + New
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${activeSession?.id === session.id ? 'active' : ''}`}
              onClick={() => selectSession(session)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && selectSession(session)}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="session-item-title">{session.title}</div>
                <div className="session-item-date">
                  {new Date(session.updated_at).toLocaleDateString()}
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={(e) => deleteSession(e, session.id)}
                title="Delete session"
                aria-label={`Delete conversation ${session.title}`}
                style={{ padding: '2px 5px', fontSize: '0.7rem', color: 'var(--text-tertiary)' }}
              >
                ✕
              </button>
            </div>
          ))}

          {sessions.length === 0 && (
            <div className="empty-state" style={{ padding: '36px 16px' }}>
              <div className="empty-state-title">No conversations</div>
              <div className="empty-state-description">Click &quot;+ New&quot; to begin a session.</div>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Chat Workspace ──────────────────────────────────────────── */}
      <section className="chat-main" aria-label="Chat Workspace">
        {/* Workspace Context Header */}
        <div className="chat-header">
          <h1 className="chat-header-title">
            {activeSession ? activeSession.title : 'Engineering Chat'}
          </h1>
          <div className="chat-header-badges">
            <span className="badge badge-success">AIR-GAPPED</span>
            <span className="badge badge-neutral">CHROMA LOCAL</span>
          </div>
        </div>

        {/* Chat Context Bar (Requirement 20) */}
        <div className="chat-context-ribbon" role="region" aria-label="Runtime Task Context">
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">MODEL:</span>
            <span className="context-ribbon-value">Auto (Gemma 3 · Qwen 2.5)</span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">KB:</span>
            <span className="context-ribbon-value">MRPL Technical (ChromaDB)</span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">SANDBOX:</span>
            <span className="context-ribbon-value">Docker</span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">SECURITY:</span>
            <span className="context-ribbon-value" style={{ color: 'var(--brand-green)', fontWeight: 600 }}>
              Local Only (Seal Active)
            </span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">ROLE:</span>
            <span className="context-ribbon-value" style={{ textTransform: 'capitalize' }}>
              {user?.roles?.[0] || 'Engineer'}
            </span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <div className="context-ribbon-item">
            <span className="context-ribbon-label">CLEARANCE:</span>
            <span className="context-ribbon-value" style={{ fontWeight: 600, color: 'var(--brand-olive)' }}>
              {user?.clearance || 'CONFIDENTIAL'}
            </span>
          </div>
          <span style={{ color: 'var(--border-strong)' }}>|</span>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setShowTracePanel(!showTracePanel)}
            style={{ fontSize: '0.7rem', padding: '2px 6px', color: 'var(--brand-olive)' }}
            title="Toggle Live Execution Trace sidebar"
          >
            {showTracePanel ? 'Hide Trace ➔' : '🔍 View Trace'}
          </button>
        </div>

        {/* Messages Stream */}
        <div className="chat-messages" role="log" aria-live="polite">
          {messages.length === 0 && !activeSession && (
            <div className="empty-state" style={{ flex: 1 }}>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--brand-olive)', marginBottom: '8px' }}>
                MRPL Engineering Workbench
              </div>
              <div className="empty-state-description">
                Ask about refinery processes, operational technical manuals, vigilance guidelines, financial reports, or sandboxed Python calculations. All processing executes strictly on-premise.
              </div>
            </div>
          )}

          {messages.map((msg, i) => {
            const correctionAttempts = getCorrectionAttempts(msg);
            const deliverable = getDeliverable(msg);
            const deliverableError = getDeliverableError(msg);

            return (
              <div key={i} className={`message message-${msg.role}`}>
                <div className="message-bubble">
                  {/* Compact Model Routing Execution Trace for Assistant Answers */}
                  {msg.role === 'assistant' && (
                    <div className="execution-trace-bar" aria-label="Execution Trace">
                      <span className="trace-step active">
                        <span className="trace-step-icon" aria-hidden="true">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        <span className="trace-step-label">Query Classified</span>
                      </span>
                      <span className="trace-arrow" aria-hidden="true">→</span>

                      <span className={`trace-step ${(msg.sources && msg.sources.length > 0) || msg.task_type === 'RAG' || msg.task_type === 'HYBRID' ? 'active' : ''}`}>
                        <span className="trace-step-icon" aria-hidden="true">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        <span className="trace-step-label">Retrieval</span>
                      </span>
                      <span className="trace-arrow" aria-hidden="true">→</span>

                      <span className="trace-step active">
                        <span className="trace-step-icon" aria-hidden="true">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        <span className="trace-step-label">Model Selected</span>
                      </span>
                      <span className="trace-arrow" aria-hidden="true">→</span>

                      <span className={`trace-step ${msg.task_type === 'CODING' || msg.task_type === 'HYBRID' || deliverable ? 'active' : ''}`}>
                        <span className="trace-step-icon" aria-hidden="true">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        <span className="trace-step-label">Tools / Sandbox</span>
                      </span>
                      <span className="trace-arrow" aria-hidden="true">→</span>

                      <span className="trace-step active" style={{ color: 'var(--success)' }}>
                        <span className="trace-step-icon" aria-hidden="true">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true" focusable="false" role="img">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </span>
                        <span className="trace-step-label">Response Verified</span>
                      </span>
                    </div>
                  )}

                  {/* Human Review Required Banner (Requirement 9) */}
                  {(msg.requires_human_review || (typeof msg.content === 'string' && msg.content.includes('REQUIRES REVIEW'))) && (
                    <div className="alert alert-warning" style={{ fontSize: '0.74rem', padding: '6px 10px', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span>⚠️</span>
                      <span style={{ fontWeight: 600 }}>REQUIRES REVIEW · Action queued for human verification {msg.approval_id ? `(#${msg.approval_id})` : ''}</span>
                    </div>
                  )}

                  {/* Attached Image Thumbnail / Badge for User Message */}
                  {msg.role === 'user' && (msg.image_preview || msg.image_name) && (
                    <div className="message-image-attachment" style={{ marginBottom: '8px' }}>
                      {msg.image_preview && (
                        <img
                          src={msg.image_preview}
                          alt={msg.image_name || 'Attached image'}
                          className="message-image-thumb"
                          style={{ maxWidth: '220px', maxHeight: '160px', borderRadius: '6px', objectFit: 'cover', display: 'block', marginBottom: '4px', cursor: 'pointer', border: '1px solid var(--border-color, #e2e8f0)' }}
                          onClick={() => window.open(msg.image_preview, '_blank')}
                          title="Click to view full image"
                        />
                      )}
                      <div className="message-image-name" style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #64748b)' }}>
                        📷 {msg.image_name || 'Attached Image'}
                      </div>
                    </div>
                  )}

                  {/* Message Content */}
                  {msg.role === 'assistant' && parseCodingResponse(msg) ? (
                    <CodingResponseCard msg={msg} />
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{stripRawUiMarkers(msg.content)}</div>
                  )}

                  {/* Deterministic Verification Results for Vision Analysis */}
                  {msg.verification && (
                    <div className="vision-verification-card" style={{ marginTop: '10px', padding: '8px 12px', background: 'rgba(0,0,0,0.03)', borderRadius: '6px', border: '1px solid var(--border-color, #e2e8f0)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        {msg.verification.status === 'NO_VERIFIABLE_TAGS' ? (
                          <span className="badge badge-info" style={{ fontSize: '0.7rem', background: '#e0f2fe', color: '#0369a1', border: '1px solid #bae6fd' }}>
                            ℹ️ PROCESS SCHEMATIC (NO EQUIPMENT TAGS)
                          </span>
                        ) : (
                          <span className={`badge ${msg.verification.status === 'VERIFIED' ? 'badge-success' : 'badge-warning'}`} style={{ fontSize: '0.7rem' }}>
                            {msg.verification.status === 'VERIFIED' ? '✓ TAGS VERIFIED' : '⚠️ REVIEW REQUIRED'}
                          </span>
                        )}
                        <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--brand-olive)' }}>
                          Deterministic Visual Verification
                        </span>
                      </div>
                      {msg.verification.process_labels && msg.verification.process_labels.length > 0 && (
                        <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary, #475569)', marginBottom: '2px' }}>
                          <strong>Process Labels:</strong> {msg.verification.process_labels.join(', ')}
                        </div>
                      )}
                      {msg.verification.equipment_tags && msg.verification.equipment_tags.length > 0 && (
                        <div style={{ fontSize: '0.74rem', color: 'var(--text-primary)' }}>
                          <strong>Equipment Tags:</strong> {msg.verification.equipment_tags.join(', ')}
                        </div>
                      )}
                      {msg.verification.unknown_tags && msg.verification.unknown_tags.length > 0 && (
                        <div style={{ fontSize: '0.74rem', color: 'var(--danger, #dc2626)', marginTop: '2px' }}>
                          <strong>Unregistered Tags:</strong> {msg.verification.unknown_tags.join(', ')} (Queued for Review)
                        </div>
                      )}
                    </div>
                  )}

                  {/* Deliverable Document Artifact Card */}
                  {deliverable && (
                    <div className="message-deliverable">
                      <div className="deliverable-header">
                        <span>Generated Deliverable</span>
                        <span className="badge badge-success">{deliverable.type?.toUpperCase() || 'DOCUMENT'}</span>
                      </div>
                      <div className="deliverable-file-info">
                        <div className="deliverable-filename">
                          <span>{deliverable.type === 'pdf' ? '📄' : '📝'}</span>
                          <span>{deliverable.filename}</span>
                          {deliverable.file_size_bytes && (
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: 400 }}>
                              ({formatSize(deliverable.file_size_bytes)})
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="deliverable-actions">
                        {deliverable.type === 'pdf' && (
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => handleOpenFile(deliverable)}
                            title="Open PDF preview in new tab"
                            aria-label={`Open PDF ${deliverable.filename}`}
                          >
                            Open PDF
                          </button>
                        )}
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => handleDownloadFile(deliverable)}
                          title={`Download deliverable file ${deliverable.filename}`}
                          aria-label={`Download ${deliverable.filename}`}
                        >
                          Download {deliverable.type?.toUpperCase() || 'File'}
                        </button>
                      </div>
                    </div>
                  )}

                  {deliverableError && (
                    <div className="alert alert-error" style={{ fontSize: '0.8rem', padding: '8px 12px', marginTop: '10px' }} role="alert">
                      <span>⚠️ {deliverableError}</span>
                    </div>
                  )}
                </div>

                {/* Sources & Citations Box */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="message-sources">
                    <div className="message-sources-title">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                      </svg>
                      Verified Knowledge Sources ({msg.sources.length})
                    </div>
                    {msg.sources.map((src, j) => (
                      <div key={j} className="message-source-item">
                        • {src.source || src.document_title || src.category || 'Document'}
                        {src.page ? ` (Page ${src.page})` : ''}
                      </div>
                    ))}
                  </div>
                )}

                {/* Footer Metadata and Copy Option */}
                <div className="message-footer">
                  <div className="message-meta">
                    {msg.task_type && <span style={{ fontWeight: 600, color: 'var(--brand-olive)' }}>[{msg.task_type}] </span>}
                    {msg.model_id && <span>{msg.model_id} · </span>}
                    {correctionAttempts && <span>{correctionAttempts} attempt(s) · </span>}
                    {msg.execution_ms && <span>{msg.execution_ms}ms · </span>}
                    {msg.created_at && new Date(msg.created_at).toLocaleTimeString()}
                  </div>

                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    {msg.execution_trace && msg.execution_trace.length > 0 && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => {
                          setActiveTrace(msg.execution_trace);
                          setActiveTraceId(msg.trace_id);
                          setActiveTraceMeta({
                            task_type: msg.task_type,
                            model_id: msg.model_id,
                            requires_human_review: msg.requires_human_review,
                            approval_id: msg.approval_id,
                          });
                          setShowTracePanel(true);
                        }}
                        style={{ fontSize: '0.68rem', padding: '2px 6px', color: 'var(--brand-olive)' }}
                        title="Inspect live execution trace"
                      >
                        🔍 Trace ({msg.trace_id || `${msg.execution_trace.length} steps`})
                      </button>
                    )}

                    {msg.role === 'assistant' && (
                      <button
                        type="button"
                        className={`btn-listen-msg ${activeAudioPlayingMsgId === (msg.trace_id || i) ? 'playing' : ''}`}
                        onClick={() => handleReplayVoice(msg)}
                        title="Listen to spoken response (Local TTS)"
                        aria-label="Listen to spoken response"
                      >
                        {activeAudioPlayingMsgId === (msg.trace_id || i) ? '🔊 Playing...' : '🔊 Listen'}
                      </button>
                    )}

                    <button
                      type="button"
                      className={`btn-copy-msg ${copiedId === i ? 'copied' : ''}`}
                      onClick={() => handleCopy(msg.coding_result?.code || msg.content, i)}
                      title={msg.role === 'user' ? 'Copy prompt to clipboard' : 'Copy answer to clipboard'}
                      aria-label={msg.role === 'user' ? 'Copy question' : 'Copy answer'}
                    >
                      {copiedId === i ? (
                        <>
                          <span className="btn-icon" aria-hidden="true">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" role="img">
                              <polyline points="20 6 9 17 4 12" />
                            </svg>
                          </span>
                          <span className="btn-label">Copied!</span>
                        </>
                      ) : (
                        <>
                          <span className="btn-icon" aria-hidden="true">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" role="img">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                          </span>
                          <span className="btn-label">Copy {msg.role === 'user' ? 'Question' : 'Answer'}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}

          {sending && (
            <div className="message message-assistant animate-fade-in">
              <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div className="spinner" />
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                  {attachedImage ? 'Analyzing image with Sovereign Vision Model & Verifying Engineering Tags...' : 'Orchestrating Sovereign Agent (Routing · Retrieval · Sandbox Execution)...'}
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ── Input Area ─────────────────────────────────────────────────── */}
        <div className="chat-input-area">
          {/* Attachment Preview Box */}
          {attachedImage && (
            <div className="chat-attachment-preview">
              <img
                src={attachedImage.previewUrl}
                alt="Attachment preview"
                className="chat-attachment-thumb"
              />
              <div className="chat-attachment-info">
                <span className="chat-attachment-name">{attachedImage.name}</span>
                <span className="chat-attachment-size">({formatSize(attachedImage.size)})</span>
              </div>
              <button
                type="button"
                className="chat-attachment-remove"
                onClick={handleRemoveImage}
                title="Remove image attachment"
                aria-label="Remove image attachment"
              >
                ✕
              </button>
            </div>
          )}

          {/* Floating Real-Time Voice Status Pill (Explicit Conversational States) */}
          {voiceState === 'CLARIFYING' && (
            <div className="voice-live-pill clarifying">
              <span>💬 Asking clarification: "{activeClarificationPrompt}"</span>
              <button type="button" className="btn-stop-rec" onClick={handleInterrupt} title="Interrupt clarification">
                Interrupt ✕
              </button>
            </div>
          )}
          {voiceState === 'LISTENING' && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <div className="voice-live-pill listening">
                <span className="pulsing-dot"></span>
                <span>
                  {activeClarificationPrompt
                    ? `Answering: "${activeClarificationPrompt}" — Speak answer`
                    : `Listening (${voiceLanguage === 'auto' ? 'Auto-Detect' : voiceLanguage.toUpperCase()})... Speak question`}
                </span>
                <button type="button" className="btn-stop-rec" onClick={stopListening} title="Click to process immediately">Done ✓</button>
              </div>
              {partialTranscript && (
                <div className="voice-partial-transcript">
                  <span>“{partialTranscript}”</span>
                </div>
              )}
            </div>
          )}
          {voiceState === 'UNDERSTANDING' && (
            <div className="voice-live-pill understanding">
              <span>◌ Understanding speech & verifying security...</span>
            </div>
          )}
          {voiceState === 'SEARCHING' && (
            <div className="voice-live-pill searching">
              <span>◌ Searching technical refinery documentation...</span>
            </div>
          )}
          {voiceState === 'THINKING' && (
            <div className="voice-live-pill thinking">
              <span>◌ Streaming model reasoning...</span>
            </div>
          )}
          {voiceState === 'SPEAKING' && (
            <div className="voice-live-pill speaking">
              <span>🔊 Speaking response...</span>
              <div className="voice-waveform" title="Streaming audio waveform">
                <span className="wave-bar bar-1"></span>
                <span className="wave-bar bar-2"></span>
                <span className="wave-bar bar-3"></span>
                <span className="wave-bar bar-4"></span>
                <span className="wave-bar bar-5"></span>
              </div>
              <button type="button" className="btn-stop-rec" onClick={handleInterrupt} title="Interrupt speech (Barge-in)">
                Interrupt ✕
              </button>
              <span style={{ fontSize: '0.68rem', opacity: 0.8, marginLeft: '4px' }}>(Speak to interrupt)</span>
            </div>
          )}
          {voiceStatusMsg && (
            <div className="voice-live-pill error">
              <span>⚠️ {voiceStatusMsg}</span>
            </div>
          )}

          {/* Normal Voice Input Live Transcription Bar */}
          {isDictating && (voiceState === 'VOICE_TO_TEXT_LISTENING' || voiceState === 'TRANSCRIBING') && (
            <div className="vtt-live-bar" id="vtt-live-bar">
              <div className="vtt-indicator">
                <span className="pulsing-dot-red"></span>
                <span className="vtt-status-label">VOICE INPUT</span>
                <span style={{ fontWeight: 600 }}>LIVE TRANSCRIPTION:</span>
                <span className="vtt-preview-text">
                  {partialTranscript ? `"${partialTranscript}"` : 'Listening... Speak now'}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  type="button"
                  className="btn-vtt-done"
                  onClick={stopVoiceToText}
                  title="Finish speaking and paste transcript into input box"
                  id="btn-vtt-done"
                >
                  Done ✓
                </button>
                <button
                  type="button"
                  className="btn-vtt-cancel"
                  onClick={cancelVoiceToText}
                  title="Cancel voice input"
                  id="btn-vtt-cancel"
                >
                  Cancel ✕
                </button>
              </div>
            </div>
          )}

          <div className="chat-input-wrapper">
            {/* Nova Conversational Voice Assistant Control */}
            <div className="voice-mode-controls">
              {voiceMode === 'NOVA' && voiceState !== 'IDLE' ? (
                <button
                  type="button"
                  className="btn-mode-nova active"
                  onClick={stopNovaAssistant}
                  title="Stop Nova Assistant"
                  id="btn-stop-nova"
                >
                  <span className="pulsing-dot-red"></span>
                  Stop Assistant
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-mode-nova"
                  onClick={startNovaAssistant}
                  disabled={sending || isDictating}
                  title="Start Nova Conversational Voice Assistant"
                  id="btn-start-nova"
                >
                  Nova
                </button>
              )}
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleSelectImage}
              accept=".jpg,.jpeg,.png,.webp"
              style={{ display: 'none' }}
              id="chat-image-upload"
            />
            <button
              type="button"
              className="chat-attach-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={sending}
              title="Attach P&ID, drawing, or photo (.jpg, .jpeg, .png, .webp)"
              aria-label="Attach image"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>

            {/* Multilingual Voice Language Selector */}
            <div className="voice-lang-selector" title="Speech Recognition Language">
              <select
                value={voiceLanguage}
                onChange={(e) => setVoiceLanguage(e.target.value)}
                disabled={(voiceMode === 'NOVA' && (voiceState === 'LISTENING' || voiceState === 'UNDERSTANDING')) || (isDictating && (voiceState === 'VOICE_TO_TEXT_LISTENING' || voiceState === 'TRANSCRIBING')) || sending}
                aria-label="Voice language"
              >
                <option value="auto">🌐 Auto</option>
                <option value="hi">🇮🇳 Hindi</option>
                <option value="mr">🇮🇳 Marathi</option>
                <option value="en">🇬🇧 English</option>
              </select>
            </div>

            {/* Normal Voice Input (Dictation) Microphone Button */}
            <button
              type="button"
              className={`chat-mic-btn ${
                isDictating && voiceState === 'VOICE_TO_TEXT_LISTENING' ? 'listening' :
                isDictating && voiceState === 'TRANSCRIBING' ? 'processing' :
                voiceState === 'ERROR' ? 'error' : ''
              }`}
              onClick={toggleNormalVoiceInput}
              disabled={sending || (voiceMode === 'NOVA' && voiceState !== 'IDLE')}
              title={
                voiceMode === 'NOVA' && voiceState !== 'IDLE' ? 'Nova Assistant is active' :
                isDictating && voiceState === 'VOICE_TO_TEXT_LISTENING' ? 'Listening... Click to finish speaking' :
                isDictating && voiceState === 'TRANSCRIBING' ? 'Transcribing speech...' :
                voiceState === 'ERROR' ? (voiceStatusMsg || 'Microphone error') :
                'Speak to type'
              }
              aria-label="Voice input"
              id="btn-voice-input"
            >
              {isDictating && voiceState === 'VOICE_TO_TEXT_LISTENING' ? (
                <span className="voice-mic-active">
                  <span className="pulse-ring"></span>
                  🔴
                </span>
              ) : isDictating && voiceState === 'TRANSCRIBING' ? (
                <span>◌</span>
              ) : voiceState === 'ERROR' ? (
                <span>⚠</span>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              )}
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={attachedImage ? "Ask a question about this image, or click send for full analysis..." : "Query refinery operations, analyze documents, or request calculation..."}
              rows={1}
              disabled={sending}
              aria-label="Chat query input"
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={(!input.trim() && !attachedImage) || sending}
              title="Submit query to Sovereign Agent"
              aria-label="Send query"
            >
              ↑
            </button>
          </div>
          <p style={{ textAlign: 'center', marginTop: '6px', fontSize: '0.68rem', color: 'var(--text-tertiary)' }}>
            Mangalore Refinery and Petrochemicals Limited · On-Premise Sovereign Inference
          </p>
        </div>
      </section>

      {/* ── Execution Trace Sidebar (Requirement 11) ────────────────────── */}
      {showTracePanel && (
        <aside className="chat-trace-panel" aria-label="Execution Trace">
          <div className="chat-trace-header">
            <div>
              <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--brand-olive)' }}>
                Execution Trace
              </span>
              {activeTraceId && (
                <div style={{ fontSize: '0.66rem', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
                  {activeTraceId}
                </div>
              )}
            </div>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setShowTracePanel(false)}
              title="Close trace panel"
              style={{ fontSize: '0.7rem', padding: '2px 5px' }}
            >
              ✕
            </button>
          </div>

          <div className="chat-trace-content">
            {/* Human Review Banner only on genuine escalation condition */}
            {(() => {
              const approvalStep = activeTrace?.find(
                (s) => s.event === 'approval_required' || s.details?.approval_id || s.details?.escalation_rule
              );
              const isRequired = Boolean(
                activeTraceMeta?.requires_human_review ||
                activeTraceMeta?.approval_id ||
                approvalStep
              );
              if (!isRequired) return null;

              const approvalId = activeTraceMeta?.approval_id || approvalStep?.details?.approval_id;
              const rule = activeTraceMeta?.escalation_rule || approvalStep?.details?.escalation_rule || (activeTraceMeta?.confidence_decision && !activeTraceMeta.confidence_decision.is_sufficient ? 'LOW_RETRIEVAL_CONFIDENCE' : null);
              const reason = activeTraceMeta?.escalation_reason || approvalStep?.details?.reason || approvalStep?.title;

              return (
                <div className="alert alert-warning" style={{ fontSize: '0.72rem', padding: '8px 10px', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 700, marginBottom: '2px' }}>⚠️ REQUIRES REVIEW</div>
                  <div>Action routed to Human Approval Queue{approvalId ? ` (#${approvalId})` : ''}</div>
                  {rule && (
                    <div style={{ fontSize: '0.66rem', marginTop: '3px', color: 'var(--text-secondary)' }}>
                      <strong>Trigger Rule:</strong> {rule}
                    </div>
                  )}
                  {reason && (
                    <div style={{ fontSize: '0.64rem', marginTop: '2px', color: 'var(--text-tertiary)' }}>
                      {reason}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* If trace is available */}
            {activeTrace && activeTrace.length > 0 ? (
              <div className="trace-card">
                {activeTrace.map((step, idx) => (
                  <div key={idx}>
                    <div className="trace-node">
                      <div className={`trace-node-bullet status-${step.status || 'allowed'}`}>
                        {step.status === 'allowed' || step.status === 'verified' || step.status === 'sufficient' ? '✓' :
                         step.status === 'blocked' || step.status === 'denied' ? '✕' : '⚠'}
                      </div>
                      <div className="trace-node-content">
                        <div className="trace-node-title">{step.title}</div>
                        <div className="trace-node-subtitle">
                          {step.event.replace(/_/g, ' ').toUpperCase()} · {new Date(step.timestamp).toLocaleTimeString()}
                        </div>
                        {step.details && Object.keys(step.details).length > 0 && (
                          <div className="trace-node-details">
                            {JSON.stringify(step.details, null, 2)}
                          </div>
                        )}
                      </div>
                    </div>
                    {idx < activeTrace.length - 1 && (
                      <div className="trace-step-arrow">↓</div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '30px 12px' }}>
                <div className="empty-state-title" style={{ fontSize: '0.85rem' }}>No Active Trace</div>
                <div className="empty-state-description" style={{ fontSize: '0.72rem' }}>
                  Submit an engineering query or task to view live execution stages: Scope Check, Authorization, Retrieval, Model Selection, and Citation Grounding.
                </div>
              </div>
            )}
          </div>
        </aside>
      )}

      {/* Real-Time Conversational Voice Assistant Animation Modal */}
      <VoiceAssistantModal
        isOpen={voiceModalOpen}
        onClose={stopNovaAssistant}
        onStopAssistant={stopNovaAssistant}
        voiceState={voiceState}
        voiceLanguage={voiceLanguage}
        onLanguageChange={setVoiceLanguage}
        partialTranscript={partialTranscript}
        rawTranscript={rawTranscript}
        normalizedTranscript={normalizedTranscript}
        isNormalized={isNormalized}
        voiceStatusMsg={voiceStatusMsg}
        onStopListening={stopListening}
        onInterrupt={handleInterrupt}
        onRetry={() => {
          setVoiceState('IDLE');
          setTimeout(() => startVoiceSession(), 80);
        }}
        analyser={analyserRef.current}
        activeSpokenText={activeSpokenText}
        activeClarificationPrompt={activeClarificationPrompt}
        vadDiagnostics={vadDiagnostics}
        micDiagnostics={micDiagnostics}
      />

      {/* 3D Interactive Robot Companion in the Chat Corner */}
      <RobotCorner onRobotClick={() => {
        if (voiceMode === 'NOVA' && voiceState !== 'IDLE') {
          handleInterrupt();
        } else {
          startNovaAssistant();
        }
      }} />
    </div>
  );
}
