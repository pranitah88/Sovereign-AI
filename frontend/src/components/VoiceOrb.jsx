import React, { useEffect, useRef } from 'react';

/**
 * VoiceOrb Component
 * 
 * Central visualizer for the MRPL Sovereign Conversational Voice Assistant.
 * Powered by an audio-reactive canvas orb themed in MRPL corporate brand colors.
 * Strictly adheres to the single microphone pipeline: receives Web Audio
 * analyser directly from the parent engine without duplicate microphone acquisition.
 */

export const MRPL_ORB_PALETTE = {
  deepGreen: '#143013',
  darkGreen: '#1b3e19',
  brandGreen: '#234d20',
  olive: '#2c4d28',
  sage: '#c3d9c0',
  sageLight: '#e1ede0',
  accentOrange: '#d96b27',
  white: '#ffffff',
};

export default function VoiceOrb({
  voiceState = 'IDLE',
  analyser = null,
  size = 140,
  className = '',
  onClick = null,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;

    let dataArray = null;
    if (analyser) {
      const bufferLength = analyser.frequencyBinCount || 64;
      dataArray = new Uint8Array(bufferLength);
    }

    const render = (time) => {
      animationId = requestAnimationFrame(render);
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const t = time * 0.002;

      ctx.clearRect(0, 0, w, h);

      // Compute audio level from parent analyzer if available
      let audioLevel = 0;
      if (analyser && dataArray) {
        try {
          analyser.getByteFrequencyData(dataArray);
          let sum = 0;
          const count = Math.min(dataArray.length, 32);
          for (let i = 0; i < count; i++) {
            sum += dataArray[i];
          }
          audioLevel = sum / (count * 255);
        } catch {
          audioLevel = 0;
        }
      }

      // Base radius & pulsation
      const baseRadius = (size * 0.34);
      const pulse = Math.sin(t * 2) * 0.08 + audioLevel * 0.35;
      const currentRadius = Math.max(12, baseRadius * (1 + pulse));

      // Theme colors based on voice state
      let coreColor = MRPL_ORB_PALETTE.brandGreen;
      let glowColor = MRPL_ORB_PALETTE.sage;
      let accentColor = MRPL_ORB_PALETTE.sageLight;

      if (voiceState === 'THINKING' || voiceState === 'SEARCHING' || voiceState === 'UNDERSTANDING') {
        coreColor = MRPL_ORB_PALETTE.darkGreen;
        glowColor = MRPL_ORB_PALETTE.accentOrange;
        accentColor = '#f59e0b';
      } else if (voiceState === 'SPEAKING' || voiceState === 'SPEAKING_CLARIFICATION') {
        coreColor = MRPL_ORB_PALETTE.olive;
        glowColor = MRPL_ORB_PALETTE.sage;
        accentColor = '#a7f3d0';
      } else if (voiceState === 'LISTENING' || voiceState === 'WAITING_FOR_USER_REPLY') {
        coreColor = MRPL_ORB_PALETTE.brandGreen;
        glowColor = '#34d399';
        accentColor = MRPL_ORB_PALETTE.white;
      } else if (voiceState === 'ERROR') {
        glowColor = MRPL_ORB_PALETTE.accentOrange;
      }

      // Outer ambient halo
      const outerGrad = ctx.createRadialGradient(cx, cy, currentRadius * 0.5, cx, cy, currentRadius * 1.5);
      outerGrad.addColorStop(0, glowColor + '55');
      outerGrad.addColorStop(0.6, glowColor + '22');
      outerGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = outerGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius * 1.5, 0, Math.PI * 2);
      ctx.fill();

      // Core sphere gradient
      const coreGrad = ctx.createRadialGradient(
        cx - currentRadius * 0.28,
        cy - currentRadius * 0.28,
        currentRadius * 0.1,
        cx,
        cy,
        currentRadius
      );
      coreGrad.addColorStop(0, accentColor);
      coreGrad.addColorStop(0.35, glowColor);
      coreGrad.addColorStop(0.85, coreColor);
      coreGrad.addColorStop(1, MRPL_ORB_PALETTE.deepGreen);

      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius, 0, Math.PI * 2);
      ctx.fill();

      // Audio-reactive dynamic ripple ring
      if (audioLevel > 0.04 || voiceState === 'LISTENING' || voiceState === 'SPEAKING') {
        const rippleCount = 2;
        for (let i = 0; i < rippleCount; i++) {
          const rPhase = ((t * 1.2 + i * 0.5) % 1.0);
          const rRadius = currentRadius + rPhase * (size * 0.22);
          const rAlpha = (1 - rPhase) * (0.35 + audioLevel * 0.45);
          ctx.strokeStyle = glowColor + Math.floor(rAlpha * 255).toString(16).padStart(2, '0');
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx, cy, rRadius, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Subtle specular shine
      const shineGrad = ctx.createRadialGradient(
        cx - currentRadius * 0.35,
        cy - currentRadius * 0.35,
        1,
        cx - currentRadius * 0.35,
        cy - currentRadius * 0.35,
        currentRadius * 0.38
      );
      shineGrad.addColorStop(0, 'rgba(255, 255, 255, 0.65)');
      shineGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = shineGrad;
      ctx.beginPath();
      ctx.arc(cx - currentRadius * 0.35, cy - currentRadius * 0.35, currentRadius * 0.38, 0, Math.PI * 2);
      ctx.fill();
    };

    animationId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationId);
  }, [voiceState, analyser, size]);

  return (
    <div
      className={`voice-orb-wrapper ${voiceState.toLowerCase()} ${className}`}
      onClick={onClick}
      style={{
        width: size,
        height: size,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: onClick ? 'pointer' : 'default',
        position: 'relative',
        borderRadius: '50%',
        overflow: 'hidden',
      }}
      title={`Voice Agent: ${voiceState}`}
    >
      <canvas
        ref={canvasRef}
        width={size * 2}
        height={size * 2}
        style={{
          width: size,
          height: size,
          display: 'block',
        }}
      />
    </div>
  );
}
