import { useEffect, useState } from 'react';
import { MeshGradient } from '@paper-design/shaders-react';

/**
 * HeroSection Component
 *
 * Interactive hero section featuring a living WebGL MeshGradient background.
 * Colors strictly adhere to the MRPL / ONGC corporate design palette:
 * Deepest green (#143013), dark green (#1b3e19), brand green (#234d20),
 * medium olive (#2c4d28), MRPL signature orange (#d96b27), and muted sage (#c3d9c0).
 *
 * Zero external colors or purple/neon AI slop.
 */

// Authoritative MRPL Corporate Brand Colors
export const MRPL_BRAND_COLORS = [
  '#143013', // Deepest brand green (--brand-green-dark)
  '#1b3e19', // Dark green hover (--brand-green-hover)
  '#234d20', // Core MRPL/ONGC green (--brand-green)
  '#2c4d28', // Medium olive (--brand-olive-light)
  '#d96b27', // MRPL signature orange accent (--brand-orange)
  '#c3d9c0', // Muted sage border/canvas tint (--brand-green-border)
];

export function HeroSection({
  title = 'Sovereign Industrial AI for',
  highlightText = 'MRPL Refinery',
  description = 'Air-gapped on-premise agentic intelligence platform for Mangalore Refinery and Petrochemicals Limited. Local LLM reasoning, deterministic verification, zero cloud egress.',
  buttonText = 'Sign In to Sovereign Workbench',
  onButtonClick,
  colors = MRPL_BRAND_COLORS,
  distortion = 0.75,
  swirl = 0.55,
  speed = 0.35,
  offsetX = 0.08,
  className = '',
  titleClassName = '',
  descriptionClassName = '',
  buttonClassName = '',
  maxWidth = 'max-w-6xl',
  veilOpacity = 'bg-white/20 dark:bg-black/25',
  fontFamily = 'inherit',
  fontWeight = 700,
  children,
}) {
  const [dimensions, setDimensions] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1920,
    height: typeof window !== 'undefined' ? window.innerHeight : 1080,
  });
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const update = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  const handleButtonClick = () => {
    if (onButtonClick) {
      onButtonClick();
    }
  };

  return (
    <section className={`hero-mesh-section relative w-full min-h-screen overflow-hidden flex items-center justify-center ${className}`}>
      {/* Dynamic Animated MeshGradient Canvas Background */}
      <div className="hero-mesh-canvas-container fixed inset-0 w-screen h-screen pointer-events-none" style={{ position: 'fixed', inset: 0, width: '100vw', height: '100vh', zIndex: 0 }}>
        {mounted && (
          <>
            <MeshGradient
              width={dimensions.width}
              height={dimensions.height}
              colors={colors}
              distortion={distortion}
              swirl={swirl}
              grainMixer={0}
              grainOverlay={0}
              speed={speed}
              offsetX={offsetX}
            />
            {/* Subtle optical veil ensuring text readability while letting the organic mesh flow shine */}
            <div
              className={`hero-mesh-veil absolute inset-0 pointer-events-none ${veilOpacity}`}
              style={{
                position: 'absolute',
                inset: 0,
                background: 'radial-gradient(ellipse at center, rgba(20, 48, 19, 0.45) 0%, rgba(12, 28, 11, 0.75) 100%)',
                backdropFilter: 'blur(1px)',
              }}
            />
          </>
        )}
      </div>

      {/* Hero Foreground Content */}
      <div className={`hero-mesh-content relative z-10 ${maxWidth} mx-auto px-6 w-full`} style={{ position: 'relative', zIndex: 10, maxWidth: '1280px', margin: '0 auto', padding: '24px 20px', width: '100%' }}>
        {children ? (
          children
        ) : (
          <div className="hero-mesh-text-container text-center" style={{ textAlign: 'center' }}>
            <h1
              className={`hero-mesh-title font-bold text-foreground text-balance text-4xl sm:text-5xl md:text-6xl xl:text-[80px] leading-tight mb-6 ${titleClassName}`}
              style={{ fontFamily, fontWeight, color: '#ffffff', letterSpacing: '-0.02em', margin: '0 0 16px 0', textShadow: '0 2px 10px rgba(0,0,0,0.4)' }}
            >
              {title}{' '}
              <span className="hero-mesh-highlight text-primary" style={{ color: 'var(--brand-orange, #d96b27)' }}>
                {highlightText}
              </span>
            </h1>
            <p
              className={`hero-mesh-desc text-lg sm:text-xl text-white text-pretty max-w-2xl mx-auto leading-relaxed mb-10 px-4 ${descriptionClassName}`}
              style={{ color: '#edf4eb', maxWidth: '680px', margin: '0 auto 28px', lineHeight: 1.6, textShadow: '0 1px 4px rgba(0,0,0,0.3)' }}
            >
              {description}
            </p>
            {buttonText && (
              <button
                type="button"
                onClick={handleButtonClick}
                className={`hero-mesh-btn px-6 py-4 sm:px-8 sm:py-6 rounded-full border-4 bg-[rgba(63,63,63,1)] border-card text-sm sm:text-base text-white hover:bg-[rgba(63,63,63,0.9)] transition-colors ${buttonClassName}`}
                style={{
                  padding: '12px 28px',
                  borderRadius: '30px',
                  background: 'var(--brand-green, #234d20)',
                  border: '2px solid rgba(255, 255, 255, 0.25)',
                  color: '#ffffff',
                  fontWeight: 700,
                  fontSize: '0.92rem',
                  cursor: 'pointer',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
                  transition: 'all 0.2s ease',
                }}
              >
                {buttonText}
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default HeroSection;
