import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { ContactShadows } from "@react-three/drei";
import { RobotPrototype } from "./robot-hero";

/**
 * RobotCorner Component
 * 
 * Seamless 3D Robot Companion positioned at the chat border.
 * Completely transparent background — ONLY the large interactive 3D robot is visible.
 * Zero cards, zero white boxes, zero external APIs, zero backend touches.
 */
export default function RobotCorner({
  className = "",
  pantallaColor = "#00ffc6",
  color = "#c8ccc6",
  onRobotClick = null,
  right = "6px",
  bottom = "90px",
}) {
  return (
    <div
      className={`robot-corner-widget ${className}`}
      style={{
        position: "fixed",
        bottom,
        right,
        width: "205px",
        height: "240px",
        zIndex: 999,
        pointerEvents: "auto",
        background: "transparent",
        border: "none",
        boxShadow: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        overflow: "visible",
      }}
      onClick={onRobotClick}
      title="Click 3D Robot to interact"
    >
      <Canvas
        shadows
        gl={{ alpha: true, antialias: true, premultipliedAlpha: false }}
        camera={{ position: [0, 0.05, 2.5], fov: 38 }}
        style={{
          width: "100%",
          height: "100%",
          background: "transparent",
          pointerEvents: "auto",
        }}
      >
        <Suspense fallback={null}>
          {/* 100% Local Sovereign Lighting */}
          <ambientLight intensity={1.2} color="#ffffff" />
          <directionalLight position={[2, 5, 4]} intensity={1.8} color="#ffffff" castShadow />
          <directionalLight position={[-3, 2, -2]} intensity={0.9} color="#c3d9c0" />
          <directionalLight position={[0, -2, 2]} intensity={0.5} color="#e1ede0" />

          <group position={[0, 0.12, 0]}>
            <ContactShadows
              position={[0, -0.74, 0]}
              opacity={0.35}
              scale={5}
              resolution={512}
              blur={1.6}
              far={1.8}
              color="#000000"
            />
            <RobotPrototype
              color={color}
              pantallaColor={pantallaColor}
              pantallaBrillo={1.5}
              blinkCycle={2.8}
              metalness={0.12}
              enableTranslation={false}
            />
          </group>
        </Suspense>
      </Canvas>
    </div>
  );
}
