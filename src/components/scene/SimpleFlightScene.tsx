import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { Html, Line, OrbitControls, PerspectiveCamera, Sky } from '@react-three/drei';
import { Component, lazy, Suspense, type ErrorInfo, type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { BackendEffectMode, BackendTrajectoryProfile, useSimulationStore } from '../../state/simulationStore';
import { TelemetryPoint } from '../../simulation/contracts/telemetryTypes';
import { envelopeStatus } from '../../simulation/aero/flightVisualizationModel';

const TrajectoryCharts = lazy(() => import('./TrajectoryCharts').then(module => ({ default: module.TrajectoryCharts })));

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function safeWebSocketSend(socket: WebSocket | null | undefined, payload: unknown): boolean {
  // Guards against InvalidStateError thrown when ``send`` is called while the
  // socket is still CONNECTING or has been CLOSING/CLOSED. Returns false if
  // the message was dropped so callers can fall back to the next reconnect.
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  try {
    socket.send(typeof payload === 'string' ? payload : JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

// Horizontal world-to-scene scale. 1 scene unit ~= 25 m of real world, which
// keeps a full runway/mission segment in view.
const WORLD_SCALE = 0.04;
// Vertical scale is intentionally separated from horizontal scale. Real chase
// videos preserve aircraft size via camera distance, while altitude opens up
// much faster than the aircraft silhouette. 100 m therefore renders as 6.5
// scene units instead of only 4, avoiding the previous toy-like height ratio.
const ALTITUDE_VISUAL_SCALE = 0.065;
// Moderate visual gain on the real world transform keeps motion readable while
// still leaving several kilometres of apparent terrain/runway coverage.
const HORIZONTAL_MOTION_VISUAL_GAIN = 3.0;
// Natural near-ground optic-flow layer. This moves grass-toned mowing/soil
// bands only — no white debug bars — so speed reads faster without looking
// like synthetic indicators sliding under the aircraft.
const SPEED_DECK_VISUAL_GAIN = 24.0;
const MISSION_AREA_SIZE_M = 10000;
const MISSION_AREA_CENTER_N_M = MISSION_AREA_SIZE_M / 2;
const SPEED_DECK_SPACING_M = 50;
const SPEED_DECK_RANGE_M = MISSION_AREA_SIZE_M;
// Visual offset so the resized Gripen model rests on the runway (not buried)
// when inertial altitude is zero. Calibrated against the STL bounding box.
const AIRCRAFT_WHEEL_OFFSET = 0.15;

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function createCanvasTexture(
  size: number,
  paint: (ctx: CanvasRenderingContext2D, size: number) => void,
  repeat: [number, number],
) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');
  paint(ctx, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeat[0], repeat[1]);
  texture.anisotropy = 4;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeGrassTexture() {
  const rng = seededRandom(0x67a55);
  return createCanvasTexture(512, (ctx, size) => {
    const skyShade = ctx.createLinearGradient(0, 0, size, size);
    skyShade.addColorStop(0, '#40592c');
    skyShade.addColorStop(0.45, '#2e4f27');
    skyShade.addColorStop(1, '#1f3d20');
    ctx.fillStyle = skyShade;
    ctx.fillRect(0, 0, size, size);

    for (let i = 0; i < 18000; i += 1) {
      const x = rng() * size;
      const y = rng() * size;
      const g = 70 + Math.floor(rng() * 70);
      const alpha = 0.08 + rng() * 0.16;
      ctx.fillStyle = rng() > 0.82
        ? `rgba(113, 91, 49, ${alpha * 0.55})`
        : `rgba(${35 + rng() * 45}, ${g}, ${30 + rng() * 35}, ${alpha})`;
      ctx.fillRect(x, y, 1 + rng() * 2.2, 1 + rng() * 2.2);
    }

    // Long, low-contrast mowing/farm strips create believable motion parallax
    // without the synthetic wire-grid look.
    for (let row = -size; row < size * 2; row += 34) {
      ctx.save();
      ctx.translate(size / 2, size / 2);
      ctx.rotate(-0.18);
      ctx.fillStyle = 'rgba(178, 196, 112, 0.045)';
      ctx.fillRect(-size, row - size / 2, size * 2, 9);
      ctx.fillStyle = 'rgba(7, 25, 10, 0.05)';
      ctx.fillRect(-size, row - size / 2 + 17, size * 2, 6);
      ctx.restore();
    }

    // Occasional dry soil blotches break the flat green carpet.
    for (let i = 0; i < 42; i += 1) {
      const x = rng() * size;
      const y = rng() * size;
      const r = 12 + rng() * 36;
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(122, 97, 54, ${0.07 + rng() * 0.08})`);
      grd.addColorStop(1, 'rgba(122, 97, 54, 0)');
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [42, 42]);
}

function makeAsphaltTexture() {
  const rng = seededRandom(0xa55a17);
  return createCanvasTexture(512, (ctx, size) => {
    ctx.fillStyle = '#17191c';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 22000; i += 1) {
      const v = 22 + Math.floor(rng() * 42);
      const a = 0.12 + rng() * 0.2;
      ctx.fillStyle = `rgba(${v}, ${v + 1}, ${v + 3}, ${a})`;
      ctx.fillRect(rng() * size, rng() * size, 1 + rng() * 2, 1 + rng() * 2);
    }
    for (let i = 0; i < 65; i += 1) {
      ctx.strokeStyle = `rgba(5, 6, 7, ${0.08 + rng() * 0.16})`;
      ctx.lineWidth = 1 + rng() * 3;
      ctx.beginPath();
      const y = rng() * size;
      ctx.moveTo(rng() * size * 0.25, y);
      ctx.bezierCurveTo(size * 0.35, y + rng() * 18 - 9, size * 0.65, y + rng() * 24 - 12, size, y + rng() * 28 - 14);
      ctx.stroke();
    }
    // Subtle tire rubber in the center third of the runway.
    const rubber = ctx.createLinearGradient(0, 0, size, 0);
    rubber.addColorStop(0, 'rgba(0,0,0,0)');
    rubber.addColorStop(0.42, 'rgba(0,0,0,0.22)');
    rubber.addColorStop(0.58, 'rgba(0,0,0,0.22)');
    rubber.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = rubber;
    ctx.fillRect(0, 0, size, size);
  }, [4, 54]);
}

function worldFromBody(latest?: TelemetryPoint): {
  position: [number, number, number];
  rotation: [number, number, number];
} {
  // Flight-sim convention: the aircraft is anchored at the scene origin with
  // its nose along scene +z, right wing along +x, belly along -y. We express
  // the inertial world origin (0,0,0)_NED in the body frame and then map
  //   body x (nose)        → scene +z
  //   body y (right wing)  → scene +x
  //   body z (down)        → scene -y
  // so that as the aircraft pitches/rolls (handled by the aircraft mesh) the
  // world slides realistically underneath it, and as it yaws the world rotates
  // about scene +y by -ψ so its visual heading lines up with the cockpit view.
  const altitudeM = Math.max(latest?.altitude ?? 0, 0);
  const pn = latest?.forwardPosition ?? 0;
  const pe = latest?.eastPosition ?? 0;
  const psi = THREE.MathUtils.degToRad(normalizeDegrees(latest?.yawAngle ?? 0));
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  // Aircraft inertial NED position is (pn, pe, -altitude); the inertial origin
  // relative to the aircraft is therefore (-pn, -pe, +altitude). Rotate this
  // into the body frame using the heading transform (pitch/roll already live
  // on the aircraft mesh).
  const dxBody = -pn * c - pe * s; // along nose
  const dyBody = pn * s - pe * c;  // along right wing
  return {
    position: [
      dyBody * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN,
      -(altitudeM * ALTITUDE_VISUAL_SCALE + AIRCRAFT_WHEEL_OFFSET),
      dxBody * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN,
    ],
    rotation: [0, -psi, 0],
  };
}

type SmoothedFlightPose = {
  pn: number;
  pe: number;
  altitude: number;
  yaw: number;
};

function poseFromTelemetry(latest?: TelemetryPoint): SmoothedFlightPose {
  return {
    pn: latest?.forwardPosition ?? 0,
    pe: latest?.eastPosition ?? 0,
    altitude: Math.max(latest?.altitude ?? 0, 0),
    yaw: normalizeDegrees(latest?.yawAngle ?? 0),
  };
}

function smoothPoseTowards(
  pose: SmoothedFlightPose,
  target: SmoothedFlightPose,
  delta: number,
  response = 5.2,
) {
  const factor = 1 - Math.exp(-delta * response);
  pose.pn = THREE.MathUtils.lerp(pose.pn, target.pn, factor);
  pose.pe = THREE.MathUtils.lerp(pose.pe, target.pe, factor);
  pose.altitude = THREE.MathUtils.lerp(pose.altitude, target.altitude, factor);
  pose.yaw = normalizeDegrees(pose.yaw + headingErrorDegrees(pose.yaw, target.yaw) * factor);
}

function worldFromPose(pose: SmoothedFlightPose): {
  position: [number, number, number];
  rotation: [number, number, number];
} {
  const psi = THREE.MathUtils.degToRad(normalizeDegrees(pose.yaw));
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  const dxBody = -pose.pn * c - pose.pe * s;
  const dyBody = pose.pn * s - pose.pe * c;
  return {
    position: [
      dyBody * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN,
      -(pose.altitude * ALTITUDE_VISUAL_SCALE + AIRCRAFT_WHEEL_OFFSET),
      dxBody * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN,
    ],
    rotation: [0, -psi, 0],
  };
}

function aircraftVisualTargetRotation(latest: TelemetryPoint | undefined, visualPhase: number, fightBlend: number) {
  // Only pitch (X) and roll (Z) are applied to the aircraft mesh — yaw is
  // expressed as a counter-rotation of the world (see worldFromBody) so the
  // chase camera, fixed in scene space, always sees the aircraft from the
  // same relative angle. This is the cockpit-out-the-window convention used
  // by every commercial flight simulator (X-Plane, FlightGear, MSFS). Three.js
  // uses a right-handed Y-up scene: with body +x mapped to scene +z and body
  // +z/down mapped to scene -y, positive aerospace pitch/roll must be negated
  // to make pitch-up lift the nose and positive roll lower the right wing.
  const pitch = latest?.pitchAngle ?? 0;
  const flightPath = latest?.flightPathAngle ?? pitch;
  // The source MAV model uses a negative high-speed trim pitch/AoA convention;
  // rendering that literally makes the Gripen look like it is diving while
  // altitude-hold is level. For the visual shell, never draw the nose below the
  // actual flight-path angle. This is visual-only; backend dynamics remain
  // untouched.
  const headingError = headingErrorDegrees(latest?.yawAngle ?? 0, latest?.targetHeading ?? latest?.yawAngle ?? 0);
  const turnDemand = clamp(headingError / 42, -1, 1);
  const verticalDemand = clamp(((latest?.targetAltitude ?? latest?.altitude ?? 0) - (latest?.altitude ?? 0)) / 90, -1, 1);
  const baseBank = clamp(headingError * 2.45, -118, 118);
  const invertedExtension = Math.sign(turnDemand) * 28 * Math.max(0, Math.abs(turnDemand) - 0.62) / 0.38;
  const fightPitchBoost = fightBlend * (verticalDemand * 14 + 3.2 * Math.sin(visualPhase * 0.48) + 2.5 * Math.abs(turnDemand));
  const fightRollBoost = fightBlend * (baseBank + invertedExtension);
  const visualPitch = Math.max(pitch, flightPath) + fightPitchBoost;
  return [
    -THREE.MathUtils.degToRad(visualPitch),
    0,
    -THREE.MathUtils.degToRad((latest?.rollAngle ?? 0) + fightRollBoost),
  ] as [number, number, number];
}

function AircraftVisualAttitude({ latest, children }: { latest?: TelemetryPoint; children: ReactNode }) {
  const groupRef = useRef<THREE.Group>(null);
  const visualPhaseRef = useRef(0);
  const fightBlendRef = useRef(0);
  const targetEulerRef = useRef(new THREE.Euler(0, 0, 0, 'XYZ'));
  const targetQuaternionRef = useRef(new THREE.Quaternion());

  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;
    visualPhaseRef.current += delta;
    const fightTarget = latest?.trajectoryProfile === 'fight_mode'
      ? clamp(((latest?.altitude ?? 0) - 70) / 135, 0, 1)
      : 0;
    fightBlendRef.current = THREE.MathUtils.lerp(
      fightBlendRef.current,
      fightTarget,
      1 - Math.exp(-delta * 1.45),
    );
    const targetRotation = aircraftVisualTargetRotation(latest, visualPhaseRef.current, fightBlendRef.current);
    targetEulerRef.current.set(targetRotation[0], targetRotation[1], targetRotation[2], 'XYZ');
    targetQuaternionRef.current.setFromEuler(targetEulerRef.current);
    group.quaternion.slerp(targetQuaternionRef.current, 1 - Math.exp(-delta * 4.2));
  });

  return <group ref={groupRef}>{children}</group>;
}

type GripenPartSpec = {
  file: string;
  color: string;
  emissive?: string;
  emissiveIntensity?: number;
  metalness?: number;
  roughness?: number;
  opacity?: number;
  pivotOriginal?: [number, number, number];
  rotation?: (latest?: TelemetryPoint) => [number, number, number];
};

const GRIPEN_CENTER_ORIGINAL = new THREE.Vector3(7279.705, 0, 1441.2065);
const GRIPEN_AXIS_MATRIX = new THREE.Matrix4().set(
  0, 1, 0, 0,
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, 0, 0, 1,
);
// The STL is millimeter-scale and spans ~14.56 m in its source X dimension.
// At WORLD_SCALE a true-size Gripen would be ~0.6 scene units long, which is
// too small for a chase-camera demo; the old value made it ~9 units long
// (~225 visual meters), causing altitude to look wildly out of proportion.
// This cinematic 2x-ish scale keeps the aircraft readable while making
// runway width and climb height close to real takeoff/chase footage.
const GRIPEN_SCALE = 0.000082;
const GRIPEN_BASE_PATH = '/models/ro3code/saab-gripen';
const GRIPEN_ASSET_VERSION = 'saab-gripen-v3-clean-only';
const UI_TELEMETRY_SAMPLE_INTERVAL_S = 0.10;
const BACKEND_FRAME_SCHEMA_VERSION = 1;

function gripenScenePoint(original: [number, number, number]): [number, number, number] {
  const shifted = new THREE.Vector3(...original).sub(GRIPEN_CENTER_ORIGINAL);
  // The STL source's +X axis runs aft (canopy is at lower X, rudder/nozzle at
  // higher X). Map *negative* STL X to scene +Z so the rendered nose matches
  // the flight-sim body convention used by worldFromBody.
  return [shifted.y, shifted.z, -shifted.x];
}

const GRIPEN_PARTS: GripenPartSpec[] = [
  { file: 'Body.stl', color: '#cbd5e1', metalness: 0.58, roughness: 0.34 },
  { file: 'Canopy.stl', color: '#0f172a', emissive: '#0369a1', emissiveIntensity: 0.18, metalness: 0.2, roughness: 0.16, opacity: 0.78 },
  { file: 'Canopy_Front.stl', color: '#111827', emissive: '#0891b2', emissiveIntensity: 0.16, metalness: 0.2, roughness: 0.16, opacity: 0.8 },
  { file: 'Canopy_Rear.stl', color: '#111827', emissive: '#0891b2', emissiveIntensity: 0.14, metalness: 0.2, roughness: 0.16, opacity: 0.8 },
  { file: 'FP_Left.stl', color: '#94a3b8', metalness: 0.52, roughness: 0.36 },
  { file: 'FP_Right.stl', color: '#94a3b8', metalness: 0.52, roughness: 0.36 },
  { file: 'LE_Left.stl', color: '#a7f3d0', emissive: '#064e3b', emissiveIntensity: 0.05, metalness: 0.45, roughness: 0.34 },
  { file: 'LE_Right.stl', color: '#a7f3d0', emissive: '#064e3b', emissiveIntensity: 0.05, metalness: 0.45, roughness: 0.34 },
  {
    file: 'Elevon_Left.stl',
    color: '#38bdf8',
    emissive: '#075985',
    emissiveIntensity: 0.18,
    metalness: 0.45,
    roughness: 0.32,
    pivotOriginal: [11254.8, -2419.2661, 426.3549],
    rotation: (latest) => [THREE.MathUtils.degToRad(clamp(latest?.leftAileron ?? 0, -28, 28)), 0, 0],
  },
  {
    file: 'Elevon_Right.stl',
    color: '#38bdf8',
    emissive: '#075985',
    emissiveIntensity: 0.18,
    metalness: 0.45,
    roughness: 0.32,
    pivotOriginal: [11254.8, 2419.2661, 426.3549],
    rotation: (latest) => [THREE.MathUtils.degToRad(clamp(latest?.rightAileron ?? 0, -28, 28)), 0, 0],
  },
  {
    file: 'Rudder.stl',
    color: '#60a5fa',
    emissive: '#1d4ed8',
    emissiveIntensity: 0.15,
    metalness: 0.48,
    roughness: 0.32,
    pivotOriginal: [13275.235, -0.150595, 1387.342],
    rotation: (latest) => [0, THREE.MathUtils.degToRad(clamp(latest?.rudder ?? 0, -30, 30)), 0],
  },
  { file: 'AB_Left.stl', color: '#fb923c', emissive: '#ea580c', emissiveIntensity: 0.28, metalness: 0.35, roughness: 0.28 },
  { file: 'AB_Right.stl', color: '#fb923c', emissive: '#ea580c', emissiveIntensity: 0.28, metalness: 0.35, roughness: 0.28 },
];

function useGripenGeometry(part: GripenPartSpec) {
  const rawGeometry = useLoader(STLLoader, `${GRIPEN_BASE_PATH}/${part.file}?v=${GRIPEN_ASSET_VERSION}`);
  return useMemo(() => {
    const geometry = rawGeometry.clone();
    geometry.translate(-GRIPEN_CENTER_ORIGINAL.x, -GRIPEN_CENTER_ORIGINAL.y, -GRIPEN_CENTER_ORIGINAL.z);
    geometry.applyMatrix4(GRIPEN_AXIS_MATRIX);
    const pivot = part.pivotOriginal ? gripenScenePoint(part.pivotOriginal) : ([0, 0, 0] as [number, number, number]);
    if (part.pivotOriginal) {
      geometry.translate(-pivot[0], -pivot[1], -pivot[2]);
    }
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return { geometry, pivot };
  }, [rawGeometry, part]);
}

function GripenStlPart({ part, latest }: { part: GripenPartSpec; latest?: TelemetryPoint }) {
  const { geometry, pivot } = useGripenGeometry(part);
  const rotation = part.rotation?.(latest) ?? [0, 0, 0];
  const opacity = part.opacity ?? 1;
  return (
    <group position={pivot} rotation={rotation}>
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshPhysicalMaterial
          color={part.color}
          emissive={part.emissive ?? '#000000'}
          emissiveIntensity={part.emissiveIntensity ?? 0}
          metalness={part.metalness ?? 0.42}
          roughness={part.roughness ?? 0.38}
          clearcoat={0.32}
          transparent={opacity < 1}
          opacity={opacity}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

function GripenFlexibleWingOverlay({ latest }: { latest?: TelemetryPoint }) {
  const bend = clamp((latest?.tipDeflection ?? 0) / 1000, -0.42, 0.42) / GRIPEN_SCALE;
  const leftPoints: [number, number, number][] = [[-0.25 / GRIPEN_SCALE, -0.18 / GRIPEN_SCALE, 0.15 / GRIPEN_SCALE], [-2.45 / GRIPEN_SCALE, bend, 1.45 / GRIPEN_SCALE]];
  const rightPoints: [number, number, number][] = [[0.25 / GRIPEN_SCALE, -0.18 / GRIPEN_SCALE, 0.15 / GRIPEN_SCALE], [2.45 / GRIPEN_SCALE, bend, 1.45 / GRIPEN_SCALE]];
  return (
    <group>
      <Line points={leftPoints} color="#facc15" lineWidth={2.2} transparent opacity={0.72} />
      <Line points={rightPoints} color="#facc15" lineWidth={2.2} transparent opacity={0.72} />
    </group>
  );
}

function AfterburnerPlume({ latest }: { latest?: TelemetryPoint }) {
  const throttle = clamp(latest?.throttle ?? 0, 0, 1);
  const opacity = clamp(0.12 + throttle * 0.5, 0.16, 0.62);
  const scale = 0.75 + throttle * 1.05;
  const coreLength = 0.72 + throttle * 0.85;
  return (
    <group position={[0, -0.015, -0.52]} scale={[scale, scale, scale]}>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -coreLength * 0.38]} scale={[0.11, coreLength, 0.11]}>
        <coneGeometry args={[1, 1, 24, 1, true]} />
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={opacity * 0.52}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -coreLength * 0.55]} scale={[0.20, coreLength * 1.25, 0.20]}>
        <coneGeometry args={[1, 1, 28, 1, true]} />
        <meshBasicMaterial
          color="#fb923c"
          transparent
          opacity={opacity * 0.34}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
      <pointLight color="#fb923c" intensity={0.18 + throttle * 0.85} distance={3.2} decay={2.2} />
    </group>
  );
}

function GripenModel({ latest }: { latest?: TelemetryPoint }) {
  return (
    <group>
      <group scale={[GRIPEN_SCALE, GRIPEN_SCALE, GRIPEN_SCALE]}>
        {GRIPEN_PARTS.map((part) => <GripenStlPart key={part.file} part={part} latest={latest} />)}
        <GripenFlexibleWingOverlay latest={latest} />
      </group>
      <AfterburnerPlume latest={latest} />
    </group>
  );
}

function GripenLoadingPlaceholder({ latest }: { latest?: TelemetryPoint }) {
  const envelope = envelopeStatus(latest);
  const materialColor = envelope.severity === 'critical' ? '#fb7185' : '#94a3b8';
  return (
    <group scale={[0.95, 0.95, 0.95]}>
      <mesh castShadow receiveShadow>
        <boxGeometry args={[0.42, 0.28, 3.8]} />
        <meshPhysicalMaterial color={materialColor} metalness={0.55} roughness={0.32} clearcoat={0.45} />
      </mesh>
      <mesh position={[0, 0, -0.15]} rotation={[0, 0, Math.PI / 4]} castShadow receiveShadow>
        <boxGeometry args={[3.4, 0.075, 1.35]} />
        <meshPhysicalMaterial color="#64748b" metalness={0.48} roughness={0.36} />
      </mesh>
      <mesh position={[0, 0.45, 1.45]} castShadow receiveShadow>
        <boxGeometry args={[0.12, 0.9, 0.55]} />
        <meshPhysicalMaterial color="#475569" metalness={0.48} roughness={0.36} />
      </mesh>
      <Html position={[0, 1.2, 0]} center className="pointer-events-none">
        <div className="rounded border border-cyan-500/40 bg-slate-950/80 px-2 py-1 font-mono text-[9px] uppercase text-cyan-100">Loading Gripen STL</div>
      </Html>
    </group>
  );
}

interface GripenErrorBoundaryState {
  hasError: boolean;
}

class GripenModelErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, GripenErrorBoundaryState> {
  state: GripenErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): GripenErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface STL/load failures in the dev console without crashing the whole
    // scene. We deliberately keep this lightweight — the placeholder mesh
    // already communicates the degraded visual state to the operator.
    console.warn('[Gripen model] STL render failed, falling back to placeholder:', error.message, info.componentStack);
  }

  render() {
    return this.state.hasError ? this.props.fallback : this.props.children;
  }
}

function AttitudeReference(_props: { position: [number, number, number]; rotation: [number, number, number] }) {
  // Legacy gimbal-ring overlay retained for binary compatibility; the new
  // flight-sim world already conveys attitude via the rendered horizon and
  // runway. This component intentionally renders nothing.
  void _props;
  return null;
}

function Ring(_props: { axis: string; color: string; rotation: [number, number, number] }) {
  void _props;
  return null;
}

// ============================================================================
// Flight-simulator world: sky, ground, runway, distance markers
// ============================================================================

function FlightSimWorld({ latest }: { latest?: TelemetryPoint }) {
  const groupRef = useRef<THREE.Group>(null);
  const poseRef = useRef<SmoothedFlightPose>(poseFromTelemetry(latest));
  const targetPosition = useMemo(() => new THREE.Vector3(), []);
  const targetEuler = useMemo(() => new THREE.Euler(0, 0, 0, 'XYZ'), []);
  const targetQuaternion = useMemo(() => new THREE.Quaternion(), []);

  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;
    const targetPose = poseFromTelemetry(latest);
    if (!latest || useSimulationStore.getState().experiment.time <= 0.12) {
      poseRef.current = targetPose;
    } else {
      smoothPoseTowards(poseRef.current, targetPose, delta, latest.trajectoryProfile === 'fight_mode' ? 5.2 : 6.4);
    }
    const { position, rotation } = worldFromPose(poseRef.current);
    targetPosition.set(position[0], position[1], position[2]);
    targetEuler.set(rotation[0], rotation[1], rotation[2], 'XYZ');
    targetQuaternion.setFromEuler(targetEuler);

    group.position.copy(targetPosition);
    group.quaternion.copy(targetQuaternion);
  });

  return (
    <group ref={groupRef}>
      <Ground />
      <HighSpeedGroundDeck latest={latest} />
      <Runway />
      <DistanceMarkers />
      <TerrainPatches />
      <DistantTreeLine />
      <DistantCloudLayer />
    </group>
  );
}

function Ground() {
  const grassTexture = useMemo(() => makeGrassTexture(), []);
  useEffect(() => () => grassTexture.dispose(), [grassTexture]);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[250000 * WORLD_SCALE, 250000 * WORLD_SCALE, 1, 1]} />
        <meshStandardMaterial map={grassTexture} roughness={0.98} metalness={0} color="#9fb17c" />
      </mesh>
    </group>
  );
}

function MissionArea() {
  const areaSize = MISSION_AREA_SIZE_M * WORLD_SCALE;
  const centerZ = MISSION_AREA_CENTER_N_M * WORLD_SCALE;
  const squarePoints = useMemo(() => {
    const h = areaSize / 2;
    return [
      new THREE.Vector3(-h, 0.03, centerZ - h),
      new THREE.Vector3(h, 0.03, centerZ - h),
      new THREE.Vector3(h, 0.03, centerZ + h),
      new THREE.Vector3(-h, 0.03, centerZ + h),
      new THREE.Vector3(-h, 0.03, centerZ - h),
    ];
  }, [areaSize, centerZ]);
  const circlePoints = (radiusM: number, y: number) => Array.from({ length: 129 }, (_, i) => {
    const a = (i / 128) * Math.PI * 2;
    return new THREE.Vector3(
      Math.cos(a) * radiusM * WORLD_SCALE,
      y,
      centerZ + Math.sin(a) * radiusM * WORLD_SCALE,
    );
  });
  const circle2km = useMemo(() => circlePoints(1000, 0.038), [centerZ]);
  const circle5km = useMemo(() => circlePoints(2500, 0.036), [centerZ]);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.006, centerZ]} receiveShadow>
        <planeGeometry args={[areaSize, areaSize, 1, 1]} />
        <meshStandardMaterial color="#3f8f36" transparent opacity={0.38} roughness={1} metalness={0} depthWrite={false} />
      </mesh>
      <Line points={squarePoints} color="#4ade80" lineWidth={1.6} transparent opacity={0.88} />
      <Line points={circle5km} color="#22d3ee" lineWidth={1.7} transparent opacity={0.82} />
      <Line points={circle2km} color="#facc15" lineWidth={1.9} transparent opacity={0.88} />
      <Html position={[-areaSize / 2 + 120 * WORLD_SCALE, 3.0 * WORLD_SCALE, centerZ - areaSize / 2 + 160 * WORLD_SCALE]} transform sprite>
        <div className="rounded border border-emerald-300/50 bg-slate-950/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-emerald-100">
          10 km green mission area
        </div>
      </Html>
      <Html position={[1050 * WORLD_SCALE, 3.0 * WORLD_SCALE, centerZ]} transform sprite>
        <div className="rounded bg-slate-950/70 px-1.5 py-0.5 font-mono text-[9px] text-yellow-200">200 m circle</div>
      </Html>
      <Html position={[2550 * WORLD_SCALE, 3.0 * WORLD_SCALE, centerZ]} transform sprite>
        <div className="rounded bg-slate-950/70 px-1.5 py-0.5 font-mono text-[9px] text-cyan-100">mission area</div>
      </Html>
    </group>
  );
}

function TerrainPatches() {
  const dryPatchRef = useRef<THREE.InstancedMesh>(null);
  const darkPatchRef = useRef<THREE.InstancedMesh>(null);
  const soilPatchRef = useRef<THREE.InstancedMesh>(null);
  const matrixDummy = useMemo(() => new THREE.Object3D(), []);
  const patchSets = useMemo(() => {
    const rng = seededRandom(0x5ce7a11);
    const makeSet = (count: number, minSize: number, maxSize: number) => Array.from({ length: count }, () => {
      let x = (rng() - 0.5) * MISSION_AREA_SIZE_M * WORLD_SCALE;
      const z = (-800 + rng() * (MISSION_AREA_SIZE_M + 1600)) * WORLD_SCALE;
      if (Math.abs(x) < 45 * WORLD_SCALE && z > -260 * WORLD_SCALE && z < 2350 * WORLD_SCALE) {
        x += Math.sign(x || 1) * (80 + rng() * 240) * WORLD_SCALE;
      }
      return {
        x,
        z,
        sx: (minSize + rng() * (maxSize - minSize)) * WORLD_SCALE,
        sz: (minSize + rng() * (maxSize - minSize)) * WORLD_SCALE,
        rot: rng() * Math.PI,
      };
    });
    return {
      dry: makeSet(140, 50, 260),
      dark: makeSet(120, 42, 210),
      soil: makeSet(70, 24, 130),
    };
  }, []);

  useLayoutEffect(() => {
    const setPatchInstances = (
      mesh: THREE.InstancedMesh | null,
      patches: { x: number; z: number; sx: number; sz: number; rot: number }[],
    ) => {
      if (!mesh) return;
      patches.forEach((patch, index) => {
        matrixDummy.position.set(patch.x, 0.011, patch.z);
        matrixDummy.rotation.set(-Math.PI / 2, 0, patch.rot);
        matrixDummy.scale.set(patch.sx, patch.sz, 1);
        matrixDummy.updateMatrix();
        mesh.setMatrixAt(index, matrixDummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      matrixDummy.scale.set(1, 1, 1);
    };
    setPatchInstances(dryPatchRef.current, patchSets.dry);
    setPatchInstances(darkPatchRef.current, patchSets.dark);
    setPatchInstances(soilPatchRef.current, patchSets.soil);
  }, [matrixDummy, patchSets]);

  return (
    <group>
      <instancedMesh ref={dryPatchRef} args={[undefined, undefined, patchSets.dry.length]}>
        <circleGeometry args={[1, 18]} />
        <meshBasicMaterial color="#8f8a52" transparent opacity={0.18} depthWrite={false} />
      </instancedMesh>
      <instancedMesh ref={darkPatchRef} args={[undefined, undefined, patchSets.dark.length]}>
        <circleGeometry args={[1, 18]} />
        <meshBasicMaterial color="#18331a" transparent opacity={0.20} depthWrite={false} />
      </instancedMesh>
      <instancedMesh ref={soilPatchRef} args={[undefined, undefined, patchSets.soil.length]}>
        <circleGeometry args={[1, 16]} />
        <meshBasicMaterial color="#6f5230" transparent opacity={0.24} depthWrite={false} />
      </instancedMesh>
    </group>
  );
}

function DistantTreeLine() {
  const treeRef = useRef<THREE.InstancedMesh>(null);
  const matrixDummy = useMemo(() => new THREE.Object3D(), []);
  const trees = useMemo(() => {
    const rng = seededRandom(0x7eecafe);
    return Array.from({ length: 120 }, (_, index) => {
      const side = index < 60 ? -1 : 1;
      return {
        x: side * (5200 + rng() * 420) * WORLD_SCALE,
        z: (-500 + rng() * (MISSION_AREA_SIZE_M + 1200)) * WORLD_SCALE,
        h: (9 + rng() * 15) * WORLD_SCALE,
        r: (3.5 + rng() * 5.5) * WORLD_SCALE,
      };
    });
  }, []);

  useLayoutEffect(() => {
    const mesh = treeRef.current;
    if (!mesh) return;
    trees.forEach((tree, index) => {
      matrixDummy.position.set(tree.x, tree.h * 0.42, tree.z);
      matrixDummy.rotation.set(0, index * 0.37, 0);
      matrixDummy.scale.set(tree.r, tree.h, tree.r);
      matrixDummy.updateMatrix();
      mesh.setMatrixAt(index, matrixDummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    matrixDummy.scale.set(1, 1, 1);
  }, [matrixDummy, trees]);

  return (
    <instancedMesh ref={treeRef} args={[undefined, undefined, trees.length]}>
      <coneGeometry args={[1, 1, 5]} />
      <meshStandardMaterial color="#18311c" roughness={1} metalness={0} />
    </instancedMesh>
  );
}

function DistantCloudLayer() {
  const cloudRef = useRef<THREE.InstancedMesh>(null);
  const matrixDummy = useMemo(() => new THREE.Object3D(), []);
  const clouds = useMemo(() => {
    const rng = seededRandom(0xc10d5);
    return Array.from({ length: 42 }, () => ({
      x: (rng() - 0.5) * 11000 * WORLD_SCALE,
      y: (360 + rng() * 620) * ALTITUDE_VISUAL_SCALE,
      z: (-1200 + rng() * (MISSION_AREA_SIZE_M + 4200)) * WORLD_SCALE,
      sx: (120 + rng() * 320) * WORLD_SCALE,
      sy: (22 + rng() * 72) * WORLD_SCALE,
      rot: (rng() - 0.5) * 0.28,
    }));
  }, []);

  useLayoutEffect(() => {
    const mesh = cloudRef.current;
    if (!mesh) return;
    clouds.forEach((cloud, index) => {
      matrixDummy.position.set(cloud.x, cloud.y, cloud.z);
      matrixDummy.rotation.set(-0.12, 0, cloud.rot);
      matrixDummy.scale.set(cloud.sx, cloud.sy, 1);
      matrixDummy.updateMatrix();
      mesh.setMatrixAt(index, matrixDummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    matrixDummy.scale.set(1, 1, 1);
  }, [clouds, matrixDummy]);

  return (
    <instancedMesh ref={cloudRef} args={[undefined, undefined, clouds.length]} frustumCulled={false}>
      <circleGeometry args={[1, 28]} />
      <meshBasicMaterial color="#f8fbff" transparent opacity={0.34} depthWrite={false} />
    </instancedMesh>
  );
}

function Runway() {
  const asphaltTexture = useMemo(() => makeAsphaltTexture(), []);
  useEffect(() => () => asphaltTexture.dispose(), [asphaltTexture]);
  // 2400 m × 50 m asphalt strip aligned with inertial north (scene +z).
  // Standard ICAO runway markings: dashed centerline, threshold piano keys,
  // designator numbers at each end.
  const lengthM = 2400;
  const widthM = 50;
  const L = lengthM * WORLD_SCALE;
  const W = widthM * WORLD_SCALE;
  const stripeCount = 24;
  const thresholdOffsets = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
  return (
    // South threshold sits 200 m behind the world origin so the aircraft
    // spawns just ahead of the displaced threshold and rolls down the runway
    // as pn increases. The asphalt strip extends from z = -200 m to z = +2200 m
    // in NED-north coordinates relative to the world group.
    <group position={[0, 0.02, (L / 2) - 200 * WORLD_SCALE]}>
      {/* Compacted grass/soil shoulder gives the runway a realistic edge
          transition instead of a hard asphalt-to-green seam. */}
      {[-1, 1].map((side) => (
        <mesh key={`shoulder-${side}`} rotation={[-Math.PI / 2, 0, 0]} position={[side * (W / 2 + 7 * WORLD_SCALE), -0.002, 0]} receiveShadow>
          <planeGeometry args={[14 * WORLD_SCALE, L]} />
          <meshStandardMaterial color="#4f5531" roughness={1} metalness={0} />
        </mesh>
      ))}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[W, L]} />
        <meshStandardMaterial map={asphaltTexture} color="#6b7074" roughness={0.92} metalness={0.03} />
      </mesh>
      {/* FAA-style continuous side stripes provide edge contrast against the
          abutting terrain/shoulder. */}
      {[-1, 1].map((side) => (
        <mesh key={`edge-stripe-${side}`} rotation={[-Math.PI / 2, 0, 0]} position={[side * (W / 2 - 1.4 * WORLD_SCALE), 0.013, 0]}>
          <planeGeometry args={[0.9 * WORLD_SCALE, L * 0.96]} />
          <meshBasicMaterial color="#f8fafc" transparent opacity={0.88} />
        </mesh>
      ))}
      {/* Dashed centerline */}
      {Array.from({ length: stripeCount }).map((_, i) => {
        const z = -L / 2 + (i + 0.5) * (L / stripeCount);
        return (
          <mesh key={`cl-${i}`} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, z]}>
            <planeGeometry args={[0.7 * WORLD_SCALE, 30 * WORLD_SCALE]} />
            <meshBasicMaterial color="#f8fafc" />
          </mesh>
        );
      })}
      {/* Aiming point blocks roughly 1000 ft from each threshold. */}
      {[-1, 1].flatMap((end) => [-1, 1].map((side) => (
        <mesh
          key={`aim-${end}-${side}`}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[side * 7.5 * WORLD_SCALE, 0.014, end * (L / 2 - 305 * WORLD_SCALE)]}
        >
          <planeGeometry args={[6 * WORLD_SCALE, 45 * WORLD_SCALE]} />
          <meshBasicMaterial color="#f8fafc" transparent opacity={0.84} />
        </mesh>
      )))}
      {/* Threshold piano keys (south end → RWY 36 numbering) */}
      {thresholdOffsets.map((k) => (
        <mesh
          key={`th-s-${k}`}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[k * 3.4 * WORLD_SCALE, 0.012, -L / 2 + 8 * WORLD_SCALE]}
        >
          <planeGeometry args={[2.2 * WORLD_SCALE, 20 * WORLD_SCALE]} />
          <meshBasicMaterial color="#f1f5f9" />
        </mesh>
      ))}
      {thresholdOffsets.map((k) => (
        <mesh
          key={`th-n-${k}`}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[k * 3.4 * WORLD_SCALE, 0.012, L / 2 - 8 * WORLD_SCALE]}
        >
          <planeGeometry args={[2.2 * WORLD_SCALE, 20 * WORLD_SCALE]} />
          <meshBasicMaterial color="#f1f5f9" />
        </mesh>
      ))}
      <RunwayEdgeLights lengthScene={L} widthScene={W} />
      {/* Runway designator labels (HTML, billboarded onto the asphalt) */}
      <Html
        position={[0, 0.02, -L / 2 + 35 * WORLD_SCALE]}
        rotation={[-Math.PI / 2, 0, 0]}
        transform
        occlude
        sprite={false}
      >
        <div className="select-none font-mono text-[42px] font-black text-slate-200" style={{ letterSpacing: '-2px' }}>36</div>
      </Html>
      <Html
        position={[0, 0.02, L / 2 - 35 * WORLD_SCALE]}
        rotation={[-Math.PI / 2, Math.PI, 0]}
        transform
        occlude
        sprite={false}
      >
        <div className="select-none font-mono text-[42px] font-black text-slate-200" style={{ letterSpacing: '-2px' }}>18</div>
      </Html>
    </group>
  );
}

function RunwayEdgeLights({ lengthScene, widthScene }: { lengthScene: number; widthScene: number }) {
  const lightRef = useRef<THREE.InstancedMesh>(null);
  const matrixDummy = useMemo(() => new THREE.Object3D(), []);
  const lights = useMemo(() => {
    const arr: { x: number; z: number; s: number }[] = [];
    const spacing = 100 * WORLD_SCALE;
    for (let z = -lengthScene / 2 + 60 * WORLD_SCALE; z <= lengthScene / 2 - 60 * WORLD_SCALE; z += spacing) {
      arr.push({ x: -widthScene / 2 - 2.2 * WORLD_SCALE, z, s: 1 });
      arr.push({ x: widthScene / 2 + 2.2 * WORLD_SCALE, z, s: 1 });
    }
    return arr;
  }, [lengthScene, widthScene]);

  useLayoutEffect(() => {
    const mesh = lightRef.current;
    if (!mesh) return;
    lights.forEach((light, index) => {
      matrixDummy.position.set(light.x, 0.055, light.z);
      matrixDummy.scale.setScalar(0.035 * light.s);
      matrixDummy.updateMatrix();
      mesh.setMatrixAt(index, matrixDummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    matrixDummy.scale.setScalar(1);
  }, [lights, matrixDummy]);

  return (
    <instancedMesh ref={lightRef} args={[undefined, undefined, lights.length]}>
      <sphereGeometry args={[1, 8, 6]} />
      <meshBasicMaterial color="#e8f7ff" transparent opacity={0.78} />
    </instancedMesh>
  );
}

function HighSpeedGroundDeck({ latest }: { latest?: TelemetryPoint }) {
  const groupRef = useRef<THREE.Group>(null);
  const majorStripeRef = useRef<THREE.InstancedMesh>(null);
  const minorStripeRef = useRef<THREE.InstancedMesh>(null);
  const matrixDummy = useMemo(() => new THREE.Object3D(), []);
  const airspeed = clamp(latest?.airspeed ?? latest?.targetAirspeed ?? 200, 0, 220);
  const spacing = SPEED_DECK_SPACING_M * WORLD_SCALE;
  const range = SPEED_DECK_RANGE_M * WORLD_SCALE;
  const visualSpeed = Math.max(airspeed, 8) * WORLD_SCALE * SPEED_DECK_VISUAL_GAIN;
  const stripes = useMemo(() => {
    const arr: number[] = [];
    for (let z = -range * 0.45; z <= range * 0.55; z += spacing) arr.push(z);
    return arr;
  }, [range, spacing]);
  const majorStripes = useMemo(() => stripes.filter((_, index) => index % 4 === 0), [stripes]);
  const minorStripes = useMemo(() => stripes.filter((_, index) => index % 4 !== 0), [stripes]);

  useLayoutEffect(() => {
    const setInstances = (
      mesh: THREE.InstancedMesh | null,
      positions: number[],
      x: number,
      y: number,
      rotZ: number,
      scaleX = 1,
    ) => {
      if (!mesh) return;
      positions.forEach((z, index) => {
        matrixDummy.position.set(x, y, z);
        matrixDummy.rotation.set(-Math.PI / 2, 0, rotZ);
        matrixDummy.scale.set(scaleX, 1, 1);
        matrixDummy.updateMatrix();
        mesh.setMatrixAt(index, matrixDummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      matrixDummy.scale.set(1, 1, 1);
    };

    setInstances(majorStripeRef.current, majorStripes, 0, 0.032, -0.08, 1.18);
    setInstances(minorStripeRef.current, minorStripes, 0, 0.033, 0.05, 0.92);
  }, [majorStripes, matrixDummy, minorStripes, stripes]);

  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;
    group.position.z -= visualSpeed * delta;
    if (group.position.z <= -spacing) group.position.z += spacing;
  });

  return (
    <group ref={groupRef}>
      <instancedMesh ref={majorStripeRef} args={[undefined, undefined, majorStripes.length]}>
        <planeGeometry args={[2200 * WORLD_SCALE, 44 * WORLD_SCALE]} />
        <meshBasicMaterial color="#2f5128" transparent opacity={0.20} depthWrite={false} />
      </instancedMesh>
      <instancedMesh ref={minorStripeRef} args={[undefined, undefined, minorStripes.length]}>
        <planeGeometry args={[1600 * WORLD_SCALE, 22 * WORLD_SCALE]} />
        <meshBasicMaterial color="#8a7c48" transparent opacity={0.13} depthWrite={false} />
      </instancedMesh>
    </group>
  );
}

function DistanceMarkers() {
  // Pole pairs every 500 m along the runway / mission-area axis give parallax cues while
  // airborne. They stop the cockpit view from feeling “floaty”. Range is
  // chosen so markers extend behind the threshold (negative n) and well past
  // the far end of the runway.
  const positions = useMemo(() => {
    const arr: number[] = [];
    for (let n = -500; n <= MISSION_AREA_SIZE_M; n += 500) arr.push(n);
    return arr;
  }, []);
  const halfWidth = 5200 * WORLD_SCALE;
  return (
    <group>
      {positions.map((n) => (
        <group key={`m-${n}`}>
          <mesh position={[-halfWidth, 1.6 * WORLD_SCALE, n * WORLD_SCALE]} castShadow>
            <cylinderGeometry args={[0.04, 0.04, 3.2 * WORLD_SCALE, 6]} />
            <meshStandardMaterial color="#b0bec5" />
          </mesh>
          <mesh position={[halfWidth, 1.6 * WORLD_SCALE, n * WORLD_SCALE]} castShadow>
            <cylinderGeometry args={[0.04, 0.04, 3.2 * WORLD_SCALE, 6]} />
            <meshStandardMaterial color="#b0bec5" />
          </mesh>
        </group>
      ))}
    </group>
  );
}

type ReferenceSample = { n: number; e: number; h: number };
const TRACE_MAX_POINTS = 260;

function writeInertialToAircraftScene(
  sample: ReferenceSample,
  pose: SmoothedFlightPose,
  target: Float32Array,
  offset: number,
) {
  const psi = THREE.MathUtils.degToRad(normalizeDegrees(pose.yaw));
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  const dN = sample.n - pose.pn;
  const dE = sample.e - pose.pe;
  const bodyX = dN * c + dE * s;
  const bodyY = -dN * s + dE * c;
  target[offset] = bodyY * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN;
  target[offset + 1] = (sample.h - pose.altitude) * ALTITUDE_VISUAL_SCALE - AIRCRAFT_WHEEL_OFFSET;
  target[offset + 2] = bodyX * WORLD_SCALE * HORIZONTAL_MOTION_VISUAL_GAIN;
}

function StableTraceLine({
  color,
  history,
  latest,
  mode,
  opacity,
}: {
  color: string;
  history: TelemetryPoint[];
  latest?: TelemetryPoint;
  mode: 'actual' | 'reference';
  opacity: number;
}) {
  const positionsRef = useRef(new Float32Array(TRACE_MAX_POINTS * 3));
  const poseRef = useRef<SmoothedFlightPose>(poseFromTelemetry(latest));
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(positionsRef.current, 3);
    attribute.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('position', attribute);
    geom.setDrawRange(0, 0);
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 20000);
    return geom;
  }, []);
  const material = useMemo(() => new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    depthTest: true,
  }), [color, opacity]);
  const line = useMemo(() => new THREE.Line(geometry, material), [geometry, material]);

  const updateGeometry = (pose: SmoothedFlightPose) => {
    if (!latest) {
      geometry.setDrawRange(0, 0);
      return;
    }
    const recent = history.slice(-TRACE_MAX_POINTS);
    const positions = positionsRef.current;
    let count = 0;
    recent.forEach((point) => {
      const sample = mode === 'reference'
        ? {
            n: point.referenceForwardPosition ?? point.forwardPosition,
            e: point.referenceEastPosition ?? point.eastPosition,
            h: point.referenceAltitude ?? point.targetAltitude ?? point.altitude,
          }
        : {
            n: point.forwardPosition,
            e: point.eastPosition,
            h: point.altitude,
          };
      writeInertialToAircraftScene(sample, pose, positions, count * 3);
      count += 1;
    });
    const position = geometry.getAttribute('position') as THREE.BufferAttribute;
    position.needsUpdate = true;
    geometry.setDrawRange(0, count);
  };

  useFrame((_, delta) => {
    if (!latest || history.length < 2) {
      geometry.setDrawRange(0, 0);
      return;
    }
    const targetPose = poseFromTelemetry(latest);
    if (useSimulationStore.getState().experiment.time <= 0.12) {
      poseRef.current = targetPose;
    } else {
      smoothPoseTowards(poseRef.current, targetPose, delta, latest.trajectoryProfile === 'fight_mode' ? 5.2 : 6.4);
    }
    updateGeometry(poseRef.current);
  });

  useLayoutEffect(() => {
    if (!latest) {
      geometry.setDrawRange(0, 0);
      return;
    }
    poseRef.current = poseFromTelemetry(latest);
    updateGeometry(poseRef.current);
  }, [geometry, history, latest, mode]);

  useEffect(() => () => {
    geometry.dispose();
    material.dispose();
  }, [geometry, material]);

  if (!latest || history.length < 2) return null;
  return <primitive object={line} frustumCulled={false} />;
}

function ReferencePathTrace({ history, latest }: { history: TelemetryPoint[]; latest?: TelemetryPoint }) {
  if (!latest) return null;
  return (
    <group>
      <StableTraceLine history={history} latest={latest} mode="reference" color="#facc15" opacity={0.92} />
      <StableTraceLine history={history} latest={latest} mode="actual" color="#38bdf8" opacity={0.86} />
    </group>
  );
}

function toFinite(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizeDegrees(value: number) {
  return ((value + 180) % 360 + 360) % 360 - 180;
}

function headingErrorDegrees(currentDeg: number, targetDeg: number) {
  return normalizeDegrees(targetDeg - currentDeg);
}

const EFFECT_OPTIONS: { mode: BackendEffectMode; label: string; description: string }[] = [
  { mode: 'calm', label: 'Calm', description: 'No wind, no turbulence' },
  { mode: 'headwind', label: 'Headwind', description: 'Nose wind, higher relative Va' },
  { mode: 'tailwind', label: 'Tailwind', description: 'Tail wind, lower relative Va' },
  { mode: 'crosswind', label: 'Crosswind', description: 'Lateral steady wind' },
  { mode: 'turbulence', label: 'Turbulence', description: 'Random vertical gusts' },
  { mode: 'gust', label: 'Gust', description: 'Step-like body vertical gust' },
  { mode: 'q_learning', label: 'Q-learning', description: 'Bounded tabular guidance-residual experiment; fixed inner-loop autopilot remains active' },
  { mode: 'sharq_hjb', label: 'SHARQ-HJB', description: 'Hamilton-Jacobi/shielded Q residual experiment; fixed inner-loop autopilot remains active' },
];

const TRAJECTORY_OPTIONS: { profile: BackendTrajectoryProfile; label: string; description: string }[] = [
  { profile: 'fight_mode', label: 'Fight Mode', description: 'Cinematic dogfight reference with high-energy S-turns, climbs and visual inverted rolls' },
  { profile: 'runway_takeoff_accel_200', label: '200m Circle', description: 'Runway takeoff, climb to 200 m, 200 m straight, then 200 m circle' },
  { profile: 'takeoff_climbout_200', label: 'Climb Circle', description: 'Runway takeoff, 200 m climb, short straight segment and compact circle' },
  { profile: 'high_speed_climb_s_turn_200', label: 'Tight Circle', description: 'Short takeoff/climb mission with visible 200 m reference circle' },
  { profile: 'loiter_orbit', label: 'Loiter 200m', description: 'Runway takeoff and sustained 200 m loiter circle inside green area' },
  { profile: 'straight_climb_altitude_hold', label: 'Hold Circle', description: 'Takeoff, 200 m straight reference, then 200 m circular route' },
  { profile: 'figure_eight', label: '8 Compact', description: 'Takeoff, climb and compact circular reference visualization' },
  { profile: 'racetrack', label: 'Race 200m', description: 'Runway takeoff, climb and tight 200 m circular manoeuvre' },
];

function backendCommandForEffect(mode: BackendEffectMode, command = 'configure', profile: BackendTrajectoryProfile = useSimulationStore.getState().backendTrajectoryProfile) {
  const presets: Record<BackendEffectMode, Record<string, number>> = {
    calm: { steady_wind_n: 0, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: 0, turbulence_std: 0 },
    headwind: { steady_wind_n: -4, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: 0, turbulence_std: 0 },
    tailwind: { steady_wind_n: 2, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: 0, turbulence_std: 0 },
    crosswind: { steady_wind_n: 0, steady_wind_e: 4, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: 0, turbulence_std: 0 },
    turbulence: { steady_wind_n: 0, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: 0, turbulence_std: 0.22 },
    gust: { steady_wind_n: 0, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: -0.2, turbulence_std: 0.05 },
    q_learning: { steady_wind_n: 0, steady_wind_e: 0, steady_wind_d: 0, gust_body_u: 0, gust_body_v: 0, gust_body_w: -0.1, turbulence_std: 0.10 },
    sharq_hjb: { steady_wind_n: 0, steady_wind_e: 0.4, steady_wind_d: 0, gust_body_u: -0.1, gust_body_v: 0.1, gust_body_w: -0.1, turbulence_std: 0.12 },
  };
  const controller =
    mode === 'q_learning' ? 'online_q_learning' :
    mode === 'sharq_hjb' ? 'sharq_hjb' :
    'fixed_matlab_baseline';
  return {
    command,
    scenario: profile,
    profile,
    controller,
    effects: presets[mode],
  };
}

function mapBackendFrameToTelemetry(frame: Record<string, any>): Omit<TelemetryPoint, 'time'> {
  const uav = frame.uav_state ?? {};
  const wing = frame.wing_state ?? {};
  const aero = frame.aero_state ?? {};
  const control = frame.control_state ?? {};
  const metrics = frame.rl_metrics ?? {};
  const reference = frame.reference_state ?? {};
  const position = Array.isArray(uav.position) ? uav.position : [];
  const attitude = Array.isArray(uav.attitude) ? uav.attitude : [];
  const commands = Array.isArray(control.actuator_commands) ? control.actuator_commands : [];
  const ailerons = control.aileron_deflection ?? {};
  const flaps = control.flap_deflection ?? {};
  const spoilers = control.spoiler_deployment ?? {};
  const windDirection = Array.isArray(aero.wind_direction) ? aero.wind_direction : [];
  const windBody = Array.isArray(aero.wind_body) ? aero.wind_body : [];

  return {
    tipDeflection: toFinite(wing.tip_deflection),
    pitchRate: toFinite(uav.pitch_rate_deg_s ?? uav.pitchRateDegS),
    controlEffort: toFinite(control.control_energy),
    reward: toFinite(metrics.reward),
    strain: toFinite(wing.average_strain),
    forwardPosition: toFinite(position[0]),
    eastPosition: toFinite(position[1]),
    altitude: toFinite(position[2]),
    airspeed: toFinite(uav.airspeed),
    mach: toFinite(uav.mach),
    rollAngle: toFinite(attitude[0]),
    pitchAngle: toFinite(attitude[1]),
    yawAngle: normalizeDegrees(toFinite(attitude[2])),
    angleOfAttack: toFinite(aero.angle_of_attack_deg),
    sideslipAngle: toFinite(aero.sideslip_deg),
    flightPathAngle: toFinite(aero.flight_path_angle_deg),
    loadFactorNz: toFinite(aero.load_factor_nz, 1),
    leftAileron: toFinite(ailerons.left, toFinite(commands[0])),
    rightAileron: toFinite(ailerons.right, toFinite(commands[1])),
    elevator: toFinite(control.elevator_deflection, toFinite(commands[2])),
    rudder: toFinite(control.rudder_deflection, toFinite(commands[3])),
    flapLeft: toFinite(flaps.left),
    flapRight: toFinite(flaps.right),
    leftSpoiler: toFinite(spoilers.left),
    rightSpoiler: toFinite(spoilers.right),
    throttle: toFinite(control.throttle, toFinite(commands[4]) / 100),
    flightMode: String(uav.flight_mode ?? control.flight_phase ?? ''),
    autopilot: String(reference.autopilot ?? frame.controller ?? ''),
    trajectoryProfile: String(frame.profile ?? reference.trajectory_profile ?? ''),
    targetAltitude: toFinite(reference.target_altitude),
    altitudeError: toFinite(reference.altitude_error),
    targetAirspeed: toFinite(reference.target_airspeed),
    airspeedError: toFinite(reference.airspeed_error),
    distanceError: toFinite(reference.distance_error),
    distanceToReference: toFinite(reference.distance_to_reference),
    targetPitch: toFinite(reference.target_pitch_deg),
    targetRoll: toFinite(reference.target_roll_deg),
    targetHeading: normalizeDegrees(toFinite(reference.target_heading_deg)),
    targetLateralOffset: toFinite(reference.target_lateral_offset),
    referenceForwardPosition: toFinite(reference.reference_position_n),
    referenceEastPosition: toFinite(reference.reference_position_e),
    referenceAltitude: toFinite(reference.reference_altitude, toFinite(reference.target_altitude)),
    horizontalReferenceError: toFinite(reference.horizontal_reference_error),
    missionAreaSize: toFinite(reference.mission_area_size_m, MISSION_AREA_SIZE_M),
    circleDiameter: toFinite(reference.circle_diameter_m),
    circleRadius: toFinite(reference.circle_radius_m),
    circleAirspeed: toFinite(reference.circle_airspeed_mps),
    circleDirection: toFinite(reference.circle_direction, 1),
    circleStartTime: toFinite(reference.circle_start_time_s, 0),
    windSpeed: toFinite(aero.wind_speed),
    windDirectionX: toFinite(windDirection[0]),
    windDirectionY: toFinite(windDirection[1]),
    windDirectionZ: toFinite(windDirection[2]),
    windBodyX: toFinite(windBody[0]),
    windBodyY: toFinite(windBody[1]),
    windBodyZ: toFinite(windBody[2]),
    turbulenceIntensity: toFinite(aero.turbulence_intensity),
    gustLevel: toFinite(aero.gust_level),
  };
}

function BackendTelemetryBridge() {
  const effectMode = useSimulationStore(state => state.backendEffectMode);
  const trajectoryProfile = useSimulationStore(state => state.backendTrajectoryProfile);
  const resetNonce = useSimulationStore(state => state.backendResetNonce);
  const socketRef = useRef<WebSocket | null>(null);
  const lastUiTelemetryTimestampRef = useRef(-Infinity);
  const effectModeRef = useRef(effectMode);
  const trajectoryProfileRef = useRef(trajectoryProfile);

  useEffect(() => {
    effectModeRef.current = effectMode;
    safeWebSocketSend(socketRef.current, backendCommandForEffect(effectMode, 'configure', trajectoryProfileRef.current));
  }, [effectMode]);

  useEffect(() => {
    trajectoryProfileRef.current = trajectoryProfile;
    lastUiTelemetryTimestampRef.current = -Infinity;
    useSimulationStore.getState().resetTelemetry();
    safeWebSocketSend(socketRef.current, backendCommandForEffect(effectModeRef.current, 'reset', trajectoryProfile));
  }, [trajectoryProfile]);

  useEffect(() => {
    if (resetNonce === 0) return;
    lastUiTelemetryTimestampRef.current = -Infinity;
    useSimulationStore.getState().resetTelemetry();
    safeWebSocketSend(socketRef.current, backendCommandForEffect(effectModeRef.current, 'reset', trajectoryProfileRef.current));
  }, [resetNonce]);

  useEffect(() => {
    const viteEnv = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env;
    const wsUrl = viteEnv?.VITE_BACKEND_WS_URL ?? viteEnv?.VITE_UAV_BACKEND_WS ?? 'ws://localhost:8000/ws/uav-digital-twin';
    let closed = false;
    let socket: WebSocket | undefined;

    const connect = () => {
      if (closed) return;
      socket = new WebSocket(wsUrl);
      socketRef.current = socket;
      socket.onopen = () => {
        const store = useSimulationStore.getState();
        lastUiTelemetryTimestampRef.current = -Infinity;
        store.resetTelemetry();
        store.setBackendActive(true);
        safeWebSocketSend(socket, backendCommandForEffect(effectModeRef.current, 'start', trajectoryProfileRef.current));
      };
      socket.onmessage = (event) => {
        let frame: Record<string, any>;
        try {
          frame = JSON.parse(event.data) as Record<string, any>;
        } catch {
          // Backend should always emit valid JSON; if not, drop the frame
          // instead of crashing the listener and tearing down the bridge.
          return;
        }
        if (frame.type !== 'simulation_frame' || toFinite(frame.schema_version, -1) !== BACKEND_FRAME_SCHEMA_VERSION) return;
        const store = useSimulationStore.getState();
        if (!store.backendActive) store.setBackendActive(true);
        const timestamp = toFinite(frame.timestamp, store.experiment.time);
        const mode = String((frame.uav_state as Record<string, unknown> | undefined)?.flight_mode ?? '');
        const isTerminalFrame = mode === 'landed' || mode === 'failed';
        if (!isTerminalFrame && timestamp - lastUiTelemetryTimestampRef.current < UI_TELEMETRY_SAMPLE_INTERVAL_S) return;
        lastUiTelemetryTimestampRef.current = timestamp;
        store.setTime(timestamp);
        store.addTelemetry(mapBackendFrameToTelemetry(frame));
      };
      socket.onclose = () => {
        if (socketRef.current === socket) socketRef.current = null;
        useSimulationStore.getState().setBackendActive(false);
        if (!closed) window.setTimeout(connect, 2000);
      };
      socket.onerror = () => {
        useSimulationStore.getState().setBackendActive(false);
        socket?.close();
      };
    };

    connect();
    return () => {
      closed = true;
      useSimulationStore.getState().setBackendActive(false);
      socketRef.current = null;
      socket?.close();
    };
  }, []);

  return null;
}


// ============================================================================
// Backend lifecycle (Start / Stop / Reset)
// ============================================================================

type BackendProcessStatus = {
  state: 'stopped' | 'starting' | 'running' | 'stopping' | 'error' | 'unavailable';
  pid?: number | null;
  message?: string;
  lastLog?: string;
};

async function backendApi(path: 'start' | 'stop' | 'status'): Promise<BackendProcessStatus> {
  const response = await fetch(`/api/backend/${path}`, { method: path === 'status' ? 'GET' : 'POST' });
  if (!response.ok) throw new Error(`Backend API ${response.status}`);
  return response.json() as Promise<BackendProcessStatus>;
}

function useBackendLifecycle() {
  const backendActive = useSimulationStore(state => state.backendActive);
  const reset = useSimulationStore(state => state.requestBackendReset);
  const [status, setStatus] = useState<BackendProcessStatus>({ state: 'stopped', message: 'Not checked yet' });
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      setStatus(await backendApi('status'));
    } catch (error) {
      setStatus({ state: 'unavailable', message: error instanceof Error ? error.message : 'Vite backend API unavailable' });
    }
  };

  const command = async (action: 'start' | 'stop') => {
    setBusy(true);
    try {
      setStatus(await backendApi(action));
      if (action === 'start') window.setTimeout(refresh, 900);
    } catch (error) {
      setStatus({ state: 'unavailable', message: error instanceof Error ? error.message : 'Backend API unavailable' });
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    refresh();
    const refreshIfVisible = () => {
      if (!document.hidden) void refresh();
    };
    const refreshOnVisible = () => {
      if (!document.hidden) void refresh();
    };
    const id = window.setInterval(refreshIfVisible, 2500);
    const stopOnClose = () => {
      navigator.sendBeacon?.('/api/backend/stop', new Blob(['{}'], { type: 'application/json' }));
    };
    document.addEventListener('visibilitychange', refreshOnVisible);
    window.addEventListener('beforeunload', stopOnClose);
    return () => {
      window.clearInterval(id);
      document.removeEventListener('visibilitychange', refreshOnVisible);
      window.removeEventListener('beforeunload', stopOnClose);
    };
  }, []);

  return { status, busy, backendActive, reset, command };
}

type BackendLifecycle = ReturnType<typeof useBackendLifecycle>;

// ============================================================================
// Header — title, live time, connection pill, backend Start/Stop/Reset
// ============================================================================

function StatusPill({ active, state }: { active: boolean; state: string }) {
  const tone = active
    ? 'border-emerald-400/60 bg-emerald-500/15 text-emerald-200'
    : state === 'error' || state === 'unavailable'
      ? 'border-rose-400/60 bg-rose-500/15 text-rose-200'
      : state === 'starting' || state === 'stopping'
        ? 'border-amber-400/60 bg-amber-500/15 text-amber-200'
        : 'border-slate-600 bg-slate-800/70 text-slate-300';
  const dot = active
    ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]'
    : state === 'error' || state === 'unavailable'
      ? 'bg-rose-400'
      : state === 'starting' || state === 'stopping'
        ? 'bg-amber-400 animate-pulse'
        : 'bg-slate-500';
  const label = active ? 'LIVE TELEMETRY' : state.toUpperCase();
  return (
    <div className={`flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[10px] uppercase tracking-widest ${tone}`}>
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {label}
    </div>
  );
}

function TopBar({ lifecycle }: { lifecycle: BackendLifecycle }) {
  const { status, busy, backendActive, reset, command } = lifecycle;
  const time = useSimulationStore(state => state.experiment.time);
  const profile = useSimulationStore(state => state.backendTrajectoryProfile);
  const uiMode = useSimulationStore(state => state.uiMode);
  const toggleUiMode = useSimulationStore(state => state.toggleUiMode);
  const running = status.state === 'running' || backendActive;
  return (
    <header className={`z-20 flex items-center justify-between gap-4 border-b border-cyan-500/20 bg-gradient-to-r from-slate-950 via-slate-900/95 to-slate-950 px-6 shadow-[0_2px_24px_rgba(8,145,178,0.18)] transition-[padding] ${uiMode === 'cinematic' ? 'py-2' : 'py-3'}`}>
      <div className="flex items-center gap-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-cyan-400/40 bg-cyan-500/10 text-cyan-300 shadow-inner">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M12 3l9 12h-6l-3 6-3-6H3z" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="leading-tight">
          <div className="font-mono text-[10px] uppercase tracking-[0.35em] text-cyan-300/80">PythaLab Avionics</div>
          <div className="font-sans text-base font-semibold text-slate-100">UAV Flight Console <span className="text-slate-500">·</span> SAAB Gripen Digital Twin</div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="hidden flex-col items-end font-mono text-[10px] uppercase text-slate-400 sm:flex">
          <span>Mission Clock</span>
          <span className="font-sans text-lg font-medium text-slate-100 tabular-nums">{time.toFixed(2)} s</span>
        </div>
        <div className="hidden flex-col items-end font-mono text-[10px] uppercase text-slate-400 lg:flex">
          <span>Profile</span>
          <span className="font-sans text-sm font-medium text-emerald-200">{profile.replace(/_/g, ' ')}</span>
        </div>
        <StatusPill active={backendActive} state={status.state} />
        <div className="flex items-center gap-1.5">
          <button
            onClick={toggleUiMode}
            className={`rounded-md border px-4 py-1.5 font-mono text-xs uppercase tracking-wider transition ${
              uiMode === 'cinematic'
                ? 'border-amber-300/60 bg-amber-400/15 text-amber-100 shadow-[0_0_14px_rgba(251,191,36,0.18)] hover:bg-amber-400/25'
                : 'border-slate-500/60 bg-slate-700/35 text-slate-100 hover:bg-slate-700/55'
            }`}
            title={uiMode === 'cinematic' ? 'Show full operator panels' : 'Hide engineering panels for video recording'}
          >
            {uiMode === 'cinematic' ? 'Cinematic' : 'Operator'}
          </button>
          <button
            disabled={busy || running}
            onClick={() => command('start')}
            className="rounded-md border border-emerald-400/50 bg-emerald-500/15 px-4 py-1.5 font-mono text-xs uppercase tracking-wider text-emerald-200 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Start
          </button>
          <button
            disabled={busy || (!running && status.state !== 'starting')}
            onClick={() => command('stop')}
            className="rounded-md border border-rose-400/50 bg-rose-500/15 px-4 py-1.5 font-mono text-xs uppercase tracking-wider text-rose-200 transition hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Stop
          </button>
          <button
            onClick={reset}
            className="rounded-md border border-cyan-400/50 bg-cyan-500/15 px-4 py-1.5 font-mono text-xs uppercase tracking-wider text-cyan-200 transition hover:bg-cyan-500/25"
          >
            Reset
          </button>
        </div>
      </div>
    </header>
  );
}

// ============================================================================
// Left instrument cluster — PFD, compass, altitude/airspeed tapes
// ============================================================================

function AttitudeIndicator({ pitch, roll }: { pitch: number; roll: number }) {
  // Visual approximation of an artificial horizon. Sky/ground bisected,
  // pitched up/down by deg, rotated by roll. Pitch ladder shows 10° gradations.
  const pitchOffset = clamp(pitch, -45, 45) * 2.0; // px per deg, container is 180px tall
  return (
    <div className="relative h-44 w-44 overflow-hidden rounded-full border-2 border-cyan-400/30 bg-slate-950 shadow-[0_0_24px_rgba(8,145,178,0.25)]">
      <div className="absolute inset-0" style={{ transform: `rotate(${-roll}deg)` }}>
        <div className="absolute inset-0" style={{ transform: `translateY(${pitchOffset}px)` }}>
          <div className="absolute left-1/2 top-0 h-1/2 w-[300%] -translate-x-1/2 bg-gradient-to-b from-sky-700 via-sky-600 to-sky-500" />
          <div className="absolute left-1/2 top-1/2 h-1/2 w-[300%] -translate-x-1/2 bg-gradient-to-b from-amber-800 via-amber-900 to-stone-900" />
          <div className="absolute left-1/2 top-1/2 h-0.5 w-[300%] -translate-x-1/2 -translate-y-1/2 bg-white" />
          {[-30, -20, -10, 10, 20, 30].map(p => (
            <div
              key={p}
              className="absolute left-1/2 -translate-x-1/2 text-[8px] font-mono text-white"
              style={{ top: `calc(50% - ${p * 2.0}px)` }}
            >
              <div className="flex items-center gap-1">
                <span className="w-2">{Math.abs(p)}</span>
                <span className={`block h-0.5 ${Math.abs(p) === 10 ? 'w-6' : 'w-10'} bg-white/80`} />
                <span className="w-2">{Math.abs(p)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      {/* Fixed aircraft symbol */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <svg viewBox="0 0 120 40" className="h-10 w-32" fill="none" stroke="#fbbf24" strokeWidth="2.4" strokeLinecap="round">
          <line x1="10" y1="20" x2="40" y2="20" />
          <line x1="80" y1="20" x2="110" y2="20" />
          <line x1="40" y1="20" x2="48" y2="28" />
          <line x1="80" y1="20" x2="72" y2="28" />
          <circle cx="60" cy="20" r="2" fill="#fbbf24" />
        </svg>
      </div>
      {/* Roll scale */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-1.5 h-2 w-0.5 -translate-x-1/2 bg-amber-400" />
      </div>
    </div>
  );
}

function CompassRose({ heading }: { heading: number }) {
  const normalized = ((heading % 360) + 360) % 360;
  return (
    <div className="relative h-32 w-32 overflow-hidden rounded-full border-2 border-cyan-400/30 bg-slate-950">
      <div className="absolute inset-0 flex items-center justify-center" style={{ transform: `rotate(${-normalized}deg)` }}>
        {Array.from({ length: 36 }).map((_, i) => {
          const deg = i * 10;
          const isCardinal = deg % 90 === 0;
          const label = deg === 0 ? 'N' : deg === 90 ? 'E' : deg === 180 ? 'S' : deg === 270 ? 'W' : '';
          return (
            <div
              key={i}
              className="absolute inset-0 flex items-start justify-center pt-1.5 font-mono text-[8px] text-slate-300"
              style={{ transform: `rotate(${deg}deg)` }}
            >
              <div className="flex flex-col items-center">
                <span className={`block ${isCardinal ? 'h-3 w-0.5 bg-cyan-300' : 'h-1.5 w-0.5 bg-slate-500'}`} />
                {label && <span className="mt-0.5 font-bold text-cyan-200">{label}</span>}
              </div>
            </div>
          );
        })}
      </div>
      <div className="pointer-events-none absolute inset-0 flex items-start justify-center">
        <svg viewBox="0 0 10 10" className="mt-0 h-3 w-3 text-amber-400" fill="currentColor"><polygon points="5,0 8,7 5,5 2,7" /></svg>
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-1 text-center font-mono text-[10px] tabular-nums text-cyan-200">
        {normalized.toFixed(0)}°
      </div>
    </div>
  );
}

function Tape({ label, value, unit, target, min, max, fmt = (v: number) => v.toFixed(1), tickStep = 10, accent = '#38bdf8' }: {
  label: string;
  value: number;
  unit: string;
  target?: number;
  min: number;
  max: number;
  fmt?: (v: number) => string;
  tickStep?: number;
  accent?: string;
}) {
  const range = max - min;
  const ticks = Math.floor(range / tickStep) + 1;
  const valuePct = clamp(((value - min) / range) * 100, 0, 100);
  const targetPct = target !== undefined && Number.isFinite(target) ? clamp(((target - min) / range) * 100, 0, 100) : null;
  return (
    <div className="flex h-44 w-20 flex-col rounded-md border border-slate-700/80 bg-slate-950/85 p-1.5 font-mono text-[9px] uppercase text-slate-300">
      <div className="text-center text-cyan-300">{label}</div>
      <div className="relative my-1 flex-1 overflow-hidden rounded border border-slate-800 bg-slate-900/70">
        {Array.from({ length: ticks }).map((_, i) => (
          <div
            key={i}
            className="absolute left-0 right-0 border-t border-slate-700/60"
            style={{ bottom: `${(i / (ticks - 1)) * 100}%` }}
          />
        ))}
        {targetPct !== null && (
          <div className="absolute left-0 right-0 h-0.5 bg-emerald-300" style={{ bottom: `${targetPct}%` }}>
            <span className="absolute -right-0.5 -top-1 h-2 w-2 rotate-45 bg-emerald-300" />
          </div>
        )}
        <div className="absolute left-0 right-0 h-0.5" style={{ bottom: `${valuePct}%`, backgroundColor: accent, boxShadow: `0 0 8px ${accent}` }} />
        <div className="absolute left-1 right-1 bottom-1/2 translate-y-1/2 text-center font-sans text-base font-semibold tabular-nums" style={{ color: accent }}>
          {fmt(value)}
        </div>
      </div>
      <div className="text-center text-[8px] text-slate-500">{unit}</div>
    </div>
  );
}

function SurfaceBar({ label, value, min, max, unit = '°' }: { label: string; value: number; min: number; max: number; unit?: string }) {
  const range = max - min;
  const zeroPct = ((0 - min) / range) * 100;
  const valuePct = clamp(((value - min) / range) * 100, 0, 100);
  const isPositive = value >= 0;
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] uppercase">
      <span className="w-14 text-slate-400">{label}</span>
      <div className="relative h-2 flex-1 rounded-full bg-slate-800">
        <div className="absolute top-0 bottom-0 w-px bg-slate-500" style={{ left: `${zeroPct}%` }} />
        <div
          className="absolute top-0 bottom-0 rounded-full"
          style={{
            left: isPositive ? `${zeroPct}%` : `${valuePct}%`,
            width: `${Math.abs(valuePct - zeroPct)}%`,
            backgroundColor: isPositive ? '#38bdf8' : '#fb923c',
          }}
        />
      </div>
      <span className="w-12 text-right tabular-nums text-slate-100">{value.toFixed(1)}{unit}</span>
    </div>
  );
}

function ThrottleGauge({ value }: { value: number }) {
  const pct = clamp(value * 100, 0, 100);
  const color = pct > 90 ? '#ef4444' : pct > 70 ? '#f97316' : '#22c55e';
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] uppercase">
      <span className="w-14 text-slate-400">Throttle</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full transition-all" style={{ width: `${pct}%`, backgroundColor: color, boxShadow: `0 0 6px ${color}` }} />
      </div>
      <span className="w-12 text-right tabular-nums text-slate-100">{pct.toFixed(0)}%</span>
    </div>
  );
}

function LeftInstrumentCluster({ latest }: { latest?: TelemetryPoint }) {
  const pitch = latest?.pitchAngle ?? 0;
  const roll = latest?.rollAngle ?? 0;
  const yaw = latest?.yawAngle ?? 0;
  const targetHeading = latest?.targetHeading ?? yaw;
  const airspeed = latest?.airspeed ?? 0;
  const altitude = latest?.altitude ?? 0;
  const targetAltitude = latest?.targetAltitude;
  const targetAirspeed = latest?.targetAirspeed;
  const altMin = Math.max(0, Math.floor((altitude - 60) / 20) * 20);
  return (
    <aside className="z-10 flex min-h-0 w-[18rem] shrink-0 flex-col overflow-hidden border-r border-cyan-400/20 bg-gradient-to-b from-slate-950/95 via-slate-950/88 to-slate-900/92 shadow-[12px_0_38px_rgba(2,6,23,0.38)] backdrop-blur">
      <div className="border-b border-cyan-400/15 px-3 py-2.5">
        <div className="font-mono text-[9px] uppercase tracking-[0.34em] text-cyan-300/80">Operator Tab</div>
        <div className="mt-0.5 font-sans text-sm font-semibold text-slate-100">Flight Instruments</div>
      </div>
      <div className="operator-scrollbar min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <div className="rounded-xl border border-cyan-400/20 bg-slate-900/55 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-wider text-cyan-300">Primary Flight Display</span>
            <span className="font-mono text-[9px] uppercase text-slate-500">PFD</span>
          </div>
          <div className="flex flex-col items-center gap-2">
            <AttitudeIndicator pitch={pitch} roll={roll} />
            <CompassRose heading={yaw} />
            <div className="grid w-full grid-cols-3 gap-1 font-mono text-[9px] uppercase text-slate-400">
              <div className="rounded bg-slate-900/60 p-1 text-center">
                <div>HDG</div>
                <div className="font-sans text-sm font-semibold tabular-nums text-slate-100">{yaw.toFixed(0)}°</div>
              </div>
              <div className="rounded bg-slate-900/60 p-1 text-center">
                <div>TGT</div>
                <div className="font-sans text-sm font-semibold tabular-nums text-emerald-200">{targetHeading.toFixed(0)}°</div>
              </div>
              <div className="rounded bg-slate-900/60 p-1 text-center">
                <div>ROLL</div>
                <div className="font-sans text-sm font-semibold tabular-nums text-slate-100">{roll.toFixed(1)}°</div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-cyan-400/20 bg-slate-900/55 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-cyan-300">Tapes</div>
          <div className="flex justify-center gap-2">
            <Tape label="Spd" value={airspeed} unit="m/s" target={targetAirspeed} min={0} max={220} tickStep={20} accent="#fbbf24" fmt={(v) => v.toFixed(1)} />
            <Tape label="Alt" value={altitude} unit="m" target={targetAltitude} min={altMin} max={altMin + 120} tickStep={20} accent="#38bdf8" fmt={(v) => v.toFixed(0)} />
          </div>
        </div>

        <div className="rounded-xl border border-cyan-400/20 bg-slate-900/55 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-cyan-300">Control Surfaces</div>
          <div className="space-y-1.5">
            <SurfaceBar label="L Aileron" value={latest?.leftAileron ?? 0} min={-25} max={25} />
            <SurfaceBar label="R Aileron" value={latest?.rightAileron ?? 0} min={-25} max={25} />
            <SurfaceBar label="Elevator" value={latest?.elevator ?? 0} min={-25} max={25} />
            <SurfaceBar label="Rudder" value={latest?.rudder ?? 0} min={-30} max={30} />
            <ThrottleGauge value={latest?.throttle ?? 0} />
          </div>
        </div>
      </div>
    </aside>
  );
}

// ============================================================================
// Right control column — Mission Profile, Wind & Effects, Telemetry summary
// ============================================================================

function PanelCard({ title, badge, children }: { title: string; badge?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-cyan-400/20 bg-slate-900/55 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <header className="mb-2 flex items-center justify-between border-b border-slate-700/70 pb-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-cyan-300">{title}</span>
        {badge}
      </header>
      {children}
    </section>
  );
}

function MissionProfilePanel() {
  const profile = useSimulationStore(state => state.backendTrajectoryProfile);
  const setProfile = useSimulationStore(state => state.setBackendTrajectoryProfile);
  return (
    <PanelCard title="Mission Profile">
      <div className="grid grid-cols-2 gap-2">
        {TRAJECTORY_OPTIONS.map((option) => {
          const active = profile === option.profile;
          return (
            <button
              key={option.profile}
              onClick={() => setProfile(option.profile)}
              title={option.description}
              className={`flex min-h-[3.35rem] flex-col items-start justify-center gap-0.5 rounded-lg border px-2.5 py-2 text-left font-mono text-[10px] uppercase transition ${
                active
                  ? 'border-emerald-300 bg-emerald-500/20 text-emerald-100 shadow-[0_0_14px_rgba(52,211,153,0.24)]'
                  : 'border-slate-700/80 bg-slate-950/70 text-slate-300 hover:border-emerald-500/60 hover:bg-slate-900'
              }`}
            >
              <span className="font-sans font-medium">{option.label}</span>
              <span className="line-clamp-2 text-[8px] normal-case leading-tight text-slate-500">{option.description.slice(0, 54)}</span>
            </button>
          );
        })}
      </div>
    </PanelCard>
  );
}

function WindEffectsPanel() {
  const mode = useSimulationStore(state => state.backendEffectMode);
  const setMode = useSimulationStore(state => state.setBackendEffectMode);
  return (
    <PanelCard title="Wind & Effects">
      <div className="grid grid-cols-2 gap-2">
        {EFFECT_OPTIONS.map((option) => {
          const active = mode === option.mode;
          return (
            <button
              key={option.mode}
              onClick={() => setMode(option.mode)}
              title={option.description}
              className={`flex min-h-[3.15rem] flex-col items-start justify-center gap-0.5 rounded-lg border px-2.5 py-2 text-left font-mono text-[10px] uppercase transition ${
                active
                  ? 'border-cyan-300 bg-cyan-500/20 text-cyan-100 shadow-[0_0_14px_rgba(34,211,238,0.24)]'
                  : 'border-slate-700/80 bg-slate-950/70 text-slate-300 hover:border-cyan-500/60 hover:bg-slate-900'
              }`}
            >
              <span className="font-sans font-medium">{option.label}</span>
              <span className="line-clamp-2 text-[8px] normal-case leading-tight text-slate-500">{option.description.slice(0, 52)}</span>
            </button>
          );
        })}
      </div>
    </PanelCard>
  );
}

function TelemetrySummary({ latest }: { latest?: TelemetryPoint }) {
  const backendActive = useSimulationStore(state => state.backendActive);
  const envelope = envelopeStatus(latest);
  const altErr = latest?.altitudeError;
  const spdErr = latest?.airspeedError;
  const distErr = latest?.distanceError;
  const wind = latest?.windSpeed ?? 0;

  const items: { label: string; value: string; tone?: string }[] = [
    { label: 'Mach', value: `M ${(latest?.mach ?? 0).toFixed(3)}` },
    { label: 'AoA α', value: `${(latest?.angleOfAttack ?? 0).toFixed(1)}°` },
    { label: 'Sideslip β', value: `${(latest?.sideslipAngle ?? 0).toFixed(1)}°` },
    { label: 'FPA γ', value: `${(latest?.flightPathAngle ?? 0).toFixed(1)}°` },
    { label: 'Nz', value: `${(latest?.loadFactorNz ?? 1).toFixed(2)} g` },
    { label: 'Pitch rate', value: `${(latest?.pitchRate ?? 0).toFixed(1)} °/s` },
    { label: 'Alt error', value: altErr === undefined ? '—' : `${altErr.toFixed(1)} m`, tone: Math.abs(altErr ?? 0) > 10 ? 'text-amber-300' : 'text-slate-100' },
    { label: 'Spd error', value: spdErr === undefined ? '—' : `${spdErr.toFixed(1)} m/s`, tone: Math.abs(spdErr ?? 0) > 3 ? 'text-amber-300' : 'text-slate-100' },
    { label: 'Dist error', value: distErr === undefined ? '—' : `${distErr.toFixed(1)} m` },
    { label: 'Wind |W|', value: `${wind.toFixed(2)} m/s` },
    { label: 'Autopilot', value: backendActive ? (latest?.autopilot || 'fixed_matlab') : 'offline' },
    { label: 'Mode', value: backendActive ? (latest?.flightMode ?? '—') : 'offline' },
  ];
  return (
    <PanelCard
      title="Live Telemetry"
      badge={
        <span className={`font-mono text-[9px] uppercase tracking-wider ${
          envelope.severity === 'nominal' ? 'text-emerald-300' : envelope.severity === 'caution' ? 'text-amber-300' : 'text-rose-300'
        }`}>
          ENV · {envelope.severity}
        </span>
      }
    >
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono text-[10px] uppercase">
        {items.map(item => (
          <div key={item.label} className="flex flex-col">
            <span className="text-[9px] text-slate-500">{item.label}</span>
            <span className={`font-sans text-sm font-medium tabular-nums ${item.tone ?? 'text-slate-100'}`}>{item.value}</span>
          </div>
        ))}
      </div>
      {envelope.messages.length > 0 && (
        <div className="mt-2 rounded border border-amber-400/40 bg-amber-500/10 p-2 font-mono text-[9px] uppercase text-amber-200">
          {envelope.messages.join(' · ')}
        </div>
      )}
    </PanelCard>
  );
}

function BackendDiagnosticsPanel({ lifecycle }: { lifecycle: BackendLifecycle }) {
  const { status, backendActive } = lifecycle;
  return (
    <PanelCard
      title="Backend Diagnostics"
      badge={<span className="font-mono text-[9px] uppercase text-slate-500">{backendActive ? 'WS open' : 'offline'}</span>}
    >
      <div className="space-y-1.5 font-mono text-[10px] uppercase">
        <div className="flex justify-between">
          <span className="text-slate-500">Process</span>
          <span className="text-slate-100">{status.state}{status.pid ? ` · pid ${status.pid}` : ''}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Health</span>
          <span className={backendActive ? 'text-emerald-300' : 'text-slate-400'}>{backendActive ? 'streaming' : 'no frames'}</span>
        </div>
        <div className="rounded border border-slate-700 bg-slate-950/70 p-1.5 text-[9px] normal-case leading-snug text-slate-300">
          {status.message ?? '—'}
        </div>
        {status.state === 'unavailable' && (
          <div className="rounded border border-amber-500/40 bg-amber-500/10 p-1.5 text-[9px] normal-case text-amber-100">
            Start button only works inside the Vite dev server. Fallback: <span className="font-mono">cd backend && ../.venv/bin/python -m uavsim.server</span>
          </div>
        )}
      </div>
    </PanelCard>
  );
}

type OperatorRightTab = 'mission' | 'telemetry' | 'system';

function RightControlColumn({ latest, lifecycle }: { latest?: TelemetryPoint; lifecycle: BackendLifecycle }) {
  const [activeTab, setActiveTab] = useState<OperatorRightTab>('mission');
  const tabs: { id: OperatorRightTab; label: string; hint: string }[] = [
    { id: 'mission', label: 'Mission', hint: 'profiles + wind' },
    { id: 'telemetry', label: 'Telemetry', hint: 'live values' },
    { id: 'system', label: 'System', hint: 'backend' },
  ];
  return (
    <aside className="z-10 flex min-h-0 w-[22rem] shrink-0 flex-col overflow-hidden border-l border-cyan-400/20 bg-gradient-to-b from-slate-950/95 via-slate-950/88 to-slate-900/92 shadow-[-12px_0_38px_rgba(2,6,23,0.38)] backdrop-blur">
      <div className="border-b border-cyan-400/15 px-3 py-2.5">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-mono text-[9px] uppercase tracking-[0.34em] text-cyan-300/80">Operator Tab</div>
            <div className="mt-0.5 font-sans text-sm font-semibold text-slate-100">Controls & Telemetry</div>
          </div>
          <span className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[9px] uppercase text-emerald-200">Ready</span>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-1 rounded-xl border border-slate-700/70 bg-slate-950/70 p-1">
          {tabs.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                title={tab.hint}
                className={`rounded-lg px-2 py-2 text-left transition ${
                  active
                    ? 'bg-cyan-400/15 text-cyan-100 shadow-[0_0_18px_rgba(34,211,238,0.16)]'
                    : 'text-slate-400 hover:bg-slate-800/80 hover:text-slate-100'
                }`}
              >
                <div className="font-mono text-[9px] uppercase tracking-wider">{tab.label}</div>
                <div className="mt-0.5 truncate font-mono text-[7px] uppercase text-slate-500">{tab.hint}</div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="operator-scrollbar min-h-0 flex-1 overflow-y-auto p-3">
        {activeTab === 'mission' && (
          <div className="space-y-3">
            <MissionProfilePanel />
            <WindEffectsPanel />
          </div>
        )}
        {activeTab === 'telemetry' && <TelemetrySummary latest={latest} />}
        {activeTab === 'system' && <BackendDiagnosticsPanel lifecycle={lifecycle} />}
      </div>
    </aside>
  );
}


function ChartStrip() {
  const shouldLoadCharts = useSimulationStore(state => state.backendActive || Boolean(state.latestTelemetry));
  if (!shouldLoadCharts) {
    return (
      <div className="z-10 flex h-[120px] items-center justify-center border-t border-cyan-500/20 bg-slate-950/80 backdrop-blur">
        <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
          Awaiting backend telemetry · press <span className="text-emerald-300">Start</span> to launch the autopilot
        </div>
      </div>
    );
  }
  return (
    <Suspense fallback={<div className="z-10 flex h-[120px] items-center justify-center border-t border-cyan-500/20 bg-slate-950/80 font-mono text-[10px] uppercase tracking-wider text-slate-400 backdrop-blur">Loading telemetry charts…</div>}>
      <TrajectoryCharts />
    </Suspense>
  );
}

function missionPhase(latest?: TelemetryPoint) {
  if (!latest) return { label: 'PRE-FLIGHT', detail: 'awaiting backend telemetry', progress: 0 };
  if (latest.trajectoryProfile === 'fight_mode') {
    const fightElapsed = Math.max(0, latest.time - 18);
    if (latest.altitude < 35) return { label: 'FIGHT MODE · TAKEOFF', detail: 'combat climb from runway 36', progress: clamp(latest.airspeed / 90, 0, 0.3) };
    if (latest.altitude < 210) return { label: 'FIGHT MODE · CLIMB', detail: 'arming maneuver corridor', progress: clamp(0.3 + latest.altitude / 240 * 0.35, 0.3, 0.65) };
    return {
      label: 'FIGHT MODE · DOGFIGHT',
      detail: 'S-turn reference · visual inverted roll package',
      progress: clamp(0.68 + (fightElapsed % 24) / 75, 0.68, 1),
    };
  }
  const targetAltitude = latest.targetAltitude ?? 200;
  const circleStart = latest.circleStartTime ?? 0;
  if ((latest.time ?? 0) > 0 && circleStart > 0 && latest.time >= circleStart) {
    return { label: 'ORBIT CAPTURE', detail: `${(latest.circleDiameter ?? 200).toFixed(0)} m reference circle`, progress: 1 };
  }
  if (latest.altitude < 8) {
    return { label: 'RUNWAY ROLL', detail: 'accelerating from threshold 36', progress: clamp(latest.airspeed / 90, 0, 0.32) };
  }
  if (latest.altitude < targetAltitude - 12) {
    return { label: 'CLIMB-OUT', detail: `target ${targetAltitude.toFixed(0)} m`, progress: clamp(0.32 + latest.altitude / targetAltitude * 0.38, 0.32, 0.7) };
  }
  return { label: 'WAYPOINT LEG', detail: 'stabilising before orbit entry', progress: 0.82 };
}

function CinematicTelemetryOverlay({ latest }: { latest?: TelemetryPoint }) {
  const phase = missionPhase(latest);
  const storeTime = useSimulationStore(state => state.experiment.time);
  const time = latest?.time ?? storeTime;
  const airspeed = latest?.airspeed ?? 0;
  const altitude = latest?.altitude ?? 0;
  const trackError = latest?.horizontalReferenceError ?? latest?.distanceError ?? 0;
  const roll = latest?.rollAngle ?? 0;
  const throttle = clamp(latest?.throttle ?? 0, 0, 1);
  const live = useSimulationStore(state => state.backendActive);
  const selectedProfile = useSimulationStore(state => state.backendTrajectoryProfile);
  const fightMode = latest?.trajectoryProfile === 'fight_mode' || selectedProfile === 'fight_mode';

  return (
    <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
      <div className="cinematic-vignette absolute inset-0" />
      <div className="cinematic-film-grain absolute inset-0 opacity-45" />
      <div className="absolute left-0 right-0 top-0 h-14 bg-gradient-to-b from-black/65 to-transparent" />
      <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-black/70 to-transparent" />

      <div className="absolute left-6 top-5 max-w-[46rem]">
        <div className="flex items-center gap-3">
          <span className={`h-2.5 w-2.5 rounded-full ${live ? 'bg-red-500 shadow-[0_0_18px_rgba(239,68,68,0.95)]' : 'bg-slate-500'}`} />
          <span className="font-mono text-[10px] uppercase tracking-[0.42em] text-slate-200/90">
            {live ? 'REC · LIVE DIGITAL TWIN' : 'REC STANDBY · START BACKEND'}
          </span>
        </div>
        <div className="mt-3 font-sans text-3xl font-semibold tracking-[-0.04em] text-white drop-shadow-[0_2px_18px_rgba(15,23,42,0.85)]">
          {fightMode ? 'FIGHT MODE · cinematic dogfight reference' : 'SAAB Gripen visual shell · compact 200 m orbit mission'}
        </div>
        <div className="mt-2 max-w-xl font-mono text-[10px] uppercase tracking-[0.24em] text-cyan-100/80">
          {fightMode ? 'high-energy S-turns · visual inverted rolls · chase camera package' : 'fixed-gain autopilot · 12-state MAV dynamics · cinematic chase camera'}
        </div>
      </div>

      <div className="absolute right-6 top-5 w-80 rounded-2xl border border-white/10 bg-slate-950/45 p-4 shadow-[0_18px_60px_rgba(2,6,23,0.42)] backdrop-blur-md">
        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.24em] text-cyan-100/75">
          <span>Mission Phase</span>
          <span className="text-slate-300 tabular-nums">{time.toFixed(1)} s</span>
        </div>
        <div className="mt-2 font-sans text-xl font-semibold text-white">{phase.label}</div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-300">{phase.detail}</div>
        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-300 shadow-[0_0_16px_rgba(34,211,238,0.6)] transition-[width]"
            style={{ width: `${clamp(phase.progress * 100, 0, 100)}%` }}
          />
        </div>
      </div>

      <div className="absolute bottom-6 left-1/2 grid w-[min(860px,calc(100%-3rem))] -translate-x-1/2 grid-cols-5 gap-2">
        {[
          ['ALT', altitude.toFixed(0), 'm'],
          ['Va', airspeed.toFixed(1), 'm/s'],
          ['ROLL', roll.toFixed(1), 'deg'],
          ['TRACK Δ', trackError.toFixed(1), 'm'],
          ['THR', (throttle * 100).toFixed(0), '%'],
        ].map(([label, value, unit]) => (
          <div key={label} className="rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-center shadow-[0_12px_34px_rgba(2,6,23,0.36)] backdrop-blur-md">
            <div className="font-mono text-[9px] uppercase tracking-[0.24em] text-slate-400">{label}</div>
            <div className="mt-1 font-sans text-xl font-semibold tabular-nums text-slate-50">{value}<span className="ml-1 text-xs font-medium text-cyan-200">{unit}</span></div>
          </div>
        ))}
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-300/50 to-transparent" />
    </div>
  );
}

// ============================================================================
// 3D scene contents + outer layout
// ============================================================================

function SceneContents() {
  const latest = useSimulationStore(state => state.latestTelemetry);
  const history = useSimulationStore(state => state.telemetry.history);
  return (
    <>
      {/* Atmospheric sky dome (Preetham model). Sun low in the southwest for
          a warm cockpit-out-the-window mood. */}
      <Sky
        distance={4500}
        sunPosition={[140, 55, 90]}
        inclination={0.48}
        azimuth={0.25}
        mieCoefficient={0.005}
        mieDirectionalG={0.8}
        rayleigh={1.55}
        turbidity={5.2}
      />
      <hemisphereLight args={['#d8e8ff', '#44522f', 0.92]} />
      <directionalLight
        position={[80, 120, 60]}
        intensity={1.45}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-55, 34, -80]} intensity={0.42} color="#7dd3fc" />
      <ambientLight intensity={0.24} />

      <FlightSimWorld latest={latest} />
      <ReferencePathTrace latest={latest} history={history} />
      <CinematicCameraRig latest={latest} />

      <AircraftVisualAttitude latest={latest}>
        <GripenModelErrorBoundary fallback={<GripenLoadingPlaceholder latest={latest} />}>
          <Suspense fallback={<GripenLoadingPlaceholder latest={latest} />}>
            <GripenModel latest={latest} />
          </Suspense>
        </GripenModelErrorBoundary>
      </AircraftVisualAttitude>
    </>
  );
}

// Recording-oriented camera: the aircraft stays anchored at the scene origin,
// so a smooth camera dolly can create chase-footage motion without expanding
// world coordinates or allocating trace data. Roll and airspeed slightly bias
// the shot, making turns and the climb read clearly on screen recordings.
function CinematicCameraRig({ latest }: { latest?: TelemetryPoint }) {
  const { camera } = useThree();
  const uiMode = useSimulationStore(state => state.uiMode);
  const selectedProfile = useSimulationStore(state => state.backendTrajectoryProfile);
  const target = useMemo(() => new THREE.Vector3(0, 0.08, 0.22), []);
  const desired = useMemo(() => new THREE.Vector3(), []);
  const fightBlendRef = useRef(0);
  const cameraMetricsRef = useRef({ speed: 0, altitude: 0, turn: 0, vertical: 0 });
  useEffect(() => {
    if (uiMode !== 'operator') return;
    camera.position.set(1.65, 1.05, -3.35);
    camera.lookAt(0, 0.05, 0);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = 46;
      camera.updateProjectionMatrix();
    }
  }, [camera, uiMode]);
  useFrame((state, delta) => {
    if (uiMode !== 'cinematic') return;
    const fightTarget = latest?.trajectoryProfile === 'fight_mode' || selectedProfile === 'fight_mode' ? 1 : 0;
    fightBlendRef.current = THREE.MathUtils.lerp(
      fightBlendRef.current,
      fightTarget,
      1 - Math.exp(-delta * 1.25),
    );
    const fightBlend = fightBlendRef.current;
    const targetSpeed = clamp(latest?.airspeed ?? 0, 0, 160) / 160;
    const targetAltitude = clamp(latest?.altitude ?? 0, 0, 390) / 390;
    const targetTurn = clamp(headingErrorDegrees(latest?.yawAngle ?? 0, latest?.targetHeading ?? latest?.yawAngle ?? 0) / 48, -1, 1);
    const targetVertical = clamp(((latest?.targetAltitude ?? latest?.altitude ?? 0) - (latest?.altitude ?? 0)) / 110, -1, 1);
    const metricFactor = 1 - Math.exp(-delta * 4.8);
    cameraMetricsRef.current.speed = THREE.MathUtils.lerp(cameraMetricsRef.current.speed, targetSpeed, metricFactor);
    cameraMetricsRef.current.altitude = THREE.MathUtils.lerp(cameraMetricsRef.current.altitude, targetAltitude, metricFactor);
    cameraMetricsRef.current.turn = THREE.MathUtils.lerp(cameraMetricsRef.current.turn, targetTurn, metricFactor);
    cameraMetricsRef.current.vertical = THREE.MathUtils.lerp(cameraMetricsRef.current.vertical, targetVertical, metricFactor);
    const { speed, altitude, turn, vertical } = cameraMetricsRef.current;
    const t = state.clock.elapsedTime;
    const fightOrbit = Math.sin(t * 0.58) * fightBlend;
    const breathing = Math.sin(t * (0.18 + fightBlend * 0.10)) * (0.08 + fightBlend * 0.04);
    desired.set(
      1.68 + turn * 0.82 + Math.sin(t * 0.11) * 0.18 + fightOrbit * 0.62,
      0.82 + altitude * 0.62 + vertical * 0.32 * fightBlend + breathing + Math.cos(t * 0.46) * 0.12 * fightBlend,
      -3.2 - speed * 0.95 + Math.cos(t * 0.09) * 0.15 - 0.45 * fightBlend,
    );
    camera.position.lerp(desired, 1 - Math.exp(-delta * 2.65));
    camera.lookAt(target);
    if (camera instanceof THREE.PerspectiveCamera) {
      const desiredFov = (44 + fightBlend * 3.2) - speed * 3.8 + altitude * 1.4;
      camera.fov = THREE.MathUtils.lerp(camera.fov, desiredFov, 1 - Math.exp(-delta * 2.0));
      camera.updateProjectionMatrix();
    }
  });
  return null;
}

export function SimpleFlightScene() {
  const latest = useSimulationStore(state => state.latestTelemetry);
  const uiMode = useSimulationStore(state => state.uiMode);
  const backendLifecycle = useBackendLifecycle();
  const cinematic = uiMode === 'cinematic';
  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[radial-gradient(circle_at_top,_#0b1220,_#020617_60%)] text-slate-100">
      <BackendTelemetryBridge />
      <TopBar lifecycle={backendLifecycle} />
      <main className="relative flex min-h-0 flex-1 overflow-hidden">
        {!cinematic && <LeftInstrumentCluster latest={latest} />}
        <section className="relative min-w-0 flex-1 overflow-hidden">
          <Canvas dpr={[1, 1.25]} shadows camera={{ position: [1.65, 1.05, -3.35], fov: 46 }}>
            <PerspectiveCamera makeDefault position={[1.65, 1.05, -3.35]} fov={46} />
            <OrbitControls makeDefault enabled={!cinematic} enableDamping dampingFactor={0.08} enablePan={false} target={[0, 0.05, 0]} minDistance={1.35} maxDistance={28} />
            <color attach="background" args={['#b7d2e6']} />
            <fog attach="fog" args={['#c8d8c4', 85, 380]} />
            <SceneContents />
          </Canvas>
          {cinematic && <CinematicTelemetryOverlay latest={latest} />}
        </section>
        {!cinematic && <RightControlColumn latest={latest} lifecycle={backendLifecycle} />}
      </main>
      {!cinematic && <ChartStrip />}
    </div>
  );
}
