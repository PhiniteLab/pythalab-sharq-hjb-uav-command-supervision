import { memo, type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Legend, Line as ChartLine, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useSimulationStore } from '../../state/simulationStore';
import { TelemetryPoint } from '../../simulation/contracts/telemetryTypes';

const UI_CHART_UPDATE_INTERVAL_MS = 1000;

function buildChartData(history: TelemetryPoint[]) {
  const backendHistory = history.filter(point => Number.isFinite(point.targetAltitude) && Boolean(point.autopilot));
  const recent = backendHistory.slice(-240);
  const stride = Math.max(1, Math.ceil(recent.length / 80));
  return recent
    .filter((_, index) => index % stride === 0 || index === recent.length - 1)
    .map(point => {
      const altRef = point.targetAltitude ?? point.altitude;
      const spdRef = point.targetAirspeed ?? point.airspeed;
      const hdgRef = point.targetHeading ?? point.yawAngle;
      const pitchRef = point.targetPitch ?? point.pitchAngle;
      const rollRef = point.targetRoll ?? 0;
      const hdgErrRaw = point.yawAngle - hdgRef;
      const hdgErr = ((hdgErrRaw + 180) % 360 + 360) % 360 - 180;
      return {
        t: Number(point.time.toFixed(1)),
        alt: Number(point.altitude.toFixed(2)),
        altRef: Number(altRef.toFixed(2)),
        altErrAbs: Number(Math.abs(point.altitude - altRef).toFixed(2)),
        spd: Number(point.airspeed.toFixed(2)),
        spdRef: Number(spdRef.toFixed(2)),
        spdErrAbs: Number(Math.abs(point.airspeed - spdRef).toFixed(2)),
        thrPct: Number(((point.throttle ?? 0) * 100).toFixed(1)),
        pitch: Number(point.pitchAngle.toFixed(2)),
        pitchRef: Number(pitchRef.toFixed(2)),
        roll: Number(point.rollAngle.toFixed(2)),
        rollRef: Number(rollRef.toFixed(2)),
        hdg: Number(point.yawAngle.toFixed(2)),
        hdgRef: Number(hdgRef.toFixed(2)),
        hdgErrAbs: Number(Math.abs(hdgErr).toFixed(2)),
        nz: Number((point.loadFactorNz ?? 1).toFixed(2)),
        aoa: Number((point.angleOfAttack ?? 0).toFixed(2)),
      };
    });
}

function paddedDomain(values: number[], fallback: [number, number]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return fallback;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const pad = Math.max((max - min) * 0.12, 1);
  return [Math.floor(min - pad), Math.ceil(max + pad)];
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col rounded-lg border border-cyan-500/20 bg-slate-950/85 p-2.5 shadow-inner">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-cyan-300">{title}</div>
      <div className="flex-1">{children}</div>
    </div>
  );
}

export const TrajectoryCharts = memo(function TrajectoryCharts() {
  const backendActive = useSimulationStore(state => state.backendActive);
  const [chartData, setChartData] = useState(() => buildChartData(useSimulationStore.getState().telemetry.history));
  const lastHistoryRef = useRef(useSimulationStore.getState().telemetry.history);

  useEffect(() => {
    let animationFrame = 0;
    let lastUpdateAt = 0;
    const update = () => {
      animationFrame = 0;
      lastUpdateAt = window.performance.now();
      setChartData(buildChartData(useSimulationStore.getState().telemetry.history));
    };
    update();
    const unsubscribe = useSimulationStore.subscribe((state) => {
      if (state.telemetry.history === lastHistoryRef.current) return;
      lastHistoryRef.current = state.telemetry.history;
      if (animationFrame) return;
      if (window.performance.now() - lastUpdateAt < UI_CHART_UPDATE_INTERVAL_MS) return;
      animationFrame = window.requestAnimationFrame(update);
    });
    return () => {
      unsubscribe();
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  const altitudeDomain = useMemo(() => paddedDomain(chartData.flatMap(point => [point.alt, point.altRef]), [0, 120]), [chartData]);
  const attitudeDomain = useMemo(() => paddedDomain(chartData.flatMap(point => [point.pitch, point.pitchRef, point.roll, point.rollRef]), [-15, 15]), [chartData]);
  const speedDomain = useMemo(() => paddedDomain(chartData.flatMap(point => [point.spd, point.spdRef]), [0, 160]), [chartData]);
  const altErrMax = useMemo(() => Math.max(2, ...chartData.map(p => p.altErrAbs)), [chartData]);
  const spdErrMax = useMemo(() => Math.max(1, ...chartData.map(p => p.spdErrAbs)), [chartData]);
  const hdgErrMax = useMemo(() => Math.max(2, ...chartData.map(p => p.hdgErrAbs)), [chartData]);
  const aoaDomain = useMemo(() => paddedDomain(chartData.flatMap(point => [point.aoa, point.nz * 5]), [-5, 12]), [chartData]);

  if (!backendActive || chartData.length < 2) {
    return (
      <div className="z-10 flex h-[120px] items-center justify-center border-t border-cyan-500/20 bg-slate-950/80 backdrop-blur">
        <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
          Awaiting backend telemetry · press <span className="text-emerald-300">Start</span> to launch the autopilot
        </div>
      </div>
    );
  }
  const chartProps = { data: chartData, margin: { top: 4, right: 6, bottom: 0, left: -28 } } as const;
  const axisStyle = { stroke: '#475569' } as const;
  const tickStyle = { fontSize: 9, fill: '#94a3b8' } as const;
  const tooltipStyle = { background: '#020617', border: '1px solid #155e75', fontSize: 10, padding: '4px 6px' } as const;
  const legendStyle = { fontSize: 9 } as const;
  return (
    <div className="z-10 grid grid-cols-1 gap-2 border-t border-cyan-500/20 bg-slate-950/80 p-2 backdrop-blur lg:grid-cols-4">
      <ChartCard title="Altitude (m) · ref vs actual">
        <ResponsiveContainer width="100%" height={96}>
          <ComposedChart {...chartProps}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="t" {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="alt" domain={altitudeDomain} {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="err" orientation="right" domain={[0, Math.max(altErrMax, 5)]} {...axisStyle} tick={tickStyle} width={28} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={legendStyle} iconSize={8} />
            <Area yAxisId="err" type="monotone" dataKey="altErrAbs" name="|err|" stroke="#f97316" fill="#f97316" fillOpacity={0.18} isAnimationActive={false} />
            <ChartLine yAxisId="alt" isAnimationActive={false} type="linear" dataKey="altRef" name="ref" dot={false} stroke="#22c55e" strokeDasharray="5 3" strokeWidth={2} />
            <ChartLine yAxisId="alt" isAnimationActive={false} type="linear" dataKey="alt" name="actual" dot={false} stroke="#38bdf8" strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Airspeed (m/s) · ref + throttle">
        <ResponsiveContainer width="100%" height={96}>
          <ComposedChart {...chartProps}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="t" {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="v" domain={speedDomain} {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="thr" orientation="right" domain={[0, 100]} {...axisStyle} tick={tickStyle} width={28} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={legendStyle} iconSize={8} />
            <Area yAxisId="thr" type="monotone" dataKey="thrPct" name="δt %" stroke="#10b981" fill="#10b981" fillOpacity={0.12} isAnimationActive={false} />
            <ChartLine yAxisId="v" isAnimationActive={false} type="linear" dataKey="spdRef" name="Va ref" dot={false} stroke="#22c55e" strokeDasharray="5 3" strokeWidth={2} />
            <ChartLine yAxisId="v" isAnimationActive={false} type="linear" dataKey="spd" name="Va" dot={false} stroke="#fbbf24" strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Attitude (deg) · θ / φ">
        <ResponsiveContainer width="100%" height={96}>
          <LineChart {...chartProps}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="t" {...axisStyle} tick={tickStyle} />
            <YAxis domain={attitudeDomain} {...axisStyle} tick={tickStyle} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={legendStyle} iconSize={8} />
            <ReferenceLine y={0} stroke="#334155" />
            <ChartLine isAnimationActive={false} type="linear" dataKey="pitchRef" name="θ ref" dot={false} stroke="#a78bfa" strokeDasharray="5 3" strokeWidth={1.6} />
            <ChartLine isAnimationActive={false} type="linear" dataKey="pitch" name="θ" dot={false} stroke="#f97316" strokeWidth={1.8} />
            <ChartLine isAnimationActive={false} type="linear" dataKey="rollRef" name="φ ref" dot={false} stroke="#22c55e" strokeDasharray="5 3" strokeWidth={1.6} />
            <ChartLine isAnimationActive={false} type="linear" dataKey="roll" name="φ" dot={false} stroke="#facc15" strokeWidth={1.8} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Tracking errors · α / Nz">
        <ResponsiveContainer width="100%" height={96}>
          <ComposedChart {...chartProps}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="t" {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="err" domain={[0, Math.max(altErrMax, spdErrMax, hdgErrMax)]} {...axisStyle} tick={tickStyle} />
            <YAxis yAxisId="aoa" orientation="right" domain={aoaDomain} {...axisStyle} tick={tickStyle} width={28} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={legendStyle} iconSize={8} />
            <ChartLine yAxisId="err" isAnimationActive={false} type="linear" dataKey="altErrAbs" name="|Δh|" dot={false} stroke="#38bdf8" strokeWidth={1.6} />
            <ChartLine yAxisId="err" isAnimationActive={false} type="linear" dataKey="spdErrAbs" name="|ΔVa|" dot={false} stroke="#fbbf24" strokeWidth={1.6} />
            <ChartLine yAxisId="err" isAnimationActive={false} type="linear" dataKey="hdgErrAbs" name="|Δψ|" dot={false} stroke="#a78bfa" strokeWidth={1.6} />
            <ChartLine yAxisId="aoa" isAnimationActive={false} type="linear" dataKey="aoa" name="α" dot={false} stroke="#f97316" strokeDasharray="3 2" strokeWidth={1.4} />
            <ChartLine yAxisId="aoa" isAnimationActive={false} type="linear" dataKey="nz" name="Nz" dot={false} stroke="#ef4444" strokeDasharray="3 2" strokeWidth={1.4} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
});
