import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarPoint } from "../api";

export interface PriceChartProps {
  data: BarPoint[];
  loading?: boolean;
  error?: string | null;
}

export function PriceChart({ data, loading = false, error = null }: PriceChartProps) {
  if (loading) {
    return <div className="panel">Loading bars…</div>;
  }
  if (error) {
    return <div className="panel error">{error}</div>;
  }
  if (!data.length) {
    return <div className="panel">No bars available for the selected contract.</div>;
  }
  return (
    <div className="panel chart">
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="price" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#1e293b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148, 163, 184, 0.2)" strokeDasharray="4 4" />
          <XAxis dataKey="timestamp" hide tickFormatter={(value) => new Date(value).toLocaleTimeString()} />
          <YAxis domain={["auto", "auto"]} tick={{ fill: "#94a3b8" }} width={64} />
          <Tooltip
            formatter={(value: number) => value.toFixed(2)}
            labelFormatter={(value: string) => new Date(value).toLocaleString()}
          />
          <Area type="monotone" dataKey="close" stroke="#60a5fa" fill="url(#price)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
