import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface DataItem {
  brand: string;
  count: number;
}

interface BrandChartProps {
  data: DataItem[];
}

const BrandChart: React.FC<BrandChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={Math.max(400, data.length * 25)}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{
          top: 5,
          right: 30,
          left: 100,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" label={{ value: 'Frequency', position: 'bottom' }} />
        <YAxis
          type="category"
          dataKey="brand"
          tick={{ fontFamily: 'Roboto, sans-serif' }}
        />
        <Tooltip
          contentStyle={{
            fontFamily: 'Roboto, sans-serif',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
          }}
        />
        <Bar
          dataKey="count"
          fill="#4A90E2"
          radius={[0, 4, 4, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default BrandChart;
