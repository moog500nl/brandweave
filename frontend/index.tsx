import React from 'react';
import ReactDOM from 'react-dom';
import BrandChart from './BrandChart';

interface DataItem {
  brand: string;
  count: number;
}

declare global {
  interface Window {
    brandChartData: DataItem[];
  }
}

const StreamlitBrandChart: React.FC = () => {
  return <BrandChart data={window.brandChartData || []} />;
};

ReactDOM.render(
  <React.StrictMode>
    <StreamlitBrandChart />
  </React.StrictMode>,
  document.getElementById('root')
);
