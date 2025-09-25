import React, { useState } from 'react';
import LeadScoreDashboard from './components/LeadScoreDashboard';
import MetricsDashboard from './components/MetricsDashboard';
import './index.css';

function App() {
  const [view, setView] = useState('dashboard'); // 'dashboard' or 'metrics'

  return (
    <div className="bg-gray-100 min-h-screen">
      <nav className="bg-white shadow p-4 mb-8">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">Lead HeatScore</h1>
          <div className="space-x-4">
            <button
              onClick={() => setView('dashboard')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                view === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-200'
              }`}
            >
              Leads Dashboard
            </button>
            <button
              onClick={() => setView('metrics')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                view === 'metrics' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-200'
              }`}
            >
              Metrics Dashboard
            </button>
          </div>
        </div>
      </nav>
      {view === 'dashboard' ? <LeadScoreDashboard /> : <MetricsDashboard />}
    </div>
  );
}

export default App;
