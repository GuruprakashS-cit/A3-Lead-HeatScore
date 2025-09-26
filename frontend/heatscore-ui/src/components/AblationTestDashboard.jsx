import React, { useState } from 'react';

const AblationTestDashboard = () => {
    const [leads, setLeads] = useState([]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState({});
    const [error, setError] = useState('');
    const [selectedLeads, setSelectedLeads] = useState([]);

    const fetchAblationResults = async (lead) => {
        const searchTypes = ["vector-only", "hybrid", "hybrid-rerank"];
        const newResults = {};

        for (const type of searchTypes) {
            try {
                const response = await fetch(`http://localhost:8000/recommend/ablation-test/${type}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(lead),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `Failed to run ${type} test`);
                }

                const data = await response.json();
                newResults[type] = data;
            } catch (err) {
                setError(err.message);
                return;
            }
        }
        setResults(prev => ({ ...prev, [lead.lead_id]: newResults }));
    };

    const handleAblationTest = () => {
        setLoading(true);
        setError('');
        setResults({});
        
        // This is a simplified way to get 3 leads for the demo
        // In a real scenario, you'd want to pick 1 Hot, 1 Warm, 1 Cold
        const leadsToTest = leads.slice(0, 3);
        setSelectedLeads(leadsToTest);
        
        const promises = leadsToTest.map(lead => fetchAblationResults(lead));
        
        Promise.all(promises).finally(() => {
            setLoading(false);
        });
    };

    const handleLeadsUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Simplified CSV parsing for demo
        const reader = new FileReader();
        reader.onload = (event) => {
            const text = event.target.result;
            const rows = text.split('\n').slice(1).map(row => {
                const values = row.split(',');
                return {
                    lead_id: parseInt(values[0]) || 0,
                    source: values[1],
                    recency_days: parseInt(values[2]) || 0,
                    region: values[3],
                    role: values[4],
                    campaign: values[5],
                    page_views: parseInt(values[6]) || 0,
                    last_touch: values[7],
                    prior_course_interest: parseInt(values[8]) || 0,
                };
            });
            setLeads(rows.filter(r => r.lead_id > 0));
        };
        reader.readAsText(file);
    };

    return (
        <div className="container mx-auto p-4">
            <div className="bg-white shadow rounded-lg p-6 mb-8">
                <h2 className="text-2xl font-bold mb-4">Ablation Testing Dashboard</h2>
                <div className="flex items-center space-x-4">
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleLeadsUpload}
                        className="block w-full text-sm text-gray-500
                          file:mr-4 file:py-2 file:px-4
                          file:rounded-full file:border-0
                          file:text-sm file:font-semibold
                          file:bg-blue-50 file:text-blue-700
                          hover:file:bg-blue-100"
                    />
                    <button
                        onClick={handleAblationTest}
                        disabled={loading || leads.length === 0}
                        className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50"
                    >
                        {loading ? 'Running Tests...' : 'Run Ablation Test'}
                    </button>
                </div>
                {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}
            </div>

            {Object.keys(results).length > 0 && (
                <div className="space-y-8">
                    {selectedLeads.map(lead => (
                        <div key={lead.lead_id} className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-xl font-bold mb-2">Lead ID: {lead.lead_id}</h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {Object.entries(results[lead.lead_id]).map(([type, result]) => (
                                    <div key={type} className="bg-gray-50 p-4 rounded-lg shadow-inner">
                                        <h4 className="font-semibold text-gray-700 capitalize mb-2">{type.replace('-', ' ')}</h4>
                                        <p className="text-sm">
                                            <span className="font-medium">Message:</span> {result.message}
                                        </p>
                                        <p className="text-sm">
                                            <span className="font-medium">Rationale:</span> {result.rationale}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default AblationTestDashboard;
