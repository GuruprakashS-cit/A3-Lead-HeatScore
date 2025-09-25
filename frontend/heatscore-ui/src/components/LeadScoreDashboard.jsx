import React, { useState } from 'react';
import Papa from 'papaparse';
import { CSVLink } from 'react-csv';
import NextActionModal from './NextActionModal'; // Import the new modal component

const LeadScoreDashboard = () => {
  const [csvFile, setCsvFile] = useState(null);
  const [csvData, setCsvData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [filterClass, setFilterClass] = useState('All');
  
  // State for modal and next action
  const [nextAction, setNextAction] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [selectedLeadId, setSelectedLeadId] = useState(null);

  const getClassColor = (cls) => {
    switch (cls) {
      case 'Hot':
        return 'bg-red-500 text-white';
      case 'Warm':
        return 'bg-yellow-400 text-yellow-900';
      case 'Cold':
        return 'bg-blue-500 text-white';
      default:
        return 'bg-gray-200 text-gray-800';
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length) {
      setCsvFile(e.target.files[0]);
      setError('');
    }
  };

  const handleFileUpload = async () => {
    if (!csvFile) {
      setError('Please select a CSV file first.');
      return;
    }

    setLoading(true);
    setError('');

    Papa.parse(csvFile, {
      header: true,
      skipEmptyLines: true,
      complete: async function (results) {
        const leads = results.data;
        const formattedLeads = leads.map(lead => ({
          lead_id: parseInt(lead.lead_id) || 0,
          source: lead.source,
          recency_days: parseInt(lead.recency_days) || 0,
          region: lead.region,
          role: lead.role,
          campaign: lead.campaign,
          page_views: parseInt(lead.page_views) || 0,
          last_touch: lead.last_touch,
          prior_course_interest: parseInt(lead.prior_course_interest) || 0,
        }));
        
        try {
          const response = await fetch('http://localhost:8000/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formattedLeads),
          });

          if (!response.ok) {
            const errorData = await response.json();
            let errorMessage = 'An error occurred while scoring leads.';
            if (errorData.detail && Array.isArray(errorData.detail)) {
                errorMessage = 'Validation Error: ';
                errorData.detail.forEach(err => {
                  const field = err.loc[err.loc.length - 1]; 
                  errorMessage += `${field}: ${err.msg}. `;
                });
            } else {
              errorMessage = errorData.detail || 'Failed to fetch data from API';
            }
            throw new Error(errorMessage);
          }

          const scoredLeads = await response.json();
          setCsvData(scoredLeads);
          setFilterClass('All');
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const handleGenerateNextAction = async (lead) => {
    setActionLoading(true);
    setSelectedLeadId(lead.lead_id);
    setNextAction(null);
    setIsModalOpen(true);

    try {
        const response = await fetch('http://localhost:8000/recommend/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(lead),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate next action');
        }

        const actionData = await response.json();
        setNextAction(actionData);
    } catch (err) {
        setError(err.message);
    } finally {
        setActionLoading(false);
    }
  };

  const renderTableHeaders = () => {
    if (csvData.length === 0) return null;
    
    const leadHeaders = Object.keys(csvData[0].lead);
    const otherHeaders = Object.keys(csvData[0]).filter(key => key !== 'lead');
    const headers = [...leadHeaders, ...otherHeaders, 'Next Action'];
    
    return (
      <tr>
        {headers.map((header) => (
          <th key={header} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            {header.replace(/_/g, ' ')}
          </th>
        ))}
      </tr>
    );
  };

  const renderTableRows = () => {
    const filteredData = filterClass === 'All'
      ? csvData
      : csvData.filter(lead => lead.class === filterClass);

    const sortedData = [...filteredData].sort((a, b) => {
      if (filterClass === 'All') {
        return 0;
      }
      const probA = a.probabilities[filterClass] || 0;
      const probB = b.probabilities[filterClass] || 0;
      return probB - probA;
    });

    return sortedData.map((result, index) => {
      const allValues = { ...result.lead, ...result };
      delete allValues.lead;

      const nextActionButton = (
        <button
          onClick={() => handleGenerateNextAction(result.lead)}
          disabled={actionLoading && selectedLeadId === result.lead.lead_id}
          className="bg-green-600 text-white px-3 py-1 rounded-md text-xs font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {'Get Action'}
        </button>
      );

      return (
        <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
          {Object.entries(allValues).map(([key, value], i) => (
            <td key={i} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
              {key === 'class' && typeof value === 'string' ? (
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getClassColor(value)}`}>
                  {value}
                </span>
              ) : key === 'probabilities' && typeof value === 'object' ? (
                <div className="flex flex-col space-y-1">
                  {Object.entries(value).map(([cls, prob]) => (
                    <div key={cls}>
                      <span className="font-semibold">{cls}:</span> {`${(prob * 100).toFixed(1)}%`}
                    </div>
                  ))}
                </div>
              ) : key === 'top_features' && Array.isArray(value) ? (
                <div className="flex flex-wrap gap-1">
                  {value.map((feature) => (
                    <span key={feature} className="bg-gray-200 text-gray-700 px-2 py-0.5 rounded-full text-xs font-medium">
                      {feature.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              ) : typeof value === 'object' ? JSON.stringify(value) : value}
            </td>
          ))}
          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            {nextActionButton}
          </td>
        </tr>
      );
    });
  };

  return (
    <div className="container mx-auto p-4">
      <div className="bg-white shadow rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-bold mb-4">Upload Leads for Scoring</h2>
        <div className="flex items-center space-x-4">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
          <button
            onClick={handleFileUpload}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Score Leads'}
          </button>
        </div>
        {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}
      </div>

      {csvData.length > 0 && (
        <>
          <div className="bg-white shadow rounded-lg p-6 mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Scored Leads</h2>
              <div className="flex items-center space-x-2">
                <span className="text-gray-700">Filter by Class:</span>
                <select
                  value={filterClass}
                  onChange={(e) => setFilterClass(e.target.value)}
                  className="block w-40 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                >
                  <option value="All">All</option>
                  <option value="Hot">Hot</option>
                  <option value="Warm">Warm</option>
                  <option value="Cold">Cold</option>
                </select>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  {renderTableHeaders()}
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {renderTableRows()}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
      
      <NextActionModal 
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        nextAction={nextAction}
        actionLoading={actionLoading}
        selectedLeadId={selectedLeadId}
      />
    </div>
  );
};

export default LeadScoreDashboard;
