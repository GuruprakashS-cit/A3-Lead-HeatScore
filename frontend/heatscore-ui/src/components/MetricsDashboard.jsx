import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const MetricsDashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/metrics/');
        if (!response.ok) {
          throw new Error('Failed to fetch metrics data');
        }
        const data = await response.json();
        setMetrics(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  const classLabels = ['Cold', 'Hot', 'Warm'];

  // Data for ROC curve plot
  const rocCurveData = {
    labels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    datasets: metrics ?
      Object.keys(metrics.roc_curve_data).map((cls, i) => {
        const color = ['rgb(54, 162, 235)', 'rgb(255, 99, 132)', 'rgb(255, 205, 86)'][i];
        const dataPoints = metrics.roc_curve_data[cls].fpr.map((fpr, index) => ({
          x: fpr,
          y: metrics.roc_curve_data[cls].tpr[index]
        }));
        return {
          label: `${cls} (AUC = ${metrics.roc_curve_data[cls].auc.toFixed(2)})`,
          data: dataPoints,
          borderColor: color,
          backgroundColor: color,
          tension: 0.1,
          fill: false,
        };
      }) : []
  };

  const rocOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'ROC Curve per Class' },
    },
    scales: {
      x: { 
        title: { display: true, text: 'False Positive Rate' },
        min: 0,
        max: 1,
        type: 'linear',
        position: 'bottom',
      },
      y: { 
        title: { display: true, text: 'True Positive Rate' },
        min: 0,
        max: 1,
      }
    }
  };

  // Data for reliability plot
  const reliabilityData = {
    labels: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    datasets: metrics ? [
      {
        label: 'Ideal Calibration',
        data: [{x: 0, y: 0}, {x: 1, y: 1}],
        borderColor: 'rgb(153, 102, 255)',
        borderDash: [5, 5],
        fill: false,
        pointStyle: false
      },
      ...Object.keys(metrics.reliability_plot_data).map((cls, i) => {
        const color = ['rgb(54, 162, 235)', 'rgb(255, 99, 132)', 'rgb(255, 205, 86)'][i];
        
        // This is where you would process your data to create the reliability plot
        // For now, we'll use a simplified example as your API doesn't return the raw data needed for this.
        const calibrationPoints = metrics.reliability_plot_data[cls].mean_predicted_value.map((mean, index) => ({
            x: mean,
            y: metrics.reliability_plot_data[cls].fraction_of_positives[index]
        }));

        return {
          label: `${cls} (Brier Score = ${metrics.brier_scores[cls].toFixed(2)})`,
          data: calibrationPoints,
          borderColor: color,
          backgroundColor: color,
          tension: 0.1,
          fill: false,
        };
      })
    ] : []
  };

  const reliabilityOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Model Calibration (Reliability Plot)' },
    },
    scales: {
      x: { 
        title: { display: true, text: 'Predicted Probability' },
        min: 0,
        max: 1,
        type: 'linear',
        position: 'bottom',
      },
      y: { 
        title: { display: true, text: 'Actual Positive Rate' },
        min: 0,
        max: 1,
      }
    }
  };

  if (loading) {
    return <div className="p-4 text-center">Loading metrics...</div>;
  }

  if (error) {
    return <div className="p-4 text-center text-red-500">Error: {error}</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <div className="bg-white shadow rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-bold mb-4">Model Evaluation Metrics</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="p-4 bg-gray-100 rounded-lg shadow-inner">
            <h3 className="text-lg font-semibold text-gray-700">Macro F1 Score</h3>
            <p className="text-3xl font-bold text-blue-600">{metrics?.macro_f1_score.toFixed(2)}</p>
          </div>
          <div className="p-4 bg-gray-100 rounded-lg shadow-inner">
            <h3 className="text-lg font-semibold text-gray-700">Accuracy</h3>
            <p className="text-3xl font-bold text-blue-600">{metrics?.classification_report.accuracy.toFixed(2)}</p>
          </div>
          <div className="p-4 bg-gray-100 rounded-lg shadow-inner">
            <h3 className="text-lg font-semibold text-gray-700">Brier Score (Avg)</h3>
            <p className="text-3xl font-bold text-blue-600">
              {metrics ? 
                (Object.values(metrics.brier_scores).reduce((sum, score) => sum + score, 0) / Object.keys(metrics.brier_scores).length).toFixed(2)
              : 'N/A'}
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Confusion Matrix</h3>
            <div className="flex justify-center items-center">
              <table className="min-w-max border rounded-lg overflow-hidden text-sm">
                <thead>
                  <tr className="bg-gray-100 text-gray-600 uppercase leading-normal">
                    <th className="py-2 px-4 text-left"></th>
                    {classLabels.map(label => <th key={label} className="py-2 px-4 text-left">Predicted {label}</th>)}
                  </tr>
                </thead>
                <tbody className="text-gray-600 font-light">
                  {metrics?.confusion_matrix.map((row, i) => (
                    <tr key={i} className="border-b border-gray-200">
                      <td className="py-2 px-4 text-left font-bold">Actual {classLabels[i]}</td>
                      {row.map((val, j) => (
                        <td key={j} className="py-2 px-4 text-left">{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Per-Class Report</h3>
            <div className="flex justify-center items-center">
              <table className="min-w-max border rounded-lg overflow-hidden text-sm">
                <thead>
                  <tr className="bg-gray-100 text-gray-600 uppercase leading-normal">
                    <th className="py-2 px-4 text-left">Class</th>
                    <th className="py-2 px-4 text-left">Precision</th>
                    <th className="py-2 px-4 text-left">Recall</th>
                    <th className="py-2 px-4 text-left">F1-Score</th>
                  </tr>
                </thead>
                <tbody className="text-gray-600 font-light">
                  {classLabels.map((label) => (
                    <tr key={label} className="border-b border-gray-200">
                      <td className="py-2 px-4 text-left font-bold">{label}</td>
                      <td className="py-2 px-4 text-left">{metrics?.classification_report[label].precision.toFixed(2)}</td>
                      <td className="py-2 px-4 text-left">{metrics?.classification_report[label].recall.toFixed(2)}</td>
                      <td className="py-2 px-4 text-left">{metrics?.classification_report[label]['f1-score'].toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">ROC Curve per Class</h3>
            <Line options={rocOptions} data={rocCurveData} />
          </div>

          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Model Calibration (Reliability Plot)</h3>
            <Line options={reliabilityOptions} data={reliabilityData} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsDashboard;
