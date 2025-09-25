import React from 'react';

const NextActionModal = ({ isOpen, onClose, nextAction, actionLoading, selectedLeadId }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
        <div className="mt-3 text-center">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Next Action Recommendation</h3>
          <div className="mt-2 px-7 py-3">
            {actionLoading ? (
              <p>Generating message...</p>
            ) : nextAction ? (
              <div className="space-y-4 text-left">
                <div>
                  <h4 className="text-sm font-semibold text-gray-700">Channel:</h4>
                  <p className="text-gray-900">{nextAction.channel}</p>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700">Message:</h4>
                  <p className="text-gray-900 whitespace-pre-wrap">{nextAction.message}</p>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-gray-700">Rationale:</h4>
                  <p className="text-gray-900 text-xs">{nextAction.rationale}</p>
                </div>
              </div>
            ) : (
              <p>No action found for Lead {selectedLeadId}.</p>
            )}
          </div>
          <div className="items-center px-4 py-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NextActionModal;
