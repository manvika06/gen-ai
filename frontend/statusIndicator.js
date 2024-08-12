// statusIndicator.js - Handles status indicator functionality

function initStatusIndicator() {
    // Initialize any necessary event listeners or setup for status indicator
    updateStatus('Idle'); // Set initial status
}

function updateStatus(status) {
    const statusElement = document.getElementById('current-status');
    if (statusElement) {
        statusElement.textContent = status;
        updateStatusStyling(status);
    }
}

function updateStatusStyling(status) {
    const statusElement = document.getElementById('current-status');
    if (statusElement) {
        // Reset classes
        statusElement.classList.remove('status-idle', 'status-running', 'status-error');

        // Add appropriate class based on status
        switch (status.toLowerCase()) {
            case 'idle':
                statusElement.classList.add('status-idle');
                break;
            case 'running':
            case 'simulating':
            case 'processing':
                statusElement.classList.add('status-running');
                break;
            case 'error':
            case 'disconnected':
                statusElement.classList.add('status-error');
                break;
            // Add more cases as needed
        }
    }
}