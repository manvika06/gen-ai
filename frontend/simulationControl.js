// simulationControl.js

let socket;
let currentRowTimer;

function initWebSocket(fileName) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }

    const encodedFileName = encodeURIComponent(fileName);
    //socket = new WebSocket(`ws://localhost:8000/ws/${encodedFileName}`);
    socket = new WebSocket(`wss://${backendUrl.replace("https://", "")}/ws/${encodedFileName}`);

    socket.onopen = function(event) {
        console.log('WebSocket connection established');
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    socket.onclose = function(event) {
        console.log('WebSocket connection closed');
        clearInterval(currentRowTimer);
    };

    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        alert('WebSocket error occurred. Check console for details.');
    };
}

function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'alert':
            handleAlertMessage(data.message);
            // displayAlert(data.message);
            break;
        case 'status':
            updateStatus(data.message);
            break;
        case 'row_data':
            updateCurrentRowData(data.data.data, data.data.alerts);
            break;
        case 'error':
            console.error('Simulation error:', data.message);
            alert('Simulation error: ' + data.message);
            stopSimulation();
            break;
    }
}
function updateCurrentRowData(rowData, alerts) {
    console.log('Row data:', rowData);
    console.log('Alerts:', alerts);
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');

    // Clear the status message when new data is being added
    document.getElementById('row-status-message').textContent = '';

    // If the header is empty, populate it with keys from rowData
    if (tableHeader.children.length === 0) {
        for (const key in rowData) {
            let headerCell = document.createElement('th');
            headerCell.textContent = key;
            tableHeader.appendChild(headerCell);
        }
    }

    // Create a new row at the top of the table for the new data
    let newRow = document.createElement('tr');
    for (const [key, value] of Object.entries(rowData)) {
        let cell = document.createElement('td');
        cell.textContent = value;
        
        // Check if the value has breached its threshold and apply red background
        if (alerts[key]) {
            cell.style.backgroundColor = '#ffcccc'; // Light red background
            cell.style.color = '#cc0000'; // Dark red text
        }
        
        newRow.appendChild(cell);
    }

    // Insert the new row at the top of the table body
    tableBody.insertBefore(newRow, tableBody.firstChild);
}

function initSimulationControl() {
    const csvSelector = document.getElementById('csv-selector');
    const startButton = document.getElementById('start-simulation');
    const stopButton = document.getElementById('stop-simulation');

    fetchCSVFiles();

    startButton.addEventListener('click', startSimulation);
    stopButton.addEventListener('click', stopSimulation);
}

async function fetchCSVFiles() {
    try {
        //const response = await fetch('http://localhost:8000/api/csv-files');
	const response = await fetch(`${backendUrl}/api/csv-files`);
        const data = await response.json();
        populateCSVSelector(data.files);
    } catch (error) {
        console.error('Error fetching CSV files:', error);
        alert('Failed to fetch CSV files');
    }
}

function populateCSVSelector(files) {
    const csvSelector = document.getElementById('csv-selector');
    csvSelector.innerHTML = '<option value="">Select a CSV file</option>';  // Ensure there is a default option

    if (files.length > 0) {
        files.forEach((file, index) => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            csvSelector.appendChild(option);

            // Automatically select the first file as the default option
            if (index === 0) {
                csvSelector.value = file;  // Set the first file as selected
            }
        });
    } else {
        csvSelector.innerHTML = '<option value="">No CSV files found</option>';  // Handle case where no files are found
    }
}


async function startSimulation() {
    const csvSelector = document.getElementById('csv-selector');
    const selectedFile = csvSelector.value;

    if (!selectedFile) {
        alert('Please select a CSV file');
        return;
    }

    try {
        //const response = await fetch('http://localhost:8000/api/start-simulation', {
        const response = await fetch(`${backendUrl}/api/start-simulation`, {

            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file: selectedFile }),
        });

        if (response.ok) {
            updateStatus('Simulating');
            updateSimulationControls(true);
            clearAlerts();
            initWebSocket(selectedFile);
            
            // Set up timer to clear row status message every 4 seconds
            // currentRowTimer = setInterval(() => {
            //     document.getElementById('row-status-message').textContent = 'Waiting for next row...';
            // }, 4000);

        } else {
            throw new Error('Failed to start simulation');
        }
    } catch (error) {
        console.error('Error starting simulation:', error);
        alert('Failed to start simulation');
    }
}


function stopSimulation() {
    // updateStatus('Ready');
    // updateSimulationControls(false);
    // if (socket && socket.readyState === WebSocket.OPEN) {
    //     socket.close();
    // }
    // clearInterval(currentRowTimer);
    // document.getElementById('current-row-data').innerHTML = '';
}

function updateSimulationControls(isRunning) {
    const startButton = document.getElementById('start-simulation');
    const stopButton = document.getElementById('stop-simulation');
    const csvSelector = document.getElementById('csv-selector');

    startButton.disabled = isRunning;
    stopButton.disabled = !isRunning;
    csvSelector.disabled = isRunning;
}
