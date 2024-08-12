// app.js

let backendUrl="https://12d02f8b61b542e700b547e0bb948920.serveo.net"
function init() {
    initSimulationControl();
    initAlertDisplay();
    initAudioRecorder();
    initStatusIndicator();
    updateStatus('Ready');
}

document.addEventListener('DOMContentLoaded', init);
