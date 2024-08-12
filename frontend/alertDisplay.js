

function initAlertDisplay() {
    // Any initialization code if needed
}

function clearAlerts() {
    const alertsContainer = document.getElementById('alerts-container');
    alertsContainer.innerHTML = '';
}

function handleAlertMessage(alert) {
    const alertsContainer = document.getElementById('alerts-container');
    const alertElement = document.createElement('div');
    alertElement.classList.add('alert');
    alertElement.textContent = alert;
    alertsContainer.appendChild(alertElement);

    // Scroll to the bottom of the alerts container
    alertsContainer.scrollTop = alertsContainer.scrollHeight;
    if(!getIsRecording()){
        speak(alert); // Read the alert aloud
    }
}
