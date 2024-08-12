let recognition;

let transcript = ''; // Hold the concatenated text

function initAudioRecorder() {
    const startRecordingButton = document.getElementById('start-recording');
    const stopRecordingButton = document.getElementById('stop-recording');

    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = function(event) {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            displayCommandResult(transcript + interimTranscript);
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error', event);
            displayCommandResult('Error in speech recognition: ' + event.error);
        };

        recognition.onend = function() {
            updateRecordingStatus(false);
        };
    } else {
        alert("Speech recognition not supported in your browser.");
    }

    startRecordingButton.addEventListener('click', () => {
        stopAllSpeech();  // Stop any ongoing speech synthesis
        startRecording();
    });
    stopRecordingButton.addEventListener('click', stopRecording);
}

function startRecording() {
    if (!getIsRecording()) {
        transcript = ''; // Reset the transcript
        recognition.start();
        setIsRecording(true);
        updateRecordingStatus(true);
    }
}

async function stopRecording() {
    if (getIsRecording()) {
        recognition.stop();
        updateRecordingStatus(false);
        await sendTextToBackend(transcript);
        // Note: isRecording will be set to false after speaking or stopping speech
    }
}

function sendTextToBackend(text) {
    //return fetch('http://localhost:8000/api/process-text', {
    return fetch(`${backendUrl}/api/process-text`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        displayAnswer(data.response);
        speak(data.response);  // This will set isRecording to false when done
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
        displayAnswer('Error: Unable to process your request.');
        setIsRecording(false);  // Ensure isRecording is set to false on error
    });
}

// ... rest of the code ...
function updateRecordingStatus(recording) {
    const startButton = document.getElementById('start-recording');
    const stopButton = document.getElementById('stop-recording');
    const statusElement = document.getElementById('recording-status');

    startButton.disabled = recording;
    stopButton.disabled = !recording;
    statusElement.textContent = recording ? 'Listening...' : 'Not listening';
}

function displayCommandResult(text) {
    const resultElement = document.getElementById('command-result');
    resultElement.textContent = `Recognized command: ${text}`;
}

function displayAnswer(text) {
    const answerBox = document.getElementById('answer-box');
    if (answerBox) {
        answerBox.textContent = text;
    } else {
        console.error('Answer box element not found');
    }
}
