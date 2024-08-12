let isRecording = false; // Track recording status
let speechSynthesisUtterance = null;

function stopAllSpeech() {
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }
    document.getElementById('stop-speaking').disabled = true;
}

function speak(text) {
    stopAllSpeech();  // Ensure no other speech is currently playing
    if ('speechSynthesis' in window) {
        speechSynthesisUtterance = new SpeechSynthesisUtterance(text);
        speechSynthesis.speak(speechSynthesisUtterance);
        document.getElementById('stop-speaking').disabled = false;
        
        speechSynthesisUtterance.onend = function() {
            document.getElementById('stop-speaking').disabled = true;
            isRecording = false; // Set isRecording to false when speech ends naturally
        };
    }
}

function stopSpeaking() {
    stopAllSpeech();
    isRecording = false;
}

document.getElementById('stop-speaking').addEventListener('click', stopSpeaking);

// Expose functions to get and set isRecording
function getIsRecording() {
    return isRecording;
}

function setIsRecording(value) {
    isRecording = value;
}