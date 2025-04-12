// static/js/webcam.js
const video = document.getElementById('video');
const predictionDisplay = document.getElementById('prediction');

// Get access to webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
    });

// Capture a frame and send it to the backend
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);

    const imageDataURL = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageDataURL })
    })
    .then(response => response.json())
    .then(data => {
        predictionDisplay.textContent = data.prediction;
    })
    .catch(err => {
        console.error("Prediction error:", err);
    });
}

// Repeat every 2 seconds
setInterval(captureAndSendFrame, 2000);
