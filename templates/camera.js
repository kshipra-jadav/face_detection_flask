// Get access to the user's camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        var videoElement = document.getElementById('cameraFeed');
        videoElement.srcObject = stream;
    })
    .catch(function (error) {
        console.error('Error accessing camera: ' + error);
    });
