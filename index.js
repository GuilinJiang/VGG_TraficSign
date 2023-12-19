let model;
let classIndices;

// Load the class indices
async function loadClassIndices() {
    const response = await fetch('class_indices.json');
    classIndices = await response.json();
}

// Function to get the class label by index
function getClassLabel(index) {
    // Find the class label that matches the index
    return Object.keys(classIndices).find(key => classIndices[key] === index);
}

// Load the model
async function loadModel() {
    model = await tf.loadLayersModel('VGG_0.8acc_jsonModel/model.json');
    // Enable the predict button only after the model has been loaded and class indices are available
    await loadClassIndices();
    document.getElementById('predictButton').disabled = false;
}

// Initialize model loading
loadModel();

// Function to handle image uploads and generate a preview
function handleImageUpload(event) {
    const imageUpload = document.getElementById('imageUpload');
    const predictionResult = document.getElementById('prediction');
    predictionResult.innerHTML = ''; // Clear previous predictions

    const file = imageUpload.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const imgElement = document.createElement('img');
        imgElement.src = e.target.result;
        imgElement.style.maxWidth = '100%';
        imgElement.style.borderRadius = '10px';
        predictionResult.appendChild(imgElement);
    };

    reader.readAsDataURL(file);
}

// Set up the image upload preview
document.getElementById('imageUpload').addEventListener('change', handleImageUpload);

// Set up the click event for the custom upload button
document.getElementById('uploadButton').addEventListener('click', function() {
    document.getElementById('imageUpload').click(); // Trigger the hidden input click
});

// Image prediction function
async function predict(imageElement) {
    // Preprocess the image to match the input size of the model
    const processedImage = tf.tidy(() => {
        let tensorImg = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([48, 48]) // Resize the image
            .toFloat()
            .div(tf.scalar(255.0)) // Normalize the image
            .expandDims(); // Add batch dimension
        return tensorImg;
    });

    // Make a prediction
    const prediction = await model.predict(processedImage).data();
    processedImage.dispose(); // Dispose the tensor to release memory

    // Find the index of the max value in the prediction
    const maxIndex = prediction.indexOf(Math.max(...prediction));

    return maxIndex;
}

// Event listener for the predict button
document.getElementById('predictButton').addEventListener('click', async () => {
    const predictionResult = document.getElementById('prediction');
    const imageUpload = document.getElementById('imageUpload');

    if (imageUpload.files.length > 0) {
        const imageElement = predictionResult.getElementsByTagName('img')[0]; // Get the preview image
        predictionResult.innerText = 'Predicting...'; // Provide feedback
        try {
            const predictedIndex = await predict(imageElement);
            console.log('Predicted Index:', predictedIndex); // Debugging: Log the predicted index
            const classLabel = getClassLabel(predictedIndex);
            console.log('Class Label:', classLabel); // Debugging: Log the class label
            if (classLabel === undefined) {
                // If classLabel is undefined, log the entire classIndices object for inspection
                console.log('Class Indices:', classIndices);
            }
            predictionResult.innerText = `Prediction: ${classLabel}`;
        } catch (error) {
            console.error(error);
            predictionResult.innerText = 'Error predicting image.';
        }
    } else {
        predictionResult.innerText = 'Please upload an image first.';
    }
});
