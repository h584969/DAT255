document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const imageInput = document.getElementById('image-input');
    if (imageInput.files.length > 0) {
        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        fetch('/api/v1/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayResult(data);
        })
        .catch(error => {
            console.error('Error fetching the prediction:', error);
        });
    }
});

function displayResult(resultData) {
    const resultDisplay = document.getElementById('result-display');
    resultDisplay.innerHTML = `
        <p>Name: ${resultData.name}</p>
        <p>Score: ${Math.round(resultData.amt*100,2)}%</p>
    `;
}
