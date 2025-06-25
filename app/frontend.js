const API_URL = 'http://localhost:5000/predict';
let chart = null;

// Function to send the request
async function getPrediction(data) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)  // send inputData
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const result = await response.json();
    console.log('Prediction result:', result);
    
    // Display results in the pre element
    document.getElementById('result').textContent = JSON.stringify(result, null, 2);
    
    // Create chart if predictions exist
    if (result.prediction && Array.isArray(result.prediction)) {
      createChart(result.prediction);
    }
    
    return result;

  } catch (error) {
    console.error('Error during prediction:', error);
  }
}

function csvToJSON(csv) {
  const lines = csv.trim().split('\n');
   const headers = lines[0].split(',').map(h => h.trim());

  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());     // Strip whitespace from values
    return Object.fromEntries(headers.map((h, i) => [h, parseFloat(values[i])]));
  });
}

function createChart(predictions) {
  const ctx = document.getElementById('predictionChart').getContext('2d');
  
  // Destroy existing chart if it exists
  if (chart) {
    chart.destroy();
  }

  // Create labels (just indices 1 to 169)
  const labels = predictions.map((_, index) => index + 1);

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predictions',
        data: predictions,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderWidth: 2,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Data Point'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Prediction Value'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: `Predictions (${predictions.length} values)`
        }
      }
    }
  });
}

document.getElementById('csv-file').addEventListener('change', function (e) {
  const file = e.target.files[0];
  const reader = new FileReader();

  reader.onload = function (event) {
    const csv = event.target.result;
    const data = csvToJSON(csv);
    getPrediction(data);  // call your prediction API
  };

  reader.readAsText(file);
});