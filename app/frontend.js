const PREDICT_URL = 'http://localhost:5000/predict';
const WEATHER_URL = 'http://localhost:5000/weather';
let chart = null;
let data = []; // Initialize data array to store rows
let selectedDateText = null;

dayjs.extend(dayjs_plugin_dayOfYear);
dayjs.extend(dayjs_plugin_isoWeek);

// Function to send the request
async function getPrediction(data) {
  try {
    const response = await fetch(PREDICT_URL, {
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
    
    // Create chart if predictions exist
    if (result.prediction && Array.isArray(result.prediction)) {
      createChart(result.prediction);
    }
    
    return result;

  } catch (error) {
    console.error('Error during prediction:', error);
  }
}

async function getWeatherForecast(){
  try {
    const response = await fetch(WEATHER_URL, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const result = await response.json();
    console.log('Weather Forecast:', result);
    
    return result;

  } catch (error) {
    console.error('Error getting weather:', error);
  }
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

// Initialize Flatpickr
flatpickr("#dateRangePicker", {
    dateFormat: "Y-m-d",
    onChange: function(selectedDates, dateStr, instance) {
        selectedDateText = dateStr;
        document.getElementById('startDateOutput').textContent = dateStr;
    }
});

function parseDateIntoColumns(date){
    const d = dayjs(date);

    return{
      hour : d.hour(),
      dayOfMonth : d.date(),
      month : d.month() + 1,
      year : d.year(),
      dayOfWeek : d.day(),
      dayOfYear : d.dayOfYear(),
      weekOfYear : d.isoWeek()
    }
}

function mergeRowData(dateText){
  const startDate = dayjs(dateText);  // dateText = 'YYYY-MM-DD'
  const hoursToLoop = 168; // 7 days * 24 hours

  for(let i = 0; i < hoursToLoop; i++){
    const currentDate = startDate.add(i, 'hour');
    const dateColumns = parseDateIntoColumns(currentDate);
    
    // Create row object with date columns and placeholder values
    const row = {
      hour: dateColumns.hour,
      dayofweek: dateColumns.dayOfWeek,
      month: dateColumns.month,
      year: dateColumns.year,
      dayofyear: dateColumns.dayOfYear,
      dayofmonth: dateColumns.dayOfMonth,
      weekofyear: dateColumns.weekOfYear,
      seasonNum: 2, // placeholder
      Temperature: 10, // placeholder
      averageTemp: 10 // placeholder
    };
    
    data.push(row);
    if(i == hoursToLoop){
      endDate = currentDate;
      updateEndDate(endDate);
    }
  }
}

function updateEndDate(endDate){
  const endDateElement = document.getElementById('endDateOutput');
  if(endDateElement && endDate) {
    endDateElement.textContent = endDate.format('YYYY-MM-DD');
  }
}

function showDateError(message) {
  // Create or update error display
  let errorDiv = document.getElementById('dateError');
  if (!errorDiv) {
    errorDiv = document.createElement('div');
    errorDiv.id = 'dateError';
    errorDiv.style.color = 'red';
    errorDiv.style.marginTop = '10px';
    document.getElementById('dateRangePicker').parentNode.appendChild(errorDiv);
  }
  errorDiv.textContent = message;
  
  // Clear error after 5 seconds
  setTimeout(() => {
    errorDiv.textContent = '';
  }, 5000);
}



document.getElementById('submitDateButton').addEventListener('click', function (e) {
  e.preventDefault();
  
  if (!selectedDateText) {
    showDateError('Please select a date first');
    return;
  }
  
  // Clear existing data and add new rows
  data = [];
  mergeRowData(selectedDateText);
  
  console.log('Generated data:', data);
  
  // Call prediction with the generated data
  getPrediction(data);

  getWeatherForecast();
});