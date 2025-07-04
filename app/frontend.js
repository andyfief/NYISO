const PREDICT_URL = 'http://localhost:5000/predict';
const WEATHER_URL = 'http://localhost:5000/weather';
let chart = null;
let weatherChart = null;
let data = []; // Initialize data array to store rows
let forecastDict = {};
let selectedDateText = null;

dayjs.extend(dayjs_plugin_dayOfYear);
dayjs.extend(dayjs_plugin_isoWeek);
dayjs.extend(dayjs_plugin_utc);

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

async function getWeatherForecast() {
  try {
    const response = await fetch(WEATHER_URL, { method: 'GET' });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const result = await response.json();
    console.log('Weather Forecast:', result);

    forecastDict = {};

    result.hourly_forecast.forEach(entry => {
      const localTimestamp = entry.timestamp;
      const utcTimestamp = dayjs(localTimestamp).utc().format(); // ISO UTC string
      forecastDict[utcTimestamp] = entry.temperature;
    });

    console.log("Forecast in UTC:", forecastDict);
    return forecastDict;

  } catch (error) {
    console.error('Error getting weather:', error);
    return {};
  }
}

function getChartOptions(title) {
  return {
    responsive: false,
    maintainAspectRatio: false,
    layout: {
      padding: 10
    },
    interaction: {
      intersect: false,
      mode: 'index'
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date & Time'
        },
        ticks: {
          callback: function(value, index, values) {
            return index % 12 === 0 ? this.getLabelForValue(value) : '';
          },
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        title: {
          display: true,
          text: title.includes('Temperature') ? 'Temperature (°C)' : 'Prediction Value'
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: title
      }
    }
  };
}

async function renderWeatherChart() {
  const labels = Object.keys(forecastDict);
  const temperatures = Object.values(forecastDict);

  const ctx = document.getElementById("weatherChart").getContext("2d");

  // Destroy existing chart if it exists
  if (weatherChart) {
    weatherChart.destroy();
  }

  // Force canvas to maintain container size
  const container = ctx.canvas.parentElement;
  ctx.canvas.width = container.clientWidth;
  ctx.canvas.height = container.clientHeight;
  
  weatherChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Hourly Temperature (°C)',
        data: temperatures,
        borderColor: '#9EB2FA',
        backgroundColor: 'rgb(255, 255, 255)',
        fill: true,
        tension: 0.3
      }]
    },
    options: getChartOptions('7-Day Hourly Weather Forecast (NWS)')
  });
}

function createChart(predictions) {
  const ctx = document.getElementById('predictionChart').getContext('2d');
  
  // Destroy existing chart if it exists
  if (chart) {
    chart.destroy();
  }

  // Force canvas to maintain container size
  const container = ctx.canvas.parentElement;
  ctx.canvas.width = container.clientWidth;
  ctx.canvas.height = container.clientHeight;

  const startDate = dayjs(selectedDateText);
  const labels = predictions.map((_, index) => {
    return startDate.add(index, 'hour').format('MM/DD HH:mm');
  });

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predictions',
        data: predictions,
        borderColor: '#DEBB87',
        backgroundColor: '#faefdf',
        borderWidth: 2,
        fill: true
      }]
    },
    options: getChartOptions(`7-Day NYC Power Grid Load Predictions (MWh)`)
  });
}

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
  selectedDateText = dateText;
  const startDate = dayjs.utc(dateText);  // dateText = 'YYYY-MM-DD'
  console.log(startDate.format())
  const hoursToLoop = 156; // 7 days * 24 hours - 12 hours from the weather API missing these

  const dailyTemps = {};

  data = [];
  for(let i = 0; i < hoursToLoop; i++){
    const currentDate = startDate.add(i, 'hour');
    const timestampKey = currentDate.format();

    const tempStr = forecastDict[timestampKey];
    const temp = tempStr != null ? parseFloat(tempStr) : null;
    const finalTemp = (temp != null && !isNaN(temp)) ? temp : 0;

    const dateColumns = parseDateIntoColumns(currentDate);

    const dateKey = currentDate.format('YYYY-MM-DD');
    const hour = currentDate.hour();

    if (hour >= 9 && hour <= 21) {
      if (!dailyTemps[dateKey]) {
        dailyTemps[dateKey] = [];
      }
      dailyTemps[dateKey].push(finalTemp);
    }

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
      Temperature: finalTemp,
      averageTemp: null // placeholder
    };
    data.push(row);

    if(i == hoursToLoop - 1){
      endDate = currentDate;
      updateEndDate(endDate);
    }

    data.forEach((row, index) => {
      if (index < 12) {
        row.averageTemp = row.Temperature;
      } else {
        const dateStr = `${row.year}-${String(row.month).padStart(2, '0')}-${String(row.dayofmonth).padStart(2, '0')}`;
        const temps = dailyTemps[dateStr];

        if (temps && temps.length > 0) {
          const avg = temps.reduce((a, b) => a + b, 0) / temps.length;
          row.averageTemp = parseFloat(avg.toFixed(2));
        } else {
          row.averageTemp = row.Temperature; // fallback
        }
      }
    });
  }
}

function setTodaysDate() {
  // Get all keys (timestamps) from the dict and sort them to find earliest
  const timestamps = Object.keys(forecastDict);
  timestamps.sort(); // ISO timestamps sort correctly as strings

  // Pick first timestamp or fallback
  const firstTimestamp = timestamps[0];

  const startDate = firstTimestamp
    ? dayjs.utc(firstTimestamp)
    : dayjs.utc(selectedDateText || dayjs().format('YYYY-MM-DD'));

  document.getElementById('startDateOutput').textContent = startDate.format('YYYY-MM-DD HH:mm');
  return startDate;
}

function updateEndDate(endDate){
  if(endDate) {
    endDate = endDate.format('YYYY-MM-DD HH:mm');
    document.getElementById('endDateOutput').textContent = endDate;
  }
  else{
    console.log("error parsing end date")
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

document.getElementById('forecastButton').addEventListener('click', function (e) {
  const overlay = document.getElementById('weatherOverlay');
  if (overlay) {
    overlay.style.display = 'none';
  }
  renderWeatherChart(); // This overlay is a workaround for a quirk of chartJS resizing to the length of the graph that I can't figure out.
  // we render the chart once at  DOMContentLoaded, and cover it with this overlay. Then when the button is clicked, remove the overlay
  // and re-render so that the animation is visible.
});

document.addEventListener('DOMContentLoaded', async function () {
  console.log("Preloading weather data...");
  forecastDict = await getWeatherForecast();
  renderWeatherChart();
  const today = setTodaysDate();

  if(Object.keys(forecastDict).length === 0){
      console.log("No weather data");
      return
    }
    data = [];
    mergeRowData(today);

    console.log('Generated data:', data);

    getPrediction(data);

  document.getElementById('submitDateButton').addEventListener('click', function (e) {
  const overlay = document.getElementById('predictionOverlay');
  const weatherOverlay = document.getElementById('weatherOverlay');

  // Run prediction (which triggers chart creation & animation)
  getPrediction(data).then(() => {
    // One frame after the chart starts animating, remove the overlay
    requestAnimationFrame(() => {
      if (overlay && weatherOverlay && window.getComputedStyle(weatherOverlay).display === 'none') {
        overlay.style.display = 'none';
      }
    });
  });
});
});