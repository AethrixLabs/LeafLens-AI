// LeafLens Insight client-side script
// Provides functions to load and render IoT sensor + 3-day forecast

const endpoint = "/api/insight-data";

function qs(id) {
  return document.getElementById(id);
}

async function loadInsightData(location) {
  showError(null);
  showLoading(true);
  try {
    const url = `${endpoint}?location=${encodeURIComponent(location)}`;
    const resp = await fetch(url, { method: "GET" });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error: ${resp.status} ${text}`);
    }

    const data = await resp.json();

    // Expect structure: { status, location, sensor_data, forecast_3_days, timestamp }
    const sensor = data?.sensor_data || {};
    renderSensorData(sensor);
    renderForecast(data.forecast_3_days || []);
  } catch (err) {
    console.error("loadInsightData error", err);
    showError(err.message || "Failed to load data");
  } finally {
    showLoading(false);
  }
}

function renderSensorData(data) {
  const t = data?.temperature;
  const h = data?.humidity;
  const s = data?.soil_moisture;
  const idx = data?.stress_index;

  qs("sensor-temp").textContent = t != null ? `${t} °C` : "— °C";
  qs("sensor-humidity").textContent = h != null ? `${h} %` : "— %";
  qs("sensor-soil").textContent = s != null ? `${s} %` : "— %";
  qs("sensor-stress").textContent = idx != null ? `${idx}` : "—";
}

function renderForecast(forecast) {
  const container = qs("forecast-list");
  container.innerHTML = "";

  if (!Array.isArray(forecast) || forecast.length === 0) {
    container.textContent = "No forecast available.";
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "forecast-card-wrapper";

  forecast.forEach((day) => {
    const card = document.createElement("div");
    card.className = "forecast-card";

    card.innerHTML = `
      <div class="forecast-date">${day.date}</div>
      <div class="forecast-item">
        <span>🌡 Temp</span>
        <strong>${day.avg_temp} °C</strong>
      </div>
      <div class="forecast-item">
        <span>💧 Humidity</span>
        <strong>${day.avg_humidity} %</strong>
      </div>
      <div class="forecast-item">
        <span>🌧 Rain</span>
        <strong>${day.total_rain_mm} mm</strong>
      </div>
    `;

    wrapper.appendChild(card);
  });

  container.appendChild(wrapper);
}

function showLoading(show) {
  const spinner = qs("insight-spinner");
  const btn = qs("insight-refresh");
  if (spinner) spinner.classList.toggle("hidden", !show);
  if (btn) btn.disabled = !!show;
}

function showError(message) {
  const alertEl = qs("insight-alert");
  if (!alertEl) return;
  if (!message) {
    alertEl.classList.add("hidden");
    alertEl.textContent = "";
    return;
  }
  alertEl.classList.remove("hidden");
  alertEl.classList.add("alert-error");
  alertEl.textContent = message;
}

// Wiring UI events on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const loc = qs("location-select");
  const refresh = qs("insight-refresh");

  async function reload() {
    const city = loc?.value || "Bhubaneswar";
    await loadInsightData(city);
  }

  if (loc) {
    loc.addEventListener("change", () => {
      reload();
    });
  }

  if (refresh) {
    refresh.addEventListener("click", () => {
      reload();
    });
  }

  // Initial load with default
  reload().catch((e) => console.error(e));
});
