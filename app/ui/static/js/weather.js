/* LeafLens Weather Page – modular, no inline JS */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    form: $("#weather-form"),
    cityInput: $("#weather-city"),
    languageSelect: $("#weather-language"),
    submitBtn: $("#weather-btn"),
    spinner: $("#weather-spinner"),
    alert: $("#weather-alert"),
    resultCard: $("#weather-result-card"),
    resultContent: $("#weather-result-content"),
    year: $("#year"),
  };

  function setAlert(message) {
    if (!message) {
      els.alert.classList.add("hidden");
      els.alert.textContent = "";
      return;
    }
    els.alert.textContent = message;
    els.alert.classList.remove("hidden");
  }

  function setLoading(isLoading) {
    els.submitBtn.disabled = isLoading;
    els.spinner.classList.toggle("hidden", !isLoading);
    els.submitBtn.querySelector(".btn-label").textContent = isLoading ? "Loading…" : "Get Weather";
  }

  function formatValue(val) {
    if (val === null || val === undefined) return "—";
    if (typeof val === "number" && !Number.isFinite(val)) return "—";
    return String(val);
  }

  function renderWeather(data) {
    if (!data || !data.success) {
      els.resultContent.innerHTML = '<div class="muted">No weather data available.</div>';
      els.resultContent.classList.remove("hidden");
      document.getElementById("weather-report")?.classList.add("hidden");
      els.resultCard.classList.remove("hidden");
      return;
    }

    document.getElementById("weather-report")?.classList.remove("hidden");
    els.resultContent.classList.add("hidden");
    els.resultContent.innerHTML = "";

    const weather = data.weather || {};
    const loc = data.location || {};

    document.getElementById("tempValue").innerText =
      (weather.temperature ?? weather.temp ?? "--") + "°C";
    document.getElementById("humidityValue").innerText =
      (weather.humidity ?? "--") + "%";
    document.getElementById("rainValue").innerText =
      (weather.rainfall ?? weather.rain ?? "--") + " mm";
    document.getElementById("windValue").innerText =
      (weather.wind_speed ?? weather.wind ?? "--") + " m/s";
    document.getElementById("cloudValue").innerText =
      (weather.clouds ?? "--") + "%";
    document.getElementById("conditionValue").innerText =
      weather.condition ?? "--";
    document.getElementById("weatherTitle").innerText =
      (loc.city ?? loc.region ?? "Weather") + " - Weather Report";

    els.resultCard.classList.remove("hidden");
  }

  function renderError(message) {
    els.resultContent.innerHTML = `<div class="muted">${message || "Failed to fetch weather data."}</div>`;
    els.resultContent.classList.remove("hidden");
    document.getElementById("weather-report")?.classList.add("hidden");
    els.resultCard.classList.remove("hidden");
  }

  async function fetchWeather() {
    setAlert("");
    const location = (els.cityInput?.value || "").trim();
    const languageCode = els.languageSelect?.value || "en";

    const params = new URLSearchParams();
    if (location) params.set("location", location);
    params.set("language_code", languageCode);

    const url = `/api/weather?${params.toString()}`;
    setLoading(true);
    els.resultCard.classList.add("hidden");

    try {
      const res = await fetch(url, { method: "GET" });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        const errMsg = data?.error || `Request failed (${res.status})`;
        setAlert(errMsg);
        renderError(errMsg);
        return;
      }

      renderWeather(data);
    } catch (err) {
      const msg = "Could not reach weather service. Please try again.";
      setAlert(msg);
      renderError(msg);
    } finally {
      setLoading(false);
    }
  }

  function wireEvents() {
    if (els.form) {
      els.form.addEventListener("submit", (e) => {
        e.preventDefault();
        fetchWeather();
      });
    }
  }

  function boot() {
    if (els.year) els.year.textContent = String(new Date().getFullYear());
    wireEvents();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
