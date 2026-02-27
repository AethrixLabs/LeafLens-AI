/* LeafLens Mandi Page – modular, no inline JS */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    form: $("#mandi-form"),
    cropSelect: $("#mandi-crop"),
    languageSelect: $("#mandi-language"),
    submitBtn: $("#mandi-btn"),
    spinner: $("#mandi-spinner"),
    alert: $("#mandi-alert"),
    resultCard: $("#mandi-result-card"),
    resultContent: $("#mandi-result-content"),
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
    els.submitBtn.disabled = isLoading || !els.cropSelect?.value;
    els.spinner.classList.toggle("hidden", !isLoading);
    els.submitBtn.querySelector(".btn-label").textContent = isLoading ? "Loading…" : "Get Prices";
  }

  function formatValue(val) {
    if (val === null || val === undefined) return "—";
    if (typeof val === "number" && !Number.isFinite(val)) return "—";
    return String(val);
  }

  // Translation labels for UI elements
  const translationLabels = {
    en: {
      averagePrice: "Average Price",
      totalRecords: "Total Records",
      bestMarket: "Best Market",
      worstMarket: "Worst Market",
      marketOverview: "Market Overview",
    },
    hi: {
      averagePrice: "औसत भाव",
      totalRecords: "कुल रिकॉर्ड",
      bestMarket: "सर्वश्रेष्ठ बाजार",
      worstMarket: "सबसे खराब बाजार",
      marketOverview: "बाजार का सारांश",
    },
    od: {
      averagePrice: "ଗଡ଼ ମୂଲ୍ୟ",
      totalRecords: "ମୋଟ ରେକର୍ଡ",
      bestMarket: "ସର୍ବୋତ୍ତମ ବାଜାର",
      worstMarket: "ସବୁଠାରୁ ଖରାପ ବାଜାର",
      marketOverview: "ବାଜାର ସମୀକ୍ଷା",
    },
  };

  // Crops that are typically traded per quintal (100 kg)
  const quintalCrops = new Set(["wheat", "rice", "corn", "maize"]);

  function getDisplayUnit(cropId, backendUnit) {
    // If crop is in quintal crops list, always show as quintal
    if (quintalCrops.has((cropId || "").toLowerCase())) {
      return "quintal";
    }
    // Otherwise use backend unit
    return backendUnit || "kg";
  }

  function getTranslationLabel(key, languageCode) {
    const labels = translationLabels[languageCode] || translationLabels.en;
    return labels[key] || translationLabels.en[key];
  }

  async function loadCrops() {
    if (!els.cropSelect) return;
    els.cropSelect.innerHTML = '<option value="" selected disabled>Loading crops…</option>';

    try {
      const res = await fetch("/api/mandi/crops", { method: "GET" });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        els.cropSelect.innerHTML = '<option value="" selected disabled>Failed to load crops</option>';
        return;
      }

      const crops = Array.isArray(data) ? data : [];
      els.cropSelect.innerHTML = "";
      if (!crops.length) {
        els.cropSelect.innerHTML = '<option value="" selected disabled>No crops available</option>';
        return;
      }

      for (const crop of crops) {
        const opt = document.createElement("option");
        const id = crop.id || crop;
        const name = typeof crop === "object" ? (crop.name || id) : id;
        opt.value = String(id);
        opt.textContent = String(name);
        els.cropSelect.appendChild(opt);
      }

      els.submitBtn.disabled = false;
    } catch {
      els.cropSelect.innerHTML = '<option value="" selected disabled>Failed to load crops</option>';
    }
  }

  function renderMandi(data) {
    if (!data || !data.success) {
      els.resultContent.innerHTML = '<div class="muted">No mandi data available.</div>';
      return;
    }

    const stats = data.statistics || {};
    const best = data.best_market || null;
    const worst = data.worst_market || null;
    const backendUnit = data.unit || "kg";
    const cropId = data.crop_id || "";
    const languageCode = els.languageSelect?.value || "en";

    // Get the correct display unit (override for crops typically sold per quintal)
    const displayUnit = getDisplayUnit(cropId, backendUnit);

    // Get translated labels
    const labelAveragePrice = getTranslationLabel("averagePrice", languageCode);
    const labelTotalRecords = getTranslationLabel("totalRecords", languageCode);
    const labelBestMarket = getTranslationLabel("bestMarket", languageCode);
    const labelWorstMarket = getTranslationLabel("worstMarket", languageCode);
    const labelMarketOverview = getTranslationLabel("marketOverview", languageCode);

    // Update dashboard title with translated format
    document.getElementById("mandiTitle").innerText =
      data.crop_name + " - " + labelMarketOverview;

    // Update average price with correct unit and translated label
    const avgPriceElement = document.getElementById("avgPrice");
    const avgValue = formatValue(stats.average_price ?? stats.average ?? "--");
    avgPriceElement.innerText = avgValue + " ₹/" + displayUnit;
    avgPriceElement.parentElement.querySelector(".mandi-label").innerText = labelAveragePrice;

    // Update total records with translated label
    const totalRecElement = document.getElementById("totalRecords");
    totalRecElement.innerText = formatValue(stats.total_records ?? stats.count ?? "--");
    totalRecElement.parentElement.querySelector(".mandi-label").innerText = labelTotalRecords;

    // Update best market with correct unit and translated labels
    const bestMarketElement = document.getElementById("bestMarket");
    const bestPriceElement = document.getElementById("bestPrice");
    bestMarketElement.innerText = best?.mandi_name ?? "N/A";
    bestMarketElement.parentElement.querySelector(".market-title").innerText = labelBestMarket;
    bestPriceElement.innerText = best ? best.price + " ₹/" + displayUnit : "--";

    // Update worst market with correct unit and translated labels
    const worstMarketElement = document.getElementById("worstMarket");
    const worstPriceElement = document.getElementById("worstPrice");
    worstMarketElement.innerText = worst?.mandi_name ?? "N/A";
    worstMarketElement.parentElement.querySelector(".market-title").innerText = labelWorstMarket;
    worstPriceElement.innerText = worst ? worst.price + " ₹/" + displayUnit : "--";

    els.resultCard.classList.remove("hidden");
  }

  function renderError(message) {
    els.resultContent.innerHTML = `<div class="muted">${message || "Failed to fetch mandi data."}</div>`;
    els.resultCard.classList.remove("hidden");
  }

  async function fetchMandi() {
    setAlert("");
    const cropId = els.cropSelect?.value;
    if (!cropId) {
      setAlert("Please select a crop.");
      return;
    }

    const languageCode = els.languageSelect?.value || "en";
    const url = `/api/mandi/${encodeURIComponent(cropId)}?language_code=${encodeURIComponent(languageCode)}`;

    setLoading(true);
    els.resultCard.classList.add("hidden");

    try {
      const res = await fetch(url, { method: "GET" });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        const errMsg = data?.error || data?.message || `Request failed (${res.status})`;
        setAlert(errMsg);
        renderError(errMsg);
        return;
      }

      renderMandi(data);
    } catch (err) {
      const msg = "Could not reach mandi service. Please try again.";
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
        fetchMandi();
      });
    }
    if (els.cropSelect) {
      els.cropSelect.addEventListener("change", () => {
        const isLoading = !els.spinner.classList.contains("hidden");
        els.submitBtn.disabled = isLoading || !els.cropSelect.value;
      });
    }
    // Ensure language changes trigger a re-render of existing data
    if (els.languageSelect) {
      els.languageSelect.addEventListener("change", () => {
        // If we have existing result data, re-render with the new language
        // This will update all the translated labels
        const resultCard = document.getElementById("mandi-result-card");
        if (resultCard && !resultCard.classList.contains("hidden")) {
          // Get the last crop that was selected
          const cropId = els.cropSelect?.value;
          if (cropId) {
            fetchMandi();
          }
        }
      });
    }
  }

  function boot() {
    if (els.year) els.year.textContent = String(new Date().getFullYear());
    wireEvents();
    loadCrops();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
