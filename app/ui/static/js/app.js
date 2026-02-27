/* LeafLens UI – no inline JS; production-oriented defensive handling */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    year: $("#year"),
    alert: $("#alert"),
    toastRoot: $("#toast-root"),

    form: $("#predict-form"),
    cropSelect: $("#crop-select"),
    dropZone: $("#drop-zone"),
    fileInput: $("#file-input"),
    previewImg: $("#preview-image"),
    fileMeta: $("#file-meta"),

    predictBtn: $("#predict-btn"),
    predictSpinner: $("#predict-spinner"),
    resetBtn: $("#reset-btn"),

    resultCard: $("#result-card"),
    predictedLabel: $("#predicted-label"),
    confidenceText: $("#confidence-text"),
    confidenceBar: $("#confidence-bar"),
    confidenceChip: $("#confidence-chip"),
    predictionIdPill: $("#prediction-id-pill"),
    cachedBadge: $("#cached-badge"),

    toggleProbs: $("#toggle-probs"),
    probsSection: $("#probs-section"),
    probsList: $("#probs-list"),
    toggleGradcam: $("#toggle-gradcam"),
    gradcamSection: $("#gradcam-section"),
    gradcamImg: $("#gradcam-image"),

    explainCard: $("#explain-card"),
    explainSpinner: $("#explain-spinner"),
    explainSimpleBtn: $("#explain-simple-btn"),
    explainDetailedBtn: $("#explain-detailed-btn"),
    explainContent: $("#explain-content"),
    aiModeToggle: $("#ai-mode-toggle"),
    aiModeBadge: $("#ai-mode-badge"),
    languageSelector: $("#language-selector"),
    languageSelect: $("#language-select"),

    feedbackCorrect: $("#feedback-correct"),
    feedbackIncorrect: $("#feedback-incorrect"),
    feedbackStatus: $("#feedback-status"),

    statusRow: $("#status-pill-row"),

    analyticsBlock: $("#analytics-block"),
    analyticsModel: $("#analytics-model"),
    analyticsTime: $("#analytics-time"),
    analyticsDevice: $("#analytics-device"),
    analyticsHash: $("#analytics-hash"),
  };

  /** App state */
  const state = {
    selectedFile: null,
    selectedFileUrl: null,
    predictionId: null,
    lastExplanationRequestMs: null,
    feedbackSubmitted: false,
    lastInferenceMs: null,
    aiMode: false,
    llmEnabled: false,
    selectedLanguage: "English",
  };

  const CONF_THRESH = { good: 0.8, mid: 0.5 };
  const MAX_FILE_BYTES = 5 * 1024 * 1024; // 5MB client-side safety limit

  function setAlert(message) {
    if (!message) {
      els.alert.classList.add("hidden");
      els.alert.textContent = "";
      return;
    }
    els.alert.textContent = message;
    els.alert.classList.remove("hidden");
  }

  function toast(type, title, message) {
    const node = document.createElement("div");
    node.className = `toast ${type || ""}`.trim();
    node.innerHTML = `
      <div class="toast-title"></div>
      <div class="toast-msg"></div>
    `;
    node.querySelector(".toast-title").textContent = title || "Notice";
    node.querySelector(".toast-msg").textContent = message || "";
    els.toastRoot.appendChild(node);

    // Auto-remove after a bit
    const ttl = type === "error" ? 6200 : 4200;
    window.setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(8px)";
      node.style.transition = "200ms ease";
      window.setTimeout(() => node.remove(), 240);
    }, ttl);
  }

  function formatPct(x) {
    if (typeof x !== "number" || !Number.isFinite(x)) return "—";
    return `${Math.round(x * 100)}%`;
  }

  function classifyConfidence(conf) {
    if (typeof conf !== "number" || !Number.isFinite(conf)) return "neutral";
    if (conf >= CONF_THRESH.good) return "good";
    if (conf >= CONF_THRESH.mid) return "mid";
    return "bad";
  }

  function setLoading(isLoading) {
    els.predictBtn.disabled = isLoading;
    els.resetBtn.disabled = isLoading;
    els.predictSpinner.classList.toggle("hidden", !isLoading);
    els.predictBtn.querySelector(".btn-label").textContent = isLoading ? "Predicting…" : "Predict";
  }

  function setExplainLoading(isLoading) {
    els.explainSpinner.classList.toggle("hidden", !isLoading);
    els.explainSimpleBtn.disabled = isLoading || !state.predictionId;
    els.explainDetailedBtn.disabled = isLoading || !state.predictionId;
  }

  function clearResults() {
    state.predictionId = null;
    state.feedbackSubmitted = false;
    els.resultCard.classList.add("hidden");
    els.explainCard.classList.add("hidden");
    els.cachedBadge.classList.add("hidden");
    els.probsSection.classList.add("hidden");
    els.gradcamSection.classList.add("hidden");
    els.feedbackStatus.textContent = "";
    els.feedbackCorrect.disabled = false;
    els.feedbackIncorrect.disabled = false;
    if (els.analyticsBlock) {
      els.analyticsBlock.classList.add("hidden");
      els.analyticsModel.textContent = "—";
      els.analyticsTime.textContent = "—";
      els.analyticsDevice.textContent = "—";
      els.analyticsHash.textContent = "—";
    }
  }

  function setSelectedFile(file) {
    if (file && file.size > MAX_FILE_BYTES) {
      toast("error", "File too large", "Please upload an image under 5MB.");
      file = null;
    }

    state.selectedFile = file || null;
    if (state.selectedFileUrl) URL.revokeObjectURL(state.selectedFileUrl);
    state.selectedFileUrl = null;

    if (!file) {
      els.previewImg.classList.add("hidden");
      els.previewImg.src = "";
      els.fileMeta.classList.add("hidden");
      els.fileMeta.textContent = "";
      return;
    }

    state.selectedFileUrl = URL.createObjectURL(file);
    els.previewImg.src = state.selectedFileUrl;
    els.previewImg.classList.remove("hidden");
    els.fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;
    els.fileMeta.classList.remove("hidden");
  }

  function toggleSection(sectionEl, btnEl, expandedLabel, collapsedLabel) {
    const nowHidden = !sectionEl.classList.contains("hidden");
    sectionEl.classList.toggle("hidden", nowHidden);
    if (btnEl) btnEl.textContent = nowHidden ? collapsedLabel : expandedLabel;
  }

  function renderProbabilities(probabilities) {
    els.probsList.innerHTML = "";
    if (!probabilities || typeof probabilities !== "object") {
      els.probsList.textContent = "No probability distribution available.";
      return;
    }

    const rows = Object.entries(probabilities)
      .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
      .sort((a, b) => b[1] - a[1]);

    if (!rows.length) {
      els.probsList.textContent = "No probability distribution available.";
      return;
    }

    for (const [label, p] of rows) {
      const wrap = document.createElement("div");
      wrap.className = "prob-row";
      wrap.innerHTML = `
        <div class="prob-top">
          <div></div>
          <div></div>
        </div>
        <div class="prob-bar"><div></div></div>
      `;
      wrap.querySelector(".prob-top > div:first-child").textContent = label;
      wrap.querySelector(".prob-top > div:last-child").textContent = formatPct(p);

      const inner = wrap.querySelector(".prob-bar > div");
      // delay so transition animates
      window.setTimeout(() => {
        inner.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
      }, 0);
      els.probsList.appendChild(wrap);
    }
  }

  function renderExplanation(payload, detailedRequested, durationMs) {
    const base = {
      summary: payload?.summary,
      cause: payload?.cause,
      symptoms: payload?.symptoms,
      spread: payload?.spread,
      treatment: payload?.treatment,
      prevention: payload?.prevention,
      crop: payload?.crop,
      disease: payload?.disease,
    };

    const baseOk =
      typeof base.summary === "string" &&
      typeof base.cause === "string" &&
      typeof base.symptoms === "string" &&
      typeof base.treatment === "string";

    const detailedText = payload?.detailed_explanation;

    // Cached badge heuristic: fast response usually implies DB/cache hit.
    const cached = typeof durationMs === "number" && durationMs < 350;
    els.cachedBadge.classList.toggle("hidden", !cached);

    if (!baseOk) {
      els.explainContent.innerHTML = `<div class="muted">No explanation available.</div>`;
      return;
    }

    const parts = [
      ["Summary", base.summary],
      ["Cause", base.cause],
      ["Symptoms", base.symptoms],
      ["Spread", base.spread],
      ["Treatment", base.treatment],
      ["Prevention", base.prevention],
    ];

    let detailedBlock = "";
    if (detailedRequested) {
      if (typeof detailedText === "string" && detailedText.trim().length > 0) {
        detailedBlock = `
          <div class="kv-item">
            <div class="kv-key">AI detailed explanation</div>
            <div class="kv-val">${escapeHtml(detailedText)}</div>
          </div>
        `;
      } else {
        detailedBlock = `
          <div class="kv-item">
            <div class="kv-key">AI detailed explanation</div>
            <div class="kv-val muted">
              Detailed AI explanation is currently unavailable. You are seeing the knowledge-base explanation.
            </div>
          </div>
        `;
      }
    }

    const header = `
      <div class="kv-item">
        <div class="kv-key">Context</div>
        <div class="kv-val mono">Crop: ${escapeHtml(String(base.crop || "—"))} • Disease: ${escapeHtml(
      String(base.disease || "—"),
    )}</div>
      </div>
    `;

    const kbParagraphs = parts
      .map(([k, v]) => {
        const val = typeof v === "string" ? v.trim() : "";
        if (!val) return "";
        return `<p><strong>${escapeHtml(k)}.</strong> ${escapeHtml(val)}</p>`;
      })
      .join("");

    const userPrompt = detailedRequested
      ? "Explain this prediction in detail using AI."
      : "Explain this prediction in simple, farmer-friendly terms.";

    const aiBaseBubble = `
      <div class="chat-bubble ai">
        <div class="chat-role">AI • Knowledge base</div>
        <div class="chat-body">
          <p><strong>Context.</strong> Crop: ${escapeHtml(String(base.crop || "—"))}, disease: ${escapeHtml(
            String(base.disease || "—"),
          )}.</p>
          ${kbParagraphs || "<p>No structured explanation is available for this prediction.</p>"}
        </div>
      </div>
    `;

    let detailedBubble = "";
    if (detailedRequested) {
      if (typeof detailedText === "string" && detailedText.trim().length > 0) {
        detailedBubble = `
          <div class="chat-bubble ai">
            <div class="chat-role">AI • Gemini</div>
            <div class="chat-body">
              <p>${escapeHtml(detailedText)}</p>
            </div>
          </div>
        `;
      } else {
        detailedBubble = `
          <div class="chat-bubble ai">
            <div class="chat-role">AI • Gemini</div>
            <div class="chat-body">
              <p class="muted">
                A detailed AI explanation is currently unavailable. You are seeing the knowledge-base explanation only.
              </p>
            </div>
          </div>
        `;
      }
    }

    els.explainContent.innerHTML = `
      <div class="chat">
        <div class="chat-bubble user">
          <div class="chat-role">User</div>
          <div class="chat-body">
            <p>${escapeHtml(userPrompt)}</p>
          </div>
        </div>
        ${aiBaseBubble}
        ${detailedBubble}
      </div>
    `;
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function safeJson(res) {
    try {
      return await res.json();
    } catch {
      return null;
    }
  }

  async function loadCrops() {
    els.cropSelect.innerHTML = `<option value="" selected disabled>Loading crops…</option>`;
    try {
      const res = await fetch("/crops", { method: "GET" });
      const data = await safeJson(res);
      const crops = Array.isArray(data?.crops) ? data.crops : [];

      const fallback = ["rice"];
      const list = crops.length ? crops : fallback;

      els.cropSelect.innerHTML = "";
      for (const crop of list) {
        const opt = document.createElement("option");
        const cropStr = String(crop);
        opt.value = cropStr;
        opt.textContent = cropStr.charAt(0).toUpperCase() + cropStr.slice(1);
        els.cropSelect.appendChild(opt);
      }


      // Select first crop by default.
      if (els.cropSelect.options.length) els.cropSelect.value = els.cropSelect.options[0].value;

      // Ensure Predict button is enabled now that a crop is selected
      if (els.cropSelect.options.length && els.cropSelect.value) {
        els.predictBtn.disabled = false;
      }

      if (!crops.length && data?.warning) {
        toast("warn", "Using default crop", "Model registry crops could not be loaded yet.");
      }
    } catch (err) {
      els.cropSelect.innerHTML = `<option value="rice">Rice</option>`;
      // enable predict now that fallback crop exists
      els.predictBtn.disabled = false;
      toast("warn", "Offline mode", "Could not load crop list. Using a default option.");
    }
  }

  async function loadSystemStatus() {
    if (!els.statusRow) return;
    els.statusRow.innerHTML = "";
    try {
      const res = await fetch("/system-status", { method: "GET" });
      const data = await safeJson(res);
      const modelLoaded = !!data?.model_loaded;
      const kbLoaded = !!data?.knowledge_base_loaded;
      const llmEnabled = !!data?.llm_enabled;

      state.llmEnabled = llmEnabled;

      const mkPill = (text, kind) => {
        const div = document.createElement("div");
        div.className = `status-pill ${kind}`.trim();
        div.innerHTML = `<span class="status-dot"></span><span>${text}</span>`;
        return div;
      };

      els.statusRow.appendChild(mkPill(modelLoaded ? "Model Loaded" : "Model Unavailable", modelLoaded ? "ok" : "off"));
      els.statusRow.appendChild(
        mkPill(kbLoaded ? "Knowledge Base Active" : "Knowledge Base Unavailable", kbLoaded ? "ok" : "off"),
      );
      els.statusRow.appendChild(mkPill(llmEnabled ? "LLM Enabled" : "LLM Optional", llmEnabled ? "ok" : "warn"));
    } catch {
      // Silent failure – status row just stays empty.
    }
  }

  async function predict() {
    setAlert("");

    if (!state.selectedFile) {
      setAlert("Please upload an image before predicting.");
      toast("error", "Missing image", "Choose or drop a leaf image to continue.");
      return;
    }

    const crop = els.cropSelect.value;
    if (!crop) {
      setAlert("Please select a crop.");
      toast("error", "Missing crop", "Select a crop before predicting.");
      return;
    }

    clearResults();
    setLoading(true);

    try {
      const fd = new FormData();
      fd.append("file", state.selectedFile);

      const start = performance.now();
      const res = await fetch(`/predict/${encodeURIComponent(crop)}`, {
        method: "POST",
        body: fd,
      });
      const duration = performance.now() - start;

      const data = await safeJson(res);

      if (!res.ok) {
        const detail =
          (typeof data?.detail === "string" && data.detail) ||
          (typeof data?.error === "string" && data.error) ||
          `Request failed (${res.status})`;

        if (Array.isArray(data?.available_crops) && data.available_crops.length) {
          toast("warn", "Unsupported crop", `Available crops: ${data.available_crops.join(", ")}`);
        }

        setAlert(detail);
        toast("error", "Prediction failed", detail);
        return;
      }

      const label = data?.label ?? "—";
      const confidence = data?.confidence;
      const predictionId = data?.prediction_id ?? null;
      const probs = data?.probabilities ?? null;
      const gradcam = data?.gradcam_image ?? null;
      const modelVersion = data?.model_version ?? "v1.0";
      const device = data?.device ?? "Unknown";
      const imageHash = data?.image_hash ?? null;

      state.predictionId = typeof predictionId === "number" ? predictionId : null;
      state.lastInferenceMs = duration;

      els.predictedLabel.textContent = String(label);
      els.confidenceText.textContent = formatPct(confidence);

      const cls = classifyConfidence(confidence);
      els.confidenceChip.classList.remove("chip-good", "chip-mid", "chip-bad", "chip-neutral");
      els.confidenceBar.classList.remove("mid", "bad");
      if (cls === "good") {
        els.confidenceChip.classList.add("chip-good");
        els.confidenceChip.textContent = "High";
      } else if (cls === "mid") {
        els.confidenceChip.classList.add("chip-mid");
        els.confidenceChip.textContent = "Medium";
        els.confidenceBar.classList.add("mid");
      } else if (cls === "bad") {
        els.confidenceChip.classList.add("chip-bad");
        els.confidenceChip.textContent = "Low";
        els.confidenceBar.classList.add("bad");
      } else {
        els.confidenceChip.classList.add("chip-neutral");
        els.confidenceChip.textContent = "—";
      }

      const pct = typeof confidence === "number" && Number.isFinite(confidence) ? confidence * 100 : 0;
      els.confidenceBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;

      els.predictionIdPill.textContent = state.predictionId ? `Prediction #${state.predictionId}` : "Not logged";

      if (els.analyticsBlock) {
        els.analyticsModel.textContent = String(modelVersion);
        els.analyticsDevice.textContent = String(device);
        els.analyticsTime.textContent = `${Math.round(duration)} ms`;
        els.analyticsHash.textContent = imageHash ? String(imageHash).slice(0, 8) : "—";
        els.analyticsBlock.classList.remove("hidden");
      }

      renderProbabilities(probs);
      els.toggleProbs.textContent = "Probability distribution";
      els.probsSection.classList.add("hidden");

      if (typeof gradcam === "string" && gradcam.length) {
        els.gradcamImg.src = `data:image/jpeg;base64,${gradcam}`;
      } else {
        els.gradcamImg.removeAttribute("src");
      }
      els.toggleGradcam.textContent = "Grad-CAM";
      els.gradcamSection.classList.add("hidden");

      els.resultCard.classList.remove("hidden");
      els.explainCard.classList.remove("hidden");
      els.explainContent.innerHTML = `<div class="muted">Choose an explanation mode above.</div>`;

      toast("success", "Prediction complete", `Inference finished in ${Math.round(duration)}ms.`);
    } catch (err) {
      const msg = "Unexpected error during prediction. Please try again.";
      setAlert(msg);
      toast("error", "Prediction error", msg);
    } finally {
      setLoading(false);
    }
  }

  // Find your existing fetchExplanation() function and replace it with this:

async function fetchExplanation(detailedRequested) {
  if (!state.predictionId) {
    toast("warn", "No prediction", "Run a prediction first.");
    return;
  }

  setExplainLoading(true);
  showTypingIndicator();

  const startTs = performance.now();

  try {
    // Build query safely
    const params = new URLSearchParams();

    // detailed flag (true/false)
    params.append("detailed", detailedRequested ? "true" : "false");

    // Append language ONLY when:
    // 1. Detailed explanation requested
    // 2. AI Mode is enabled
    // 3. Language exists
    if (detailedRequested && state.aiMode && state.selectedLanguage) {
      params.append("language", state.selectedLanguage);
    }

    const url = `/explain-advanced/${encodeURIComponent(
      state.predictionId
    )}?${params.toString()}`;

    const res = await fetch(url, { method: "GET" });
    const data = await safeJson(res);

    if (!res.ok) {
      const msg =
        (typeof data?.detail === "string" && data.detail) ||
        `Request failed (${res.status})`;
      throw new Error(msg);
    }

    // Ensure typing animation is visible at least briefly
    await ensureTypingMinimum(startTs);
    hideTypingIndicator();

    renderExplanation(
      data,
      detailedRequested,
      performance.now() - startTs
    );
  } catch (err) {
    hideTypingIndicator();
    toast(
      "error",
      "Explanation failed",
      "Could not load explanation. Please try again."
    );
    console.error("Explanation error:", err);
  } finally {
    setExplainLoading(false);
  }
}
  async function submitFeedback(value) {
    setAlert("");
    if (!state.predictionId) {
      toast("warn", "No prediction yet", "Run a prediction before submitting feedback.");
      return;
    }
    if (state.feedbackSubmitted) return;

    els.feedbackCorrect.disabled = true;
    els.feedbackIncorrect.disabled = true;

    try {
      const res = await fetch(`/feedback/${encodeURIComponent(state.predictionId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback: value }),
      });
      const data = await safeJson(res);

      if (!res.ok) {
        const detail =
          (typeof data?.detail === "string" && data.detail) || `Request failed (${res.status})`;
        els.feedbackCorrect.disabled = false;
        els.feedbackIncorrect.disabled = false;
        setAlert(detail);
        toast("error", "Feedback failed", detail);
        return;
      }

      state.feedbackSubmitted = true;
      els.feedbackStatus.textContent = "Thanks! Your feedback has been recorded.";
      toast("success", "Feedback recorded", "Thank you for helping improve LeafLens.");
    } catch {
      els.feedbackCorrect.disabled = false;
      els.feedbackIncorrect.disabled = false;
      const msg = "Unexpected error while submitting feedback.";
      setAlert(msg);
      toast("error", "Feedback error", msg);
    }
  }

  function wireEvents() {
    els.form.addEventListener("submit", (e) => {
      e.preventDefault();
      predict();
    });

    els.resetBtn.addEventListener("click", () => {
      setAlert("");
      clearResults();
      setSelectedFile(null);
      els.fileInput.value = "";
      toast("success", "Reset", "Ready for a new prediction.");
    });

    // Drag & drop UX
    const dz = els.dropZone;
    dz.addEventListener("dragover", (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      const file = e.dataTransfer?.files?.[0] || null;
      if (file) {
        setSelectedFile(file);
        clearResults();
        toast("success", "Image selected", "Ready to run prediction.");
      }
    });
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        els.fileInput.click();
      }
    });

    els.fileInput.addEventListener("change", () => {
      const file = els.fileInput.files?.[0] || null;
      if (file) {
        setSelectedFile(file);
        clearResults();
      }
    });

    // Enable Predict when a crop is selected
    if (els.cropSelect) {
      els.cropSelect.addEventListener("change", () => {
        els.predictBtn.disabled = !els.cropSelect.value;
      });
    }

    els.toggleProbs.addEventListener("click", () => {
      toggleSection(els.probsSection, els.toggleProbs, "Hide probability distribution", "Probability distribution");
    });

    els.toggleGradcam.addEventListener("click", () => {
      toggleSection(els.gradcamSection, els.toggleGradcam, "Hide Grad-CAM", "Grad-CAM");
    });

    els.explainSimpleBtn.addEventListener("click", () => fetchExplanation(false));
    els.explainDetailedBtn.addEventListener("click", () => {
      if (!state.aiMode) {
        toast("warn", "Enable AI Mode", "Turn on AI Mode to request detailed AI explanations.");
        return;
      }
      fetchExplanation(true);
    });

    els.feedbackCorrect.addEventListener("click", () => submitFeedback("correct"));
    els.feedbackIncorrect.addEventListener("click", () => submitFeedback("incorrect"));

    if (els.aiModeToggle) {
      els.aiModeToggle.addEventListener("change", () => {
        state.aiMode = !!els.aiModeToggle.checked;
        if (state.aiMode) {
          els.explainCard.classList.add("card-ai-active");
          els.aiModeBadge.classList.remove("hidden");
          els.languageSelector.classList.remove("hidden");
        } else {
          els.explainCard.classList.remove("card-ai-active");
          els.aiModeBadge.classList.add("hidden");
          els.languageSelector.classList.add("hidden");
        }
      });
    }

    if (els.languageSelect) {
      els.languageSelect.addEventListener("change", () => {
        state.selectedLanguage = els.languageSelect.value || "English";
      });
    }
  }

  function showTypingIndicator() {
    if (!els.explainContent) return;
    els.explainContent.innerHTML = `
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    `;
  }

  function hideTypingIndicator() {
    if (!els.explainContent) return;
    const node = els.explainContent.querySelector(".typing-indicator");
    if (node && node.parentElement) {
      node.parentElement.removeChild(node);
    }
  }

  async function ensureTypingMinimum(startTs) {
    const elapsed = performance.now() - startTs;
    if (elapsed < 400) {
      await new Promise((resolve) => setTimeout(resolve, 400 - elapsed));
    }
  }

  function boot() {
  if (els.year) els.year.textContent = String(new Date().getFullYear());

  setAlert("");
  clearResults();
  setSelectedFile(null);

  if (els.aiModeToggle) {
    state.aiMode = !!els.aiModeToggle.checked;
  }

  if (els.languageSelect) {
    state.selectedLanguage = els.languageSelect.value || "English";
  }

  if (els.predictBtn) els.predictBtn.disabled = true;

  wireEvents();
  loadCrops();
  loadSystemStatus();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}

})();