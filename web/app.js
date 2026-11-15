/* Farmer-facing app. Model and advisory values are read from the local API. */
"use strict";
const API = window.LEAFWISE_API_URL || `${location.protocol}//${location.hostname}:8000`;
const $ = (id) => document.getElementById(id);
const form = $("upload-form");
const input = $("photo");
const languagePicker = $("language");
let strings = {};
let selectedFile = null;
let currentImageUrl = null;
let latestPrediction = null;
let historyLimit = 8;
let activeExplainJob = null;
const HISTORY_KEY = "leafwise.scan-history.v1";

function safe(value) {
  const span = document.createElement("span");
  span.textContent = String(value ?? "");
  return span.innerHTML;
}
function locale() { return strings[languagePicker.value] || strings.en || {}; }
function t(key, values = {}) {
  let text = locale()[key] ?? strings.en?.[key] ?? key;
  for (const [name, value] of Object.entries(values)) text = text.replaceAll(`{${name}}`, String(value));
  return text;
}
function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "Not available";
}
function humanClass(value) {
  const [crop = "", condition = ""] = String(value || "").split("___", 2);
  const human = (part) => part.replaceAll("_", " ").replaceAll(",", "").trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
  return [human(crop), human(condition)].filter(Boolean).join(" · ");
}
async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let detail = `API returned ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* no JSON body */ }
    throw new Error(detail);
  }
  return response.json();
}
function setStatus(message, state = "") {
  const target = $("api-status");
  target.textContent = message;
  target.className = state;
}
async function initialize() {
  try {
    [strings,] = await Promise.all([fetch("/explanation_strings.json").then((r) => r.json())]);
    const settings = await api("/ui-config");
    historyLimit = Number(settings.browser_history_limit) || historyLimit;
    const health = await api("/health");
    setStatus(health.model_loaded
      ? `Prediction service ready · serving version ${safe(health.model_version || "unknown")}`
      : `API connected; model unavailable. ${health.registry_error || "Predictions are unavailable until a verified model is mounted."}`,
    health.model_loaded ? "ready" : "warning");
    const classData = await api("/classes");
    if (classData.available) {
      for (const crop of classData.crops || []) {
        const option = document.createElement("option");
        option.value = crop.key;
        option.textContent = crop.label;
        $("crop").append(option);
      }
    }
  } catch (error) {
    setStatus(`Prediction service is unavailable: ${error.message}. The page remains available; retry after starting the API.`, "warning");
  }
  await loadSamples();
  renderHistory();
  applyLanguage();
}
function applyLanguage() {
  const selected = ["en", "mr"].includes(localStorage.getItem("leafwise.language"))
    ? localStorage.getItem("leafwise.language") : "en";
  languagePicker.value = selected;
  document.documentElement.lang = selected === "mr" ? "mr" : selected === "hi" ? "hi" : "en";
  $("disclaimer").textContent = t("disclaimer");
  if (latestPrediction) renderPrediction(latestPrediction, false);
}
languagePicker.addEventListener("change", () => {
  localStorage.setItem("leafwise.language", languagePicker.value);
  applyLanguage();
});
async function loadSamples() {
  const picker = $("demo-sample");
  try {
    const response = await fetch("/demo_samples.json");
    if (!response.ok) throw new Error("Sample list is not present.");
    for (const sample of await response.json()) {
      if (typeof sample.filename !== "string" || /[\\/]/.test(sample.filename)) continue;
      const option = document.createElement("option");
      option.value = sample.filename;
      option.textContent = sample.label || sample.filename;
      picker.append(option);
    }
  } catch {
    picker.disabled = true;
    picker.options[0].textContent = "Sample photos are not included";
  }
}
function setSelectedFile(file) {
  if (!file) return;
  selectedFile = file;
  if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
  currentImageUrl = URL.createObjectURL(file);
  const preview = $("preview");
  preview.src = currentImageUrl;
  preview.hidden = false;
  $("drop-zone").classList.add("has-image");
  $("replace-photo").hidden = false;
  $("idle").hidden = false;
  $("result").hidden = true;
  $("error").hidden = true;
  latestPrediction = null;
  abortExplanation();
}
input.addEventListener("change", () => setSelectedFile(input.files?.[0]));
$("replace-photo").addEventListener("click", () => input.click());
$("demo-sample").addEventListener("change", async () => {
  if (!$("demo-sample").value) return;
  try {
    const response = await fetch(`/demo_samples/${encodeURIComponent($("demo-sample").value)}`);
    if (!response.ok) throw new Error("Sample photo could not be loaded.");
    const name = $("demo-sample").value;
    const file = new File([await response.blob()], name, { type: "image/jpeg" });
    setSelectedFile(file);
  } catch (error) {
    showError(error.message);
  }
});
const drop = $("drop-zone");
for (const eventName of ["dragenter", "dragover"]) {
  drop.addEventListener(eventName, (event) => { event.preventDefault(); drop.classList.add("dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault(); drop.classList.remove("dragging");
    if (eventName === "drop" && event.dataTransfer.files?.[0]) setSelectedFile(event.dataTransfer.files[0]);
  });
}
function showError(message) {
  $("loading").hidden = true;
  $("error").textContent = `${message} Check the connection and try again.`;
  $("error").hidden = false;
  $("result").hidden = true;
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) { showError("Choose a leaf photo first."); return; }
  $("error").hidden = true;
  $("result").hidden = true;
  $("idle").hidden = true;
  $("loading").hidden = false;
  $("submit").disabled = true;
  document.querySelector(".result-card").setAttribute("aria-busy", "true");
  try {
    const data = new FormData();
    data.append("file", selectedFile, selectedFile.name);
    data.append("crop", $("crop").value);
    if ($("temperature").value !== "") data.append("temperature_c", $("temperature").value);
    if ($("humidity").value !== "") data.append("humidity_pct", $("humidity").value);
    const prediction = await api("/predict", { method: "POST", body: data });
    latestPrediction = prediction;
    renderPrediction(prediction);
    await saveHistory(prediction, selectedFile);
  } catch (error) {
    showError(error.message);
    $("idle").hidden = false;
  } finally {
    $("loading").hidden = true;
    $("submit").disabled = false;
    document.querySelector(".result-card").setAttribute("aria-busy", "false");
  }
});
function imageUrl(base64) { return `data:image/png;base64,${base64}`; }
function renderPrediction(payload, clearExplain = true) {
  latestPrediction = payload;
  if (clearExplain) abortExplanation();
  const display = payload.display || {};
  const advice = payload.advisory || {};
  const trace = payload.advice_trace || {};
  const confidence = Number(payload.confidence);
  const confidenceValue = Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0;
  const state = display.diagnosis_state || "retake_photo";
  const stateText = state === "healthy" ? "Healthy" : state === "disease_likely" ? "Disease likely" : "Not sure — retake photo";
  const className = [display.crop?.label, display.condition?.label].filter(Boolean).join(" · ") || "No diagnosis returned";
  const band = display.confidence_band || "uncertain";
  const bandText = band.charAt(0).toUpperCase() + band.slice(1);
  const conditionText = state === "healthy" ? t("health_healthy") : state === "disease_likely" ? t("health_disease") : t("retake");
  const gradcam = payload.gradcam_png_base64
    ? `<div class="comparison-grid"><figure><figcaption>Uploaded photo</figcaption><img class="evidence-image" src="${safe(currentImageUrl)}" alt="Uploaded leaf photo"></figure><figure><figcaption>Grad-CAM for this prediction</figcaption><img class="evidence-image" src="${imageUrl(payload.gradcam_png_base64)}" alt="Grad-CAM attribution for this image"></figure></div><p>${safe(t("gradcam"))}</p>`
    : `<div class="notice info">Grad-CAM was not returned for this prediction.</div>`;
  const sourceUrl = /^https?:\/\//i.test(trace.source_url || "") ? trace.source_url : null;
  const source = sourceUrl ? `<a href="${safe(sourceUrl)}" target="_blank" rel="noopener noreferrer">Open cited source</a>` : "Source page not available";
  const localizedItems = (english, translated) => {
    const source = Array.isArray(english) ? english : [];
    const draft = Array.isArray(translated) ? translated : [];
    const translatedAllowed = languagePicker.value === "mr"
      && advice.translation_status === "MACHINE_DRAFT" && draft.length > 0;
    const values = translatedAllowed ? draft : source;
    return {
      values: values.map((entry) => typeof entry === "string" ? entry : entry.text || entry.quote)
        .filter(Boolean),
      fellBack: languagePicker.value === "mr" && source.length > 0 && !translatedAllowed,
    };
  };
  const scoutingView = localizedItems(advice.scouting, advice.scouting_mr);
  const culturalView = localizedItems(advice.cultural, advice.cultural_mr);
  const scouting = scoutingView.values;
  const cultural = culturalView.values;
  const translationFallback = scoutingView.fellBack || culturalView.fellBack ||
    (languagePicker.value === "mr" && (advice.chemical_options || []).length > 0
      && !(advice.chemical_options_mr || []).length) ||
    (languagePicker.value === "mr" && Boolean(advice.label_reminder)
      && !advice.label_reminder_mr);
  const adviceStatus = advice.status || (trace.evidence_grade === "C" ? "PLACEHOLDER" : "VERIFIED");
  const adviceText = adviceStatus === "PLACEHOLDER" ? t("advice_placeholder") :
    trace.evidence_grade === "C" ? t("advice_missing") :
      languagePicker.value === "mr"
        ? t(trace.tier === "full" ? "advice_full" : "advice_scouting")
        : (advice.message || t("advice_missing"));
  const weather = advice.weather_status === "not_provided" ? t("weather_add") :
    advice.weather_status === "rule_matched" ? advice.weather_note : t("weather_unknown");
  const weatherCitation = /^https?:\/\//i.test(advice.weather_source_citation || "")
    ? `<a href="${safe(advice.weather_source_citation)}" target="_blank" rel="noopener noreferrer">Weather-rule source</a>`
    : "";
  const confusableHints = Array.isArray(payload.confusable_hints) ? payload.confusable_hints : [];
  const hintsHtml = confusableHints.length
    ? `<div class="notice info"><strong>Possible class overlap</strong><ul>${confusableHints.map((item) => {
      const other = item.label || item.may_be_confused_with || item.class_name || item.other_class || "another class";
      const passage = item.how_to_tell_apart && item.source_url
        ? `<p>${safe(item.how_to_tell_apart)} <a href="${safe(item.source_url)}" target="_blank" rel="noopener noreferrer">Source passage</a></p>` : "";
      return `<li>This class may be confused with ${safe(humanClass(other))}.${passage}</li>`;
    }).join("")}</ul></div>` : "";
  const coverage = display.coverage_warning
    ? `<aside class="notice warning">${safe(t("coverage", { condition: display.coverage_warning.available_condition || "the only represented condition" }))}</aside>` : "";
  const draft = languagePicker.value !== "en" && advice.translation_status === "MACHINE_DRAFT"
    ? `<p class="notice warning">${safe(t("draft_notice"))}</p>` : "";
  const topK = (payload.top_k || []).slice(1).map((item) => `<li>${safe(humanClass(item.label))} · ${safe(pct(item.prob))}</li>`).join("");
  const adviceItems = [...scouting, ...cultural];
  const chemical = trace.tier === "full" && Array.isArray(advice.chemical_options) ? advice.chemical_options : [];
  const chemicalHtml = chemical.length ? `<h4>Chemical options quoted by source</h4><ul class="advice-list">${chemical.map((item) => `<li>${safe(typeof item === "string" ? item : item.quote || item.text)}</li>`).join("")}</ul>` : "";
  const bg = payload.bg_check_details || {};
  const bgImage = bg.neutralized_image_base64 ? `<img class="evidence-image" src="${imageUrl(bg.neutralized_image_base64)}" alt="Current image with background replaced by its mean colour">` : "";
  const bgChange = Number.isFinite(Number(bg.confidence_change)) ? `${Number(bg.confidence_change) >= 0 ? "+" : ""}${(Number(bg.confidence_change) * 100).toFixed(1)} percentage points` : "not available";
  const xaiTabs = `<div class="evidence-tabs" role="tablist" aria-label="Explanation method"><button type="button" role="tab" data-method="gradcam" aria-selected="true">Grad-CAM</button><button type="button" role="tab" data-method="lime" aria-selected="false">LIME</button><button type="button" role="tab" data-method="shap" aria-selected="false">SHAP</button></div><div id="evidence-pane" class="evidence-pane">${gradcam}</div>`;
  $("result").innerHTML = `
    ${languagePicker.value === "en" ? "" : `<p class="notice warning">${safe(t("draft_notice"))}</p>`}
    <div class="result-head"><div><p class="eyebrow">Result from this photo</p><h2>${safe(className)}</h2><p>${safe(t("condition_intro"))}</p></div><span class="status-badge ${safe(state)}">${safe(stateText)}</span></div>
    ${display.show_diagnosis ? `<div class="confidence-row"><strong>Model confidence</strong><div class="gauge" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(confidenceValue * 100)}"><span style="width:${Math.round(confidenceValue * 100)}%"></span></div><span class="band-badge">${safe(pct(confidence))} · ${safe(bandText)}${display.confidence_band_provisional ? " · provisional band" : ""}</span></div>` : `<div class="notice warning">${safe(t("retake"))}</div>`}
    <p class="footnote">This is the model’s own confidence, not a guarantee of correctness.</p>
    ${coverage}
    <section class="result-section"><h3>What to do next</h3><p>${safe(conditionText)}</p>${state !== "healthy" ? `<p>${safe(t("consult"))}</p>` : `<p>${safe(t("next_inspect"))}</p>`}${hintsHtml}</section>
    <section class="result-section"><h3>Why did the model say this?</h3><p>Highlighted areas show image regions associated with this output. They do not prove the diagnosis or explain its cause.</p>${xaiTabs}</section>
    <details class="result-section"><summary>Advanced checks</summary><p>Background sensitivity: <b>${safe(payload.bg_check || "unknown")}</b>. Neutral colour replacement confidence change: ${safe(bgChange)}.</p>${bgImage}</details>
    <section class="result-section"><h3>Advice and its source</h3>${draft}${translationFallback ? `<p class="footnote">${safe(t("advice_fallback"))}</p>` : ""}<div class="passport"><span class="advice-badge">${safe(adviceStatus)}</span><p>${safe(adviceText)}</p><p>Pathogen group: ${safe(display.pathogen_type || "not recorded")}</p>${adviceItems.length ? `<ul class="advice-list">${adviceItems.map((item) => `<li>${safe(typeof item === "string" ? item : item.text || item.quote)}</li>`).join("")}</ul>` : ""}${chemicalHtml}<p>Weather note: ${safe(weather)} ${weatherCitation}</p><p>${source}</p>${advice.label_reminder ? `<p>Source label reminder: ${safe(typeof advice.label_reminder === "string" ? advice.label_reminder : advice.label_reminder.quote || "")}</p>` : ""}</div><details class="technical"><summary>Technical details</summary><p>Advice tier: ${safe(trace.tier || "not available")} · Evidence grade: ${safe(trace.evidence_grade || "not available")}</p><p>Entry: ${safe(trace.kb_entry_id || "not available")} · Retrieved: ${safe(trace.retrieved_on || "not available")}</p><p>Knowledge base version: ${safe(trace.kb_version || payload.kb_version || "not available")}</p><p>Rules: ${safe((trace.rules_fired || []).join(", ") || "none returned")}</p><p>Background check: ${safe(trace.bg_check || "unknown")}</p></details></section>
    ${topK ? `<details class="result-section"><summary>Other possible model matches</summary><ol class="advice-list">${topK}</ol></details>` : ""}
    <p class="footnote">${safe(payload.disclaimer || t("disclaimer"))}</p>`;
  $("loading").hidden = true;
  $("idle").hidden = true;
  $("error").hidden = true;
  $("result").hidden = false;
  $("result").querySelectorAll("[role=tab]").forEach((button) => button.addEventListener("click", () => selectExplainMethod(button.dataset.method)));
}
async function selectExplainMethod(method) {
  document.querySelectorAll("[role=tab]").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.method === method)));
  const pane = $("evidence-pane");
  if (method === "gradcam") {
    const image = latestPrediction?.gradcam_png_base64;
    pane.innerHTML = image
      ? `<div class="comparison-grid"><figure><figcaption>Uploaded photo</figcaption><img class="evidence-image" src="${safe(currentImageUrl)}" alt="Uploaded leaf photo"></figure><figure><figcaption>Grad-CAM for this prediction</figcaption><img class="evidence-image" src="${imageUrl(image)}" alt="Grad-CAM attribution for this image"></figure></div><p>${safe(t("gradcam"))}</p>`
      : `<div class="notice info">Grad-CAM was not returned for this prediction.</div>`;
    return;
  }
  if (!selectedFile || !latestPrediction) { pane.innerHTML = `<div class="notice info">Check a leaf photo before requesting an explanation.</div>`; return; }
  abortExplanation();
  const methodText = method.toUpperCase();
  pane.innerHTML = `<div class="notice info" role="status">${safe(t("xai_wait"))}<p id="xai-state">Starting ${safe(methodText)}…</p><button class="text-button" id="cancel-xai" type="button">Cancel explanation</button></div>`;
  $("cancel-xai").addEventListener("click", abortExplanation);
  try {
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    formData.append("method", method);
    formData.append("crop", $("crop").value);
    const created = await api("/explain/jobs", { method: "POST", body: formData });
    activeExplainJob = created.job_id;
    await pollExplainJob(created.job_id, method);
  } catch (error) {
    pane.innerHTML = `<div class="notice error">${safe(error.message)} ${safe(t("xai_error"))}</div>`;
  }
}
async function pollExplainJob(jobId, method) {
  const started = Date.now();
  while (activeExplainJob === jobId) {
    await new Promise((resolve) => setTimeout(resolve, 750));
    if (activeExplainJob !== jobId) return;
    let job;
    try { job = await api(`/explain/jobs/${encodeURIComponent(jobId)}`); }
    catch (error) { $("evidence-pane").innerHTML = `<div class="notice error">${safe(error.message)}</div>`; activeExplainJob = null; return; }
    const status = $("xai-state");
    if (status) status.textContent = `${job.status} · ${Math.round((Date.now() - started) / 1000)} s`;
    if (job.status === "done") {
      const confidence = pct(job.confidence);
      $("evidence-pane").innerHTML = `<div class="comparison-grid"><figure><figcaption>Model input crop</figcaption><img class="evidence-image" src="${safe(currentImageUrl)}" alt="Original uploaded leaf photo"></figure><figure><figcaption>${safe(method.toUpperCase())} for ${safe(humanClass(job.predicted_class))} · ${safe(confidence)}</figcaption><img class="evidence-image" src="${imageUrl(job.overlay_png_base64)}" alt="${safe(method.toUpperCase())} attribution overlay for current upload"></figure></div><p>${safe(t(method))}</p><p class="footnote">Explanation runtime returned by API: ${safe(job.runtime_ms)} ms. This method describes model attribution, not disease confirmation.</p>`;
      activeExplainJob = null;
      return;
    }
    if (["failed", "timeout"].includes(job.status)) {
      $("evidence-pane").innerHTML = `<div class="notice error">${safe(job.message || t("xai_error"))} (${safe(job.reason_code || job.status)})<button class="text-button" id="retry-xai" type="button">Try again</button></div>`;
      $("retry-xai").addEventListener("click", () => selectExplainMethod(method));
      activeExplainJob = null;
      return;
    }
  }
}
async function abortExplanation() {
  const jobId = activeExplainJob;
  activeExplainJob = null;
  if (jobId) {
    try { await api(`/explain/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" }); } catch { /* cancellation is best effort */ }
  }
}
async function thumbnail(file) {
  try {
    const source = await createImageBitmap(file);
    const scale = Math.min(180 / source.width, 120 / source.height, 1);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(source.width * scale));
    canvas.height = Math.max(1, Math.round(source.height * scale));
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    source.close();
    return canvas.toDataURL("image/jpeg", 0.65);
  } catch { return ""; }
}
async function saveHistory(payload, file) {
  const display = payload.display || {};
  const item = {
    thumb: await thumbnail(file),
    crop: display.crop?.label || "Crop not returned",
    condition: display.condition?.label || (display.diagnosis_state === "retake_photo" ? "Retake photo" : "Condition not returned"),
    status: display.diagnosis_state || "unknown",
    savedAt: new Date().toISOString(),
  };
  try {
    const previous = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    localStorage.setItem(HISTORY_KEY, JSON.stringify([item, ...previous].slice(0, historyLimit)));
  } catch { /* unavailable storage does not block diagnosis */ }
  renderHistory();
}
function renderHistory() {
  let rows = [];
  try { rows = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { rows = []; }
  const host = $("history-list");
  if (!rows.length) { host.innerHTML = `<p class="empty-state">${safe(t("empty_history"))}</p>`; return; }
  host.innerHTML = rows.map((row) => `<article class="history-item">${row.thumb ? `<img src="${safe(row.thumb)}" alt="Saved scan thumbnail">` : ""}<strong>${safe(row.crop)} · ${safe(row.condition)}</strong><time datetime="${safe(row.savedAt)}">${safe(new Date(row.savedAt).toLocaleString())}</time><span class="status-badge ${safe(row.status)}">${safe(row.status.replaceAll("_", " "))}</span></article>`).join("");
}
$("clear-history").addEventListener("click", () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); });
initialize();
