"use strict";
const API = window.LEAFWISE_API_URL || location.protocol + "//" + location.hostname + ":8000";
const $ = (id) => document.getElementById(id);
let activeJob = null;
function esc(value) { const e = document.createElement("span"); e.textContent = String(value == null ? "" : value); return e.innerHTML; }
function fmt(value, digits) { const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits || 4) : "not available"; }
function pct(value) { const n = Number(value); return Number.isFinite(n) ? (n * 100).toFixed(2) + "%" : "not available"; }
async function get(path) { const response = await fetch(API + path); if (!response.ok) throw new Error(path + " returned " + response.status); return response.json(); }
function table(headers, rows) {
  if (!rows.length) return "<p class='empty-state'>No rows in this artifact.</p>";
  return "<div class='table-scroll'><table class='data-table'><thead><tr>" + headers.map((x) => "<th>" + esc(x) + "</th>").join("") + "</tr></thead><tbody>" +
    rows.map((r) => "<tr>" + r.map((x) => "<td>" + esc(x) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
}
function empty(id, message) { const node = $(id); if (node) node.textContent = message; }
function switchPanel(name) {
  document.querySelectorAll(".insights-panel").forEach((p) => { p.hidden = p.id !== "panel-" + name; });
  document.querySelectorAll(".insights-tabs [role=tab]").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.target === name)));
}
document.querySelectorAll(".insights-tabs [role=tab]").forEach((t) => t.addEventListener("click", () => switchPanel(t.dataset.target)));
document.querySelectorAll(".insights-nav a[href^='#']").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); switchPanel(a.hash.slice(1)); }));
function renderOverview(p) {
  const s = p.serving || {}, process = p.process || {}, log = p.log_window || {}, summary = log.summary || {};
  $("serving-card").innerHTML = "<b>" + (s.model_loaded ? "Version " + esc(s.model_version || "unknown") : "Model unavailable") + "</b><p>" + esc(s.model_source || "source unknown") + " · " + esc(s.device || "device unknown") + "</p><p>Health " + esc(s.status || "unknown") + " · registry " + esc(s.registry_match || "unknown") + "</p><p>Uptime " + esc(s.uptime_seconds == null ? "not available" : s.uptime_seconds + " seconds") + "</p>" + (s.registry_error ? "<p class='callout'>" + esc(s.registry_error) + "</p>" : "") + "<p><a href='http://localhost:5000' target='_blank' rel='noopener noreferrer'>MLflow UI (when locally running)</a></p>";
  $("process-card").innerHTML = "<b>" + esc(process.request_count == null ? "not available" : process.request_count) + " requests</b><p>This API process only; resets when the API restarts.</p><p>Mean latency " + esc(process.mean_latency_ms == null ? "not available" : process.mean_latency_ms) + " ms</p><p>Mean confidence " + esc(process.mean_confidence == null ? "not available" : process.mean_confidence) + "</p>";
  $("log-card").innerHTML = log.available ? "<b>" + esc(log.record_count) + " readable log entries</b><p>Returned log window, not process lifetime.</p><p>Status counts " + esc(JSON.stringify(summary.statuses || {})) + "</p><p>Traffic-source counts " + esc(JSON.stringify(summary.traffic_sources || {})) + "</p><p>Malformed rows " + esc(log.malformed_lines || 0) + (log.truncated ? " · older rows omitted" : "") + "</p>" : "<p>" + esc(log.message || "Request log not generated yet.") + "</p>";
  if (log.available) {
    const statuses = summary.statuses || {};
    const count = Number(log.record_count) || 0;
    const uncertain = Number(statuses.uncertain) || 0;
    const rejected = Object.entries(statuses).filter(([name]) => name.startsWith("rejected"))
      .reduce((sum, [, value]) => sum + (Number(value) || 0), 0);
    $("log-card").innerHTML += "<p>Within this returned log window: uncertain " +
      (count ? pct(uncertain / count) : "not available") + " · rejected " +
      (count ? pct(rejected / count) : "not available") + ".</p>";
  }
  const v = summary.latency_percentiles;
  $("percentiles-card").innerHTML = v ? "<b>p50 " + esc(v.p50_latency_ms) + " ms · p95 " + esc(v.p95_latency_ms) + " ms</b><p>Sample count " + esc(summary.percentile_sample_count) + "; configured minimum " + esc(summary.percentile_min_samples) + ".</p>" : "<p>Percentiles hidden until the configured minimum sample count is met. Current log sample count " + esc(summary.percentile_sample_count || 0) + ".</p>";
  const mlflow = p.mlflow || {};
  if (mlflow.run) {
    const run = mlflow.run;
    const params = Object.entries(run.params || {}).map(([key, value]) => [key, value]);
    const metrics = Object.entries(run.metrics || {}).map(([key, value]) => [key, value]);
    const started = Number.isFinite(Number(run.started_at_ms))
      ? new Date(Number(run.started_at_ms)).toLocaleString() : "not returned";
    $("mlflow-run").innerHTML = "<p><b>" + esc(run.experiment || "Experiment not returned") + "</b> · " +
      esc(run.status || "status unknown") + " · started " + esc(started) + "</p>" +
      (params.length ? "<h3>Allowlisted parameters</h3>" + table(["Parameter", "Value"], params) : "<p>No allowlisted parameters returned.</p>") +
      (metrics.length ? "<h3>Latest run metrics</h3>" + table(["Metric", "Value"], metrics) : "<p>No allowlisted metrics returned.</p>") +
      "<p>Read-only query to the configured MLflow tracking store. Path, run ID, hashes, and unreviewed parameter keys are omitted.</p>";
  } else {
    const messages = {
      not_configured: "MLflow tracking URI is not configured.",
      no_experiments: "No MLflow experiments are available in the configured store.",
      no_runs: "No MLflow runs are recorded in the configured store.",
      unavailable: "MLflow run details are unavailable (" + esc(mlflow.reason || "read error") + "). Open the local MLflow UI when it is running.",
      not_connected: "MLflow run details are not connected.",
    };
    $("mlflow-run").textContent = messages[mlflow.status] || "MLflow run details are not available.";
  }
}
function renderData(data) {
  const cov = data.coverage_artifact || {}, meta = cov.metadata || {};
  $("coverage-summary").innerHTML = data.class_count_from_serving_model == null ? "Serving class metadata is unavailable until a model loads." : "<b>" + esc(data.class_count_from_serving_model) + " classes · " + esc(data.crop_count_from_serving_model) + " crops</b><p>Counts from the loaded checkpoint; coverage file " + esc(meta.total_classes || "not available") + " classes, " + esc(meta.total_crops || "not available") + " crops.</p>";
  const split = data.split_counts || {};
  const splitRows = Object.entries(split.splits || {}).map(([key, value]) => [key.toUpperCase(), value.status === "available" ? value.rows + " rows" : value.status]);
  $("split-summary").innerHTML = splitRows.length ? table(["Split", "Manifest"], splitRows) : esc(split.status || "Split manifests not generated yet.");
  const counts = Object.entries(split.class_counts || {}).sort((a,b) => b[1]-a[1]);
  if (counts.length) {
    const max = Math.max(...counts.map((x) => x[1])), min = Math.min(...counts.map((x) => x[1]));
    $("class-bars").innerHTML = "<p>Training-manifest rows · largest/smallest observed class ratio " + fmt(max/min, 2) + ":1.</p>" + counts.map(([name,n]) => "<div class='bar-row'><span>" + esc(name.replace("___"," · ").replaceAll("_"," ")) + "</span><div class='bar-track'><span style='width:" + Math.max(1,n/max*100) + "%'></span></div><b>" + esc(n) + "</b></div>").join("");
  } else empty("class-bars", "TRAIN class distribution not generated yet.");
  const gaps = Object.entries(cov.crops || {}).filter(([,v]) => (v.classes || []).length === 1).map(([crop,v]) => crop + ": only " + String(v.classes[0]).split("___").pop().replaceAll("_"," ") + " class present");
  $("coverage-gaps").innerHTML = gaps.length ? "<ul>" + gaps.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul><p>Structural coverage from kb/coverage.yaml.</p>" : "Single-condition crop information not returned.";
  const audit = data.quality_audit;
  $("dataset-audit").innerHTML = audit
    ? "<p>Source reports/snapshot/dataset_audit.json</p><pre>" + esc(JSON.stringify(audit, null, 2)) + "</pre>"
    : "Dataset audit summary is not included in reports/snapshot. Duplicate counts, split-overlap results, and capture-source analysis are not inferred here.";
}
function renderModel(reports) {
  const e = reports.evaluation;
  if (!e) { ["eval-headline","calibration","crop-table","confusions"].forEach((id) => empty(id,"Evaluation snapshot not generated yet.")); }
  else {
    const h = e.headline || {};
    $("eval-headline").innerHTML = "<p><b>TEST snapshot · reports/snapshot/eval_summary.json</b></p><p>Accuracy " + pct(h.accuracy == null ? e.accuracy : h.accuracy) + " · macro-F1 " + pct(h.macro_f1 == null ? e.macro_f1 : h.macro_f1) + " · weighted F1 " + pct(h.weighted_f1 == null ? e.weighted_f1 : h.weighted_f1) + " · test count " + esc(h.n_test || "not in snapshot") + "</p>" + (e.within_crop_macro_f1_excluding_single_class_crops == null ? "" : "<p>Within-crop macro-F1 excluding single-class crops " + pct(e.within_crop_macro_f1_excluding_single_class_crops) + ".</p>") + "<p>These saved TEST values are separate from VAL release-gate metrics.</p>";
    const cal = e.calibration || {};
    $("calibration").innerHTML = "<p>ECE before temperature calibration " + fmt(cal.ece_before,4) + "</p><p>ECE after calibration " + fmt(cal.ece_after,4) + "</p><p>Source reports/snapshot/eval_summary.json. ECE is a calibration measure, not accuracy.</p>";
    const cropRows = Object.entries(e.per_crop || {}).sort((a,b) => Number(a[1].macro_f1)-Number(b[1].macro_f1)).map(([name,m]) => [name,pct(m.accuracy),pct(m.macro_f1),m.support,m.is_single_class ? "yes" : "no"]);
    $("crop-table").innerHTML = table(["Crop","Accuracy","Macro-F1","Support","Single class"],cropRows);
    const weak = (e.weakest_5_classes || []).map((m) => [m.class_name,pct(m.f1),pct(m.precision),pct(m.recall),m.support]);
    const pairs = (e.top_10_confused_pairs || []).map((m) => [m.true + " → " + m.pred,m.count]);
    $("confusions").innerHTML = (weak.length ? "<h3>Lowest class scores in saved report</h3>" + table(["Class","F1","Precision","Recall","Support"],weak) : "") + (pairs.length ? "<h3>Frequent confused pairs</h3>" + table(["True → predicted","Count"],pairs) : "<p>Confusion list not generated.</p>") + "<p>Confusion-matrix image is not included in the saved snapshot.</p>";
  }
  const seeds = reports.seed_robustness, models = seeds && seeds.models || {};
  $("seed-table").innerHTML = Object.keys(models).length ? "<p>Split " + esc(seeds.split || "not stated") + " · reports/snapshot/seed_variance.json.</p>" + table(["Model","Within-crop macro-F1","Minimum recall","ECE","Mean-fill drop"],Object.entries(models).map(([k,m]) => [k === "champion" ? "Champion" : k.replace("seed_","Seed "),pct(m.within_crop_macro_f1),pct(m.min_recall),fmt(m.ece,4),fmt(m.bg_drop_per_image_mean,4)])) + "<p>Three seeds are a small descriptive sample, per source note.</p>" : "Seed summary not generated yet.";
  const bg = reports.background_sensitivity;
  $("bg-rates").innerHTML = bg ? "<p>Source reports/snapshot/bg_check_rates.json.</p><pre>" + esc(JSON.stringify(bg,null,2)) + "</pre>" : "Background-sensitivity report not generated yet.";
}
function renderRelease(p) {
  const rows = (p.promotion_history || []).map((r) => [r.version == null ? (r.candidate_version || "not recorded") : r.version,r.verdict && r.verdict.verdict || r.tags && r.tags.gate_verdict || "not recorded",r.dry_run ? "Dry run" : "Applied/recorded",r.alias_moved == null ? "not recorded" : r.alias_moved,r.approval_valid == null ? "not recorded" : r.approval_valid,r.timestamp || "not recorded"]);
  $("promotion-table").innerHTML = rows.length ? table(["Candidate/version","Gate","Mode","Alias moved","Approval","Time"],rows) : "Promotion log not generated yet.";
  const gate = p.reports.candidate_gate;
  $("gate-summary").innerHTML = gate ? "<p>Source reports/snapshot/candidate_gate_comparison.json</p><pre>" + esc(JSON.stringify(gate,null,2)) + "</pre>" : "Candidate gate snapshot not generated yet.";
}
function renderMonitoring(p) {
  const process = p.process || {}, log = p.log_window || {};
  $("live-traffic").innerHTML = "<p>API process requests <b>" + esc(process.request_count == null ? "not available" : process.request_count) + "</b></p><p>Returned log-window rows <b>" + esc(log.record_count || 0) + "</b></p><p>These scopes differ and must not be compared as the same counter.</p>";
  const drift = p.reports.drift;
  const gate = p.monitoring_config || {};
  const thresholdLine = Object.keys(gate).length
    ? "<p>Configured PSI threshold " + fmt(gate.psi_threshold, 3) + " · KS alpha " + fmt(gate.ks_alpha, 3) +
      " · rule " + esc(gate.drift_rule || "not configured") + (gate.thresholds_provisional ? " · provisional" : " · as configured") + "</p>"
    : "<p>Drift thresholds not returned by configuration.</p>";
  const reportKind = drift && drift.data_kind === "simulation" ? "staged replay" :
    drift && drift.data_kind === "live_or_unclassified" ? "live or unclassified" : "unclassified";
  $("drift-summary").innerHTML = drift ? "<p><b>" + (drift.drift_flag ? "Drift flagged" : "No drift flag") + "</b></p><p>Report type " + esc(reportKind) + " · replay share " + pct(drift.replay_share) + "</p><p>Window " + esc(drift.window_size || "not returned") + " · report thresholds " + (drift.thresholds_provisional ? "provisional" : "as recorded") + "</p>" + thresholdLine + (drift.data_kind === "simulation" ? "<p>Replay is staged traffic, not field validation.</p>" : "") : "Saved drift report not generated yet." + thresholdLine;
  $("drift-table").innerHTML = drift && drift.features ? table(["Feature","PSI","KS p-value","Count","Flag"],Object.entries(drift.features).map(([n,v]) => [n,fmt(v.psi,4),Number.isFinite(Number(v.ks_p_value)) ? Number(v.ks_p_value).toExponential(3) : "not available",v.sample_count || "not returned",v.drift_flag ? "flagged" : "not flagged"])) : "Per-feature drift values not generated yet.";
  fetch(API + "/metrics").then((r) => r.ok ? r.text() : Promise.reject()).then((text) => {
    const rows = text.split(/\\r?\\n/).filter((x) => x && !x.startsWith("#"));
    $("prometheus-summary").textContent = rows.length ? rows.length + " API metric sample rows returned. Prometheus/Grafana runtime health is not established by this API." : "API returned no metric samples yet.";
  }).catch(() => empty("prometheus-summary","API metrics endpoint is not connected."));
}
function renderPipeline(p) {
  const pipe = p.pipeline || {};
  $("pipeline-view").innerHTML = pipe.stages && pipe.stages.length ? pipe.stages.map((s) => "<span title='" + esc((s.outputs || []).map((o) => o.path + ": " + (o.present ? "present" : "missing")).join("; ")) + "'>" + esc(s.name) + " · " + esc(s.status) + "</span>").join("") : "DVC stages " + esc(pipe.status || "not configured") + ".";
  const components = p.components || [];
  $("component-view").innerHTML = components.length ? components.map((c) => "<div class='component'><b>" + esc(c.name) + " · " + esc(c.status) + "</b>" + esc(c.detail) + "</div>").join("") : "Component status not returned.";
}
function renderResponsible(p) {
  const audit = p.reports.advisory_audit;
  $("advisory-audit").innerHTML = audit ? "<pre>" + esc(JSON.stringify(audit.counts || {},null,2)) + "</pre><p>" + esc(audit.note || "") + "</p>" : "Advisory audit snapshot not generated yet.";
  const bg = p.reports.background_sensitivity;
  $("trust-summary").innerHTML = bg ? "<pre>" + esc(JSON.stringify(bg,null,2)) + "</pre>" : "Background-sensitivity report not generated yet.";
  const registry = p.reports.registry;
  const versions = registry && Array.isArray(registry.versions) ? registry.versions : [];
  $("registry-view").innerHTML = versions.length
    ? "<p>Sanitized read-only snapshot · alias " + esc(registry.alias && registry.alias.version || "not recorded") + "</p>" + table(["Version","Gate verdict","Artifact source"],versions.map((v) => [v.version,v.gate_verdict,v.artifact_source_kind]))
    : "Sanitized registry snapshot not generated yet.";
  const limits = p.model_card_limitations || [];
  $("limitations").innerHTML = limits.length ? limits.map((x) => "<li>" + esc(x) + "</li>").join("") : "Model-card limitations unavailable.";
}
function render(p) {
  renderOverview(p); renderData(p.data || {}); renderModel(p.reports || {}); renderRelease(p);
  renderMonitoring(p); renderPipeline(p); renderResponsible(p);
  $("explain-eval").textContent = p.reports.explainability_evaluation || "Offline explanation evaluation not generated yet.";
  $("insights-status").textContent = "Connected. Live process counters and returned log-window values are separated; saved reports are labelled by source.";
}
async function refresh() {
  $("insights-status").textContent = "Refreshing local API and saved reports…";
  try { render(await get("/insights")); }
  catch (e) { $("insights-status").textContent = "Insights could not load: " + e.message + ". Check API health; panels remain explicit about unavailable evidence."; }
}
$("refresh-insights").addEventListener("click",refresh);
function imageData(base64) { return "data:image/png;base64," + base64; }
$("insights-explain-form").addEventListener("submit",async (event) => {
  event.preventDefault();
  const file = $("insights-image").files && $("insights-image").files[0];
  if (!file) return;
  $("insights-explain-output").textContent = "Submitting image to the local explanation worker…";
  const form = new FormData(); form.append("file",file,file.name); form.append("method",$("insights-method").value);
  try {
    const response = await fetch(API + "/explain/jobs",{method:"POST",body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not start explanation.");
    activeJob = data.job_id; $("insights-cancel").hidden = false; await pollJob(data.job_id,file);
  } catch (e) { $("insights-explain-output").innerHTML = "<div class='notice error'>" + esc(e.message) + "</div>"; }
});
async function pollJob(id,file) {
  while (activeJob === id) {
    await new Promise((resolve) => setTimeout(resolve,900));
    if (activeJob !== id) return;
    let job;
    try { job = await get("/explain/jobs/" + encodeURIComponent(id)); }
    catch (e) { $("insights-explain-output").textContent = e.message; activeJob=null; $("insights-cancel").hidden=true; return; }
    $("insights-explain-output").textContent = "Explanation status: " + job.status;
    if (job.status === "done") {
      const url = URL.createObjectURL(file);
      $("insights-explain-output").innerHTML = "<div class='comparison-grid'><figure><figcaption>Uploaded image</figcaption><img class='evidence-image' src='" + url + "' alt='Uploaded image'></figure><figure><figcaption>" + esc(job.method) + " · " + esc(job.predicted_class) + " · confidence " + pct(job.confidence) + "</figcaption><img class='evidence-image' src='" + imageData(job.overlay_png_base64) + "' alt='Attribution overlay for current uploaded image'></figure></div><p>" + esc(job.caption) + "</p><p>Runtime " + esc(job.runtime_ms) + " ms · model " + esc(job.model_version) + "</p>";
      activeJob=null; $("insights-cancel").hidden=true; return;
    }
    if (job.status === "failed" || job.status === "timeout") {
      $("insights-explain-output").innerHTML = "<div class='notice error'>" + esc(job.message || "Explanation failed.") + " (" + esc(job.reason_code || job.status) + ")</div>";
      activeJob=null; $("insights-cancel").hidden=true; return;
    }
  }
}
$("insights-cancel").addEventListener("click",async () => {
  const id=activeJob; activeJob=null; $("insights-cancel").hidden=true;
  if (id) { try { await fetch(API + "/explain/jobs/" + encodeURIComponent(id),{method:"DELETE"}); } catch { /* cancellation is best effort */ } }
  $("insights-explain-output").textContent="Explanation cancelled.";
});
async function initialize() {
  await refresh();
}
initialize();
