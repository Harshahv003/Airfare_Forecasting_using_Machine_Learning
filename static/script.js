/**
 * script.js — FlightFare AI Frontend
 * Handles: form validation, prediction API, Chart.js trends,
 *          loading states, price animation, modal, copy-to-clipboard
 */

/* ── DOM refs ────────────────────────────────────────────────────────────── */
const form          = document.getElementById("predictForm");
const submitBtn     = document.getElementById("submitBtn");
const loadingWrap   = document.getElementById("loadingWrap");
const resultWrap    = document.getElementById("resultWrap");
const errorWrap     = document.getElementById("errorWrap");
const loadingLabel  = document.getElementById("loadingLabel");

const resultPrice   = document.getElementById("resultPrice");
const rangeLow      = document.getElementById("rangeLow");
const rangeHigh     = document.getElementById("rangeHigh");
const metaRoute     = document.getElementById("metaRoute");
const metaDate      = document.getElementById("metaDate");
const metaDuration  = document.getElementById("metaDuration");
const errorMsg      = document.getElementById("errorMsg");
const copyBtn       = document.getElementById("copyBtn");
const resetBtn      = document.getElementById("resetBtn");
const errorResetBtn = document.getElementById("errorResetBtn");
const swapBtn       = document.getElementById("swapBtn");
const durationChip  = document.getElementById("durationChip");
const durationText  = document.getElementById("durationText");

// Inputs
const depTime   = document.getElementById("dep_time");
const arrTime   = document.getElementById("arrival_time");
const sourceEl  = document.getElementById("source");
const destEl    = document.getElementById("destination");
const journeyDateEl = document.getElementById("journey_date");

// Chart
const tabBtns     = document.querySelectorAll(".tab-btn");
const filterAirl  = document.getElementById("filterAirline");
const filterSrc   = document.getElementById("filterSource");
const filterDst   = document.getElementById("filterDest");

/* ── Loading messages ───────────────────────────────────────────────────── */
const LOADING_MSGS = [
  "Analysing route...", "Processing features...",
  "Running ML model...", "Calculating price estimate...", "Almost done..."
];
let loadingTimer = null;

function startLoading() {
  let i = 0;
  loadingLabel.textContent = LOADING_MSGS[0];
  loadingTimer = setInterval(() => {
    i = (i + 1) % LOADING_MSGS.length;
    loadingLabel.textContent = LOADING_MSGS[i];
  }, 1800);
}
function stopLoading() {
  clearInterval(loadingTimer);
  loadingTimer = null;
}

/* ── Duration auto-compute ──────────────────────────────────────────────── */
function computeDuration() {
  const dv = depTime.value, av = arrTime.value;
  if (!dv || !av) { durationChip.style.display = "none"; return; }
  const [dh, dm] = dv.split(":").map(Number);
  const [ah, am] = av.split(":").map(Number);
  let totalMin = (ah * 60 + am) - (dh * 60 + dm);
  if (totalMin < 0) totalMin += 24 * 60;
  const hrs = Math.floor(totalMin / 60), mins = totalMin % 60;
  durationText.textContent = `${hrs}h ${mins}m`;
  durationChip.style.display = "";
  return totalMin;
}

depTime.addEventListener("change", computeDuration);
arrTime.addEventListener("change", computeDuration);

/* ── Swap cities ────────────────────────────────────────────────────────── */
swapBtn.addEventListener("click", () => {
  const sv = sourceEl.value, dv = destEl.value;
  sourceEl.value = dv;
  destEl.value   = sv;
});

/* ── Set default journey date (tomorrow) ────────────────────────────────── */
(function setDefaultDate() {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  journeyDateEl.value = tomorrow.toISOString().split("T")[0];
  journeyDateEl.min   = new Date().toISOString().split("T")[0];
})();

/* ── Form validation ────────────────────────────────────────────────────── */
function clearErrors() {
  document.querySelectorAll(".field-error").forEach(el => el.textContent = "");
}
function setError(id, msg) {
  const el = document.getElementById("err-" + id);
  if (el) el.textContent = msg;
}
function validateForm(data) {
  let valid = true;
  clearErrors();
  if (!data.airline)     { setError("airline", "Please select an airline.");     valid = false; }
  if (!data.source)      { setError("source",  "Please select source city.");    valid = false; }
  if (!data.destination) { setError("destination","Please select destination."); valid = false; }
  if (data.source && data.destination && data.source === data.destination) {
    setError("destination", "Source and destination must be different."); valid = false;
  }
  if (!data.journey_date) { setError("date", "Please pick a journey date."); valid = false; }
  if (!data.dep_time)     { setError("dep",  "Please enter departure time.");    valid = false; }
  if (!data.arrival_time) { setError("arr",  "Please enter arrival time.");      valid = false; }
  return valid;
}

/* ── Price counter animation ────────────────────────────────────────────── */
function animatePrice(target, duration = 900) {
  const start = performance.now();
  const prefix = "₹";
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = Math.round(eased * target);
    resultPrice.textContent = prefix + current.toLocaleString("en-IN");
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Show result ─────────────────────────────────────────────────────────── */
function showResult(data, formData) {
  resultWrap.classList.remove("hidden");
  errorWrap.classList.add("hidden");

  animatePrice(data.price);
  rangeLow.textContent  = "₹" + data.price_low.toLocaleString("en-IN");
  rangeHigh.textContent = "₹" + data.price_high.toLocaleString("en-IN");

  metaRoute.textContent    = `${formData.source} → ${formData.destination}`;
  metaDate.textContent     = formData.journey_date;
  const durMins = computeDuration() || "–";
  metaDuration.textContent = durMins !== "–"
    ? `${Math.floor(durMins/60)}h ${durMins%60}m flight`
    : "–";

  resultWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ── Show error ──────────────────────────────────────────────────────────── */
function showError(msg) {
  errorWrap.classList.remove("hidden");
  resultWrap.classList.add("hidden");
  errorMsg.textContent = msg;
  errorWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ── Form submit ─────────────────────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = {
    airline:          document.getElementById("airline").value,
    source:           sourceEl.value,
    destination:      destEl.value,
    journey_date:     journeyDateEl.value,
    dep_time:         depTime.value,
    arrival_time:     arrTime.value,
    stops:            document.getElementById("stops").value,
    duration_minutes: computeDuration() || 0,
  };

  if (!validateForm(formData)) return;

  // UI: loading state
  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-text").textContent = "Predicting…";
  resultWrap.classList.add("hidden");
  errorWrap.classList.add("hidden");
  loadingWrap.classList.remove("hidden");
  startLoading();

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(formData),
    });
    const json = await resp.json();

    if (resp.ok && json.success) {
      showResult(json, formData);
    } else {
      showError(json.error || "Prediction failed. Please try again.");
    }
  } catch (err) {
    showError("Network error. Please check your connection and try again.");
  } finally {
    stopLoading();
    loadingWrap.classList.add("hidden");
    submitBtn.disabled = false;
    submitBtn.querySelector(".btn-text").textContent = "Predict Price";
  }
});

/* ── Copy price ──────────────────────────────────────────────────────────── */
copyBtn.addEventListener("click", async () => {
  const text = resultPrice.textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = "✓ Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => { copyBtn.textContent = "⎘ Copy"; copyBtn.classList.remove("copied"); }, 2200);
  } catch {}
});

/* ── Reset ───────────────────────────────────────────────────────────────── */
function doReset() {
  resultWrap.classList.add("hidden");
  errorWrap.classList.add("hidden");
}
resetBtn.addEventListener("click", doReset);
errorResetBtn.addEventListener("click", doReset);

/* ════════════════════════════════════════════════════════════════
   CHART.JS TREND CHART
   ════════════════════════════════════════════════════════════════ */
let trendChart = null;
let currentTab = "monthly";

const CHART_COLORS = {
  accent:  "rgba(59,130,246,0.85)",
  purple:  "rgba(139,92,246,0.7)",
  pink:    "rgba(236,72,153,0.7)",
  fill:    "rgba(59,130,246,0.1)",
};

function buildChartData(data, type) {
  if (type === "monthly") {
    return {
      labels:   data.labels,
      datasets: [{
        label:           "Avg Price (₹)",
        data:            data.avg,
        borderColor:     CHART_COLORS.accent,
        backgroundColor: CHART_COLORS.fill,
        fill:            true,
        tension:         0.4,
        pointBackgroundColor: CHART_COLORS.accent,
        pointRadius:     4,
      }]
    };
  }
  // airline or route — bar
  return {
    labels:   data.labels,
    datasets: [{
      label:           "Avg Price (₹)",
      data:            data.avg,
      backgroundColor: data.labels.map((_, i) =>
        [CHART_COLORS.accent, CHART_COLORS.purple, CHART_COLORS.pink][i % 3]
      ),
      borderRadius:    6,
    }]
  };
}

async function loadChart(type = "monthly") {
  const params = new URLSearchParams({ type });
  const src = filterSrc.value, dst = filterDst.value;
  if (src) params.set("source", src);
  if (dst) params.set("destination", dst);

  try {
    const resp = await fetch(`/api/trends?${params}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.error) return;

    const ctx  = document.getElementById("trendChart").getContext("2d");
    const isLine = (type === "monthly");
    const chartData = buildChartData(data, type);

    const options = {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0d1220",
          borderColor: "rgba(255,255,255,0.08)",
          borderWidth: 1,
          titleColor: "#94a3b8",
          bodyColor:  "#f1f5f9",
          callbacks: {
            label: ctx => "₹" + Number(ctx.raw).toLocaleString("en-IN"),
          }
        }
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
        y: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#94a3b8", font: { size: 10 },
            callback: v => "₹" + Number(v).toLocaleString("en-IN"),
          }
        }
      }
    };

    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, {
      type: isLine ? "line" : "bar",
      data: chartData,
      options,
    });
  } catch (err) {
    console.warn("Chart load error:", err);
  }
}

// Tab switching
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentTab = btn.dataset.type;
    loadChart(currentTab);
  });
});

// Filter changes
filterSrc.addEventListener("change", () => loadChart(currentTab));
filterDst.addEventListener("change", () => loadChart(currentTab));

// Initial load
loadChart("monthly");

/* ════════════════════════════════════════════════════════════════
   IMAGE MODAL
   ════════════════════════════════════════════════════════════════ */
function openModal(src, title) {
  const overlay = document.getElementById("modalOverlay");
  document.getElementById("modalImg").src   = src;
  document.getElementById("modalTitle").textContent = title;
  overlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}
function closeModal() {
  document.getElementById("modalOverlay").classList.add("hidden");
  document.body.style.overflow = "";
}
// ESC key closes modal
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeModal();
});
