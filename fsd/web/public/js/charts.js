(() => {
  if (typeof Chart === "undefined") {
    console.error(
      "[MaintIQ] Chart.js did not load (/js/chart.umd.min.js). Dashboard and analytics charts were skipped."
    );
    return;
  }

  Chart.defaults.color = "#9a9690";
  Chart.defaults.borderColor = "rgba(255,255,255,0.08)";
  Chart.defaults.font.family = "Manrope, system-ui, sans-serif";

  const lineOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { boxWidth: 10 } } },
    scales: { x: { grid: { display: false } }, y: { beginAtZero: true } },
  };

  const byId = (id) => document.getElementById(id);
  const dash = window.DASHBOARD_DATA || {};
  const analytics = window.ANALYTICS_DATA || null;

  if (byId("healthDonut") && !dash.healthCounts) {
    console.warn("[MaintIQ] DASHBOARD_DATA.healthCounts missing; donut may be empty.");
  }
  if (byId("analytics-filters") && !analytics) {
    console.error(
      "[MaintIQ] ANALYTICS_DATA missing (/js/analytics-data.js). Analytics charts and window table were skipped."
    );
  }

  const sampleSeries = (series, maxPoints = 48) => {
    if (!series?.length) return [];
    const step = Math.max(1, Math.floor(series.length / maxPoints));
    return series.filter((_, i) => i % step === 0 || i === series.length - 1);
  };

  const fmtDay = (iso) => {
    if (!iso) return "";
    return new Date(iso).toLocaleDateString(undefined, { day: "2-digit", month: "short" });
  };

  const donut = byId("healthDonut");
  if (donut) {
    const counts = dash.healthCounts || {};
    const slices = [
      ["Healthy", counts.Healthy || 0, "#4ade80"],
      ["Warning", counts.Warning || 0, "#fbbf24"],
      ["Critical", counts.Critical || 0, "#f87171"],
    ];
    if (counts.Unknown) slices.push(["Unknown", counts.Unknown, "#94a3b8"]);
    new Chart(donut, {
      type: "doughnut",
      data: {
        labels: slices.map((slice) => slice[0]),
        datasets: [{
          data: slices.map((slice) => slice[1]),
          backgroundColor: slices.map((slice) => slice[2]),
          borderWidth: 0,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, cutout: "68%", plugins: { legend: { position: "bottom" } } },
    });
  }

  const healthTrend = byId("healthTrend");
  if (healthTrend) {
    const drawTrend = (machineId) => {
      const trend = dash.trends?.[machineId] || { labels: [], anomaly: [], rms: [] };
      const existing = Chart.getChart(healthTrend);
      if (existing) {
        existing.data.labels = trend.labels;
        existing.data.datasets[0].data = trend.anomaly;
        existing.data.datasets[1].data = trend.rms;
        existing.update();
        return;
      }
      new Chart(healthTrend, {
        type: "line",
        data: {
          labels: trend.labels,
          datasets: [
            { label: "Anomaly score", data: trend.anomaly, borderColor: "#7ee7ff", tension: 0.35, fill: false },
            { label: "RMS (g)", data: trend.rms, borderColor: "#c4b5fd", tension: 0.35, fill: false },
          ],
        },
        options: lineOpts,
      });
    };
    const trendSelect = byId("health-trend-machine");
    drawTrend(trendSelect?.value || dash.defaultTrendId);
    trendSelect?.addEventListener("change", () => drawTrend(trendSelect.value));
  }

  const alertSpark = byId("alertSpark");
  if (alertSpark && !byId("analytics-filters")) {
    const baseline = analytics?.ifBaselineMean ?? dash.ifBaseline ?? 0.431;
    const sparkOpts = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9a9690", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { min: 0, ticks: { color: "#9a9690", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
      },
    };

    const drawAlertSpark = (machineId) => {
      const machine =
        analytics?.machines?.[machineId] ||
        analytics?.machines?.[dash.defaultTrendId] ||
        analytics?.machines?.["brg-003"];
      const sampled = sampleSeries(machine?.series || [], 36);
      const labels = sampled.map((p) => fmtDay(p.t));
      const anomaly = sampled.map((p) => p.anomaly);
      const ref = sampled.map(() => baseline);
      const existing = Chart.getChart(alertSpark);
      if (existing) {
        existing.data.labels = labels;
        existing.data.datasets[0].data = anomaly;
        existing.data.datasets[1].data = ref;
        existing.update();
        return;
      }
      new Chart(alertSpark, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Anomaly score",
              data: anomaly,
              borderColor: "#5eead4",
              backgroundColor: "rgba(94, 234, 212, 0.12)",
              fill: true,
              pointRadius: 0,
              tension: 0.35,
              borderWidth: 2,
            },
            {
              label: "IF early-life mean",
              data: ref,
              borderColor: "#f87171",
              borderDash: [6, 4],
              pointRadius: 0,
              tension: 0,
              borderWidth: 1.5,
            },
          ],
        },
        options: sparkOpts,
      });
    };

    const selected = document.querySelector("#alerts-table tbody tr.is-selected");
    drawAlertSpark(selected?.dataset.machineId || dash.defaultTrendId || "brg-003");
    window.updateAlertSparkChart = drawAlertSpark;
  }

  const data = window.ANALYTICS_DATA;
  if (!data || !byId("analytics-filters")) return;

  const charts = {};
  const palette = {
    rms: "#5eead4",
    peak: "#fb923c",
    kurtosis: "#c4b5fd",
    crest: "#67e8f9",
    bpfo: "#38bdf8",
    bpfi: "#f472b6",
    anomaly: "#f472b6",
    fail: "#fbbf24",
    ref: "#f87171",
    baseline: "#94a3b8",
  };

  const fmtStamp = (iso) => {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
    });
  };
  const toInputDate = (iso) => iso.slice(0, 10);
  const pct = (now, base) => {
    if (!base) return "—";
    const delta = ((now - base) / base) * 100;
    const sign = delta > 0 ? "+" : "";
    return `${sign}${delta.toFixed(0)}% vs early-life mean`;
  };

  const machineEl = byId("analytics-machine");
  const fromEl = byId("analytics-from");
  const toEl = byId("analytics-to");
  const featEl = byId("analytics-features");
  const requestedId = new URLSearchParams(window.location.search).get("machine");
  if (requestedId && machineEl && data.machines[requestedId]) {
    machineEl.value = requestedId;
  }
  const applied = {
    machineId: machineEl?.value || "",
    from: "",
    to: "",
    features: featEl?.value || "rms,peak,kurtosis",
  };

  const publishContext = () => {
    window.ANALYTICS_CONTEXT = {
      machineId: applied.machineId,
      from: applied.from,
      to: applied.to,
      features: applied.features,
    };
  };

  const defaultRange = (machine) => {
    const end = new Date(machine.end);
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - 30);
    const runStart = new Date(machine.start);
    if (Number.isNaN(end.getTime()) || Number.isNaN(runStart.getTime())) {
      return { from: machine.start || "", to: machine.end || "" };
    }
    if (start < runStart) return { from: machine.start, to: machine.end };
    return { from: start.toISOString(), to: machine.end };
  };

  const todayInput = () => new Date().toISOString().slice(0, 10);

  const applyDateBounds = (machineId) => {
    const machine = data.machines[machineId];
    const minDate = toInputDate(machine?.start || "2000-01-01T00:00:00.000Z") || "2000-01-01";
    const today = todayInput();
    if (fromEl) {
      fromEl.min = minDate;
      fromEl.max = today;
    }
    if (toEl) {
      toEl.min = minDate;
      toEl.max = today;
    }
  };

  const clampDateValue = (el) => {
    if (!el) return;
    if (el.validity?.badInput) {
      el.value = el.min || "";
      return;
    }
    if (!el.value) return;
    if (el.min && el.value < el.min) el.value = el.min;
    if (el.max && el.value > el.max) el.value = el.max;
  };

  const writeDatesToForm = (machineId) => {
    const machine = data.machines[machineId];
    if (!machine || !fromEl || !toEl) return;
    applyDateBounds(machineId);
    const range = defaultRange(machine);
    fromEl.value = toInputDate(range.from);
    toEl.value = toInputDate(range.to);
    clampDateValue(fromEl);
    clampDateValue(toEl);
  };

  const commitForm = () => {
    applied.machineId = machineEl?.value || "";
    applied.from = fromEl?.value || "";
    applied.to = toEl?.value || "";
    applied.features = featEl?.value || "rms,peak,kurtosis";
  };

  const syncFormFromApplied = () => {
    if (machineEl && applied.machineId) machineEl.value = applied.machineId;
    if (fromEl) fromEl.value = applied.from;
    if (toEl) toEl.value = applied.to;
    if (featEl) featEl.value = applied.features;
  };

  const collapseFilters = () => {
    const panel = byId("analytics-filters-panel");
    const toggle = byId("analytics-filters-toggle");
    if (panel) panel.hidden = true;
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
      toggle.classList.remove("btn-primary");
    }
  };

  const filtered = () => {
    const machine = data.machines[applied.machineId];
    if (!machine) return { machine: null, series: [], from: null, to: null };
    const series = machine.series.filter((point) => {
      const day = String(point.t).slice(0, 10);
      if (applied.from && day < applied.from) return false;
      if (applied.to && day > applied.to) return false;
      return true;
    });
    return { machine, series };
  };

  const destroy = (id) => {
    const canvas = byId(id);
    if (canvas) Chart.getChart(canvas)?.destroy();
    charts[id] = null;
  };

  const tooltip = (unit) => ({
    backgroundColor: "#12141a",
    borderColor: "rgba(255,255,255,0.12)",
    borderWidth: 1,
    titleColor: "#f3efe8",
    bodyColor: "#d4d0c8",
    callbacks: {
      label: (ctx) => {
        const v = ctx.parsed.y;
        if (v == null) return ctx.dataset.label;
        return `${ctx.dataset.label}: ${v}${unit ? ` ${unit}` : ""}`;
      },
    },
  });

  const lineBase = (yTitle, unit, yOpts = {}) => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { boxWidth: 10, color: "#d4d0c8" } },
      tooltip: tooltip(unit),
    },
    scales: {
      x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      y: {
        beginAtZero: true,
        title: { display: true, text: yTitle, color: "#9a9690" },
        grid: { color: "rgba(255,255,255,0.06)" },
        ...yOpts,
      },
    },
  });

  const featureMap = {
    rms: { key: "rms", label: "RMS (g)", color: palette.rms },
    peak: { key: "peak", label: "Peak (g)", color: palette.peak },
    kurtosis: { key: "kurtosis", label: "Kurtosis", color: palette.kurtosis },
    crest: { key: "crest", label: "Crest factor", color: palette.crest },
    bpfo: { key: "bpfo", label: "BPFO magnitude", color: palette.bpfo },
    bpfi: { key: "bpfi", label: "BPFI magnitude", color: palette.bpfi },
  };

  const setText = (id, html) => {
    const el = byId(id);
    if (el) el.innerHTML = html;
  };

  const renderVibration = (ctx) => {
    const { machine, series } = ctx;
    const labels = series.map((p) => fmtDay(p.t));
    const keys = applied.features.split(",");
    const unit = keys.includes("rms") || keys.includes("peak") ? "g" : "";
    destroy("rmsChart");
    const canvas = byId("rmsChart");
    if (canvas) {
      charts.rmsChart = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: keys.map((k) => ({
            label: featureMap[k].label,
            data: series.map((p) => p[featureMap[k].key]),
            borderColor: featureMap[k].color,
            backgroundColor: `${featureMap[k].color}22`,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 2,
            fill: false,
          })),
        },
        options: lineBase(keys.includes("rms") ? "Vibration (g)" : "Feature value", unit),
      });
    }

    const last = series[series.length - 1];
    const early = machine.earlyMean || {};
    if (last) {
      byId("m-rms").textContent = `${last.rms} g`;
      byId("m-rms-sub").textContent = pct(last.rms, early.rms);
      byId("m-peak").textContent = `${last.peak} g`;
      byId("m-peak-sub").textContent = pct(last.peak, early.peak);
      byId("m-kurt").textContent = String(last.kurtosis);
      byId("m-n").textContent = String(series.length);
    }

    const rmsUp = last && early.rms != null && last.rms > early.rms;
    const parts = [
      `<p><strong>${machine.name}</strong> · ${machine.assetId}${machine.documentedDamage ? ` · ${machine.documentedDamage}` : " · no documented IMS damage"}</p>`,
      last
        ? `<p>Latest window ${fmtStamp(last.t)}: RMS <strong>${last.rms} g</strong> (early-life mean ${early.rms ?? "—"} g), peak <strong>${last.peak} g</strong> (early-life ${early.peak ?? "—"} g), kurtosis <strong>${last.kurtosis}</strong> (early-life ${early.kurtosis ?? "—"}).</p>`
        : "<p>No scored windows in this date range.</p>",
      rmsUp
        ? "<p>RMS in the latest window is above this run’s early-life mean. That is a reading change, not a confirmed fault.</p>"
        : "<p>Latest RMS is not above this run’s early-life mean.</p>",
    ];
    if (last?.flag) {
      parts.push(`<p>Isolation Forest flagged this window. <a href="/alerts">Open related alerts</a></p>`);
    }
    setText("vibration-context", parts.join(""));

    const body = byId("analytics-windows")?.querySelector("tbody");
    if (body) {
      const rows = [...series].slice().reverse();
      body.innerHTML = rows.map((p) => {
        const flags = [
          p.flag ? "IF flag" : null,
          p.failPred ? "24h risk" : null,
        ].filter(Boolean).join(" · ") || "—";
        const rul = p.rul == null ? "—" : `${p.rul} h`;
        const fp = p.failProb == null ? "—" : String(p.failProb);
        return `<tr data-match="1">
          <td>${fmtStamp(p.t)}</td>
          <td>${p.rms}</td>
          <td>${p.peak}</td>
          <td>${p.anomaly}</td>
          <td>${fp}</td>
          <td>${rul}</td>
          <td>${flags}</td>
        </tr>`;
      }).join("");
      window.refreshTablePager?.();
    }
  };

  const renderAnomalies = (ctx) => {
    const { machine, series } = ctx;
    destroy("anomalyChart");
    const canvas = byId("anomalyChart");
    if (canvas) {
      charts.anomalyChart = new Chart(canvas, {
        type: "line",
        data: {
          labels: series.map((p) => fmtDay(p.t)),
          datasets: [
            {
              label: "Isolation Forest score",
              data: series.map((p) => p.anomaly),
              borderColor: palette.anomaly,
              backgroundColor: "rgba(244,114,182,0.14)",
              fill: true,
              tension: 0.3,
              borderWidth: 2,
              pointRadius: 2,
            },
            {
              label: "Train early-life mean (0.431)",
              data: series.map(() => data.ifBaselineMean),
              borderColor: palette.baseline,
              borderDash: [6, 4],
              pointRadius: 0,
              borderWidth: 1.5,
            },
          ],
        },
        options: lineBase("Anomaly score (−score_samples)", ""),
      });
    }
    const flagged = series.filter((p) => p.flag);
    const last = series[series.length - 1];
    setText("anomaly-context", [
      `<p>${flagged.length} of ${series.length} 12-hour bins in this range have <code>anomaly_flag = true</code> from the 1% detector.</p>`,
      last ? `<p>Latest score <strong>${last.anomaly}</strong> vs train early-life mean <strong>${data.ifBaselineMean}</strong> and this run’s early-life mean <strong>${machine.earlyMean?.anomalyScore ?? "—"}</strong>.</p>` : "",
      flagged.length
        ? `<p><a href="/alerts">View investigation alerts</a> — flags mean the window looks unlike the IF baseline, not that a bearing has failed.</p>`
        : "<p>No Isolation Forest flags in this date range.</p>",
    ].join(""));
  };

  const renderFailure = (ctx) => {
    const { series } = ctx;
    const threshold = data.classifierThreshold;
    destroy("failureChart");
    const canvas = byId("failureChart");
    if (canvas) {
      charts.failureChart = new Chart(canvas, {
        type: "line",
        data: {
          labels: series.map((p) => fmtDay(p.t)),
          datasets: [
            {
              label: "Failure probability (24 h)",
              data: series.map((p) => p.failProb),
              borderColor: palette.fail,
              backgroundColor: "rgba(251,191,36,0.12)",
              fill: true,
              tension: 0.3,
              borderWidth: 2,
              pointRadius: 2,
              spanGaps: true,
            },
            {
              label: "Threshold 0.315",
              data: series.map(() => threshold),
              borderColor: palette.ref,
              borderDash: [6, 4],
              pointRadius: 0,
              borderWidth: 1.5,
            },
          ],
        },
        options: lineBase("Calibrated probability", "", { min: 0, max: 1 }),
      });
    }
    const crossed = series.filter((p) => p.failProb != null && p.failProb >= threshold);
    const predicted = series.filter((p) => p.failPred === 1);
    const last = series[series.length - 1];
    const missing = series.every((p) => p.failProb == null);
    setText("failure-context", missing
      ? "<p>No classifier scores are stored for this bearing, so no threshold crossing can be shown.</p>"
      : [
        last ? `<p>Latest probability <strong>${last.failProb ?? "—"}</strong> vs threshold <strong>0.315</strong>${last.failPred ? " — predicted class is risk." : " — predicted class is healthy."}</p>` : "",
        `<p>${crossed.length} bins are at or above 0.315. ${predicted.length} bins have <code>predicted_failure_within_24h = 1</code>.</p>`,
        predicted.length
          ? `<p><a href="/alerts">Related alerts</a> should be treated as a 24-hour investigation cue.</p>`
          : "<p>The 24-hour classifier did not fire in this date range.</p>",
      ].join(""));
  };

  const renderFrequency = (ctx) => {
    const { machine } = ctx;
    const early = machine.earlyMean || {};
    const latestBands = Array.isArray(machine.latestBands) ? machine.latestBands : [];
    const earlyBands = Array.isArray(early.bandEnergy) ? early.bandEnergy : [];
    const fault = machine.latestFaultFreq || {};
    const bands = ["0–500 Hz", "500–1k Hz", "1–2 kHz", "2–4 kHz"];
    destroy("freqChart");
    const freqCanvas = byId("freqChart");
    if (freqCanvas) {
      charts.freqChart = new Chart(freqCanvas, {
        type: "bar",
        data: {
          labels: bands,
          datasets: [
            { label: "Latest window", data: latestBands, backgroundColor: "#38bdf8cc" },
            { label: "Early-life mean", data: earlyBands, backgroundColor: "#94a3b8aa" },
          ],
        },
        options: lineBase("Band energy", ""),
      });
    }
    destroy("faultFreqChart");
    const ff = byId("faultFreqChart");
    if (ff) {
      charts.faultFreqChart = new Chart(ff, {
        type: "bar",
        data: {
          labels: ["FTF", "BSF", "BPFO", "BPFI"],
          datasets: [{
            label: "Latest magnitude",
            data: [fault.ftf, fault.bsf, fault.bpfo, fault.bpfi],
            backgroundColor: ["#67e8f9", "#c4b5fd", "#38bdf8", "#f472b6"],
          }],
        },
        options: lineBase("Spectral magnitude", ""),
      });
    }
    const i12 = latestBands[2] ?? "—";
    const e12 = earlyBands[2] ?? "—";
    setText("frequency-context", `
      <p>1–2 kHz band energy (inside the 1–3 kHz envelope design band): latest <strong>${i12}</strong> vs early-life <strong>${e12}</strong>.</p>
      <p>Latest BPFO magnitude <strong>${fault.bpfo ?? "—"}</strong> vs early-life <strong>${early.bpfo ?? "—"}</strong>; BPFI <strong>${fault.bpfi ?? "—"}</strong> vs <strong>${early.bpfi ?? "—"}</strong>.</p>
      <p>These are feature values from the same snapshots the models scored. They are not a root-cause label.</p>
    `);
  };

  const renderPerformance = () => {
    const c = data.metrics.classifier;
    destroy("rocChart");
    const roc = byId("rocChart");
    if (roc) {
      charts.rocChart = new Chart(roc, {
        type: "bar",
        data: {
          labels: ["Precision", "Recall", "F1"],
          datasets: [
            { label: "Validation", data: [c.val.precision, c.val.recall, c.val.f1], backgroundColor: "#5eead4cc" },
            { label: "Test (Test 3)", data: [c.test.precision, c.test.recall, c.test.f1], backgroundColor: "#c4b5fdcc" },
          ],
        },
        options: lineBase("Score", "", { min: 0, max: 1 }),
      });
    }
    destroy("rulChart");
    const rul = byId("rulChart");
    if (rul) {
      const r = data.metrics.rul;
      charts.rulChart = new Chart(rul, {
        type: "bar",
        data: {
          labels: ["Last 24 h MAE", "Last 72 h MAE"],
          datasets: [
            { label: "Validation (T1-B4)", data: [r.valMae24h, r.valMae72h], backgroundColor: "#67e8f9cc" },
            { label: "Test (T3-B3)", data: [r.testMae24h, r.testMae72h], backgroundColor: "#fb923ccc" },
          ],
        },
        options: lineBase("Hours", "h"),
      });
    }
  };

  const render = () => {
    const ctx = filtered();
    if (!ctx.machine) return;
    renderVibration(ctx);
    renderAnomalies(ctx);
    renderFailure(ctx);
    renderFrequency(ctx);
    renderPerformance();
    publishContext();
    window.setTimeout(() => {
      Object.values(charts).forEach((ch) => ch?.resize());
    }, 40);
  };

  writeDatesToForm(applied.machineId);
  commitForm();
  render();

  machineEl?.addEventListener("change", () => {
    writeDatesToForm(machineEl.value);
  });

  [fromEl, toEl].forEach((el) => {
    el?.addEventListener("change", () => clampDateValue(el));
    el?.addEventListener("blur", () => clampDateValue(el));
    el?.addEventListener("input", () => {
      if (el.validity?.badInput) return;
      clampDateValue(el);
    });
  });

  byId("analytics-filters")?.addEventListener("submit", (event) => {
    event.preventDefault();
    clampDateValue(fromEl);
    clampDateValue(toEl);
    if (fromEl?.value && toEl?.value && fromEl.value > toEl.value) toEl.value = fromEl.value;
    commitForm();
    render();
    collapseFilters();
  });

  byId("analytics-apply")?.addEventListener("click", () => {
    clampDateValue(fromEl);
    clampDateValue(toEl);
    if (fromEl?.value && toEl?.value && fromEl.value > toEl.value) toEl.value = fromEl.value;
    commitForm();
    render();
    collapseFilters();
  });

  byId("analytics-reset")?.addEventListener("click", () => {
    if (machineEl?.options?.length) machineEl.value = machineEl.options[0].value;
    if (featEl) featEl.value = "rms,peak,kurtosis";
    writeDatesToForm(machineEl.value);
    commitForm();
    render();
    collapseFilters();
  });

  byId("analytics-filters-toggle")?.addEventListener("click", () => {
    const panel = byId("analytics-filters-panel");
    const toggle = byId("analytics-filters-toggle");
    if (!panel || !toggle) return;
    const opening = panel.hidden;
    if (opening) syncFormFromApplied();
    panel.hidden = !opening;
    toggle.setAttribute("aria-expanded", opening ? "true" : "false");
    toggle.classList.toggle("btn-primary", opening);
  });

  document.querySelector("[data-export-csv]")?.addEventListener("click", () => {
    const { machine, series } = filtered();
    if (!machine) return;
    const header = "timestamp,rms_g,peak_g,kurtosis,anomaly_score,if_flag,failure_probability,predicted_failure_24h,predicted_rul_hours";
    const lines = series.map((p) =>
      [p.t, p.rms, p.peak, p.kurtosis, p.anomaly, p.flag ? 1 : 0, p.failProb ?? "", p.failPred, p.rul ?? ""].join(",")
    );
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${machine.assetId || "machine"}_${applied.from || "start"}_${applied.to || "end"}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  document.querySelector("[data-download-chart]")?.addEventListener("click", () => {
    const active = document.querySelector(".tab-panel.is-active canvas");
    if (!active) return;
    const a = document.createElement("a");
    a.href = active.toDataURL("image/png");
    a.download = `analytics-chart.png`;
    a.click();
  });
})();

(() => {
  const buttons = document.querySelectorAll("[data-explain-graph]");
  if (!buttons.length) return;

  const cache = new Map();
  const inflight = new Map();
  let requestSeq = 0;

  const currentContext = () => {
    const applied = window.ANALYTICS_CONTEXT || {};
    return {
      machineId: applied.machineId || document.getElementById("analytics-machine")?.value || "",
      from: applied.from || "",
      to: applied.to || "",
      features: applied.features || "rms,peak,kurtosis",
    };
  };

  const cacheKey = (graphId, ctx) =>
    [graphId, ctx.machineId, ctx.from, ctx.to, graphId === "vibration" ? ctx.features : ""].join("|");

  const panelFor = (graphId) => document.querySelector(`[data-explain-panel="${graphId}"]`);
  const bodyFor = (graphId) => document.querySelector(`[data-explain-body="${graphId}"]`);
  const buttonFor = (graphId) => document.querySelector(`[data-explain-graph="${graphId}"]`);

  const setOpen = (graphId, open) => {
    const panel = panelFor(graphId);
    const button = buttonFor(graphId);
    if (panel) panel.hidden = !open;
    if (button) {
      button.setAttribute("aria-expanded", open ? "true" : "false");
      button.textContent = open ? "Hide" : "Explain";
    }
  };

  const node = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  };

  const renderLoading = (graphId) => {
    const body = bodyFor(graphId);
    if (!body) return;
    body.replaceChildren(node("p", "explain-status", "Generating explanation…"));
  };

  const renderError = (graphId, message) => {
    const body = bodyFor(graphId);
    if (!body) return;
    const wrap = node("div", "explain-error");
    wrap.append(node("p", "", message || "Could not explain this graph just now."));
    const retry = node("button", "btn", "Retry");
    retry.type = "button";
    retry.addEventListener("click", () => requestExplanation(graphId, { force: true }));
    wrap.append(retry);
    body.replaceChildren(wrap);
  };

  const renderExplanation = (graphId, explanation) => {
    const body = bodyFor(graphId);
    if (!body) return;
    const fragment = document.createDocumentFragment();
    fragment.append(node("p", "explain-summary", explanation.summary || ""));

    if (explanation.observations?.length) {
      const section = node("section", "explain-section");
      section.append(node("h5", "", "Observations"));
      const list = node("ul");
      explanation.observations.forEach((item) => {
        const li = node("li");
        li.append(node("span", "", item.finding || ""));
        if (item.evidence) {
          li.append(document.createTextNode(" "));
          li.append(node("span", "explain-evidence", item.evidence));
        }
        list.append(li);
      });
      section.append(list);
      fragment.append(section);
    }

    if (explanation.interpretation?.length) {
      const section = node("section", "explain-section");
      section.append(node("h5", "", "Interpretation"));
      const list = node("ul");
      explanation.interpretation.forEach((item) => list.append(node("li", "", item)));
      section.append(list);
      fragment.append(section);
    }

    if (explanation.limitations?.length) {
      const section = node("section", "explain-section");
      section.append(node("h5", "", "Limitations"));
      const list = node("ul");
      explanation.limitations.forEach((item) => list.append(node("li", "", item)));
      section.append(list);
      fragment.append(section);
    }

    if (explanation.suggested_actions?.length) {
      const section = node("section", "explain-section");
      section.append(node("h5", "", "Suggested actions"));
      const list = node("ul");
      explanation.suggested_actions.forEach((item) => {
        const li = node("li");
        li.append(node("span", "", item.action || ""));
        if (item.reason) {
          li.append(document.createTextNode(" — "));
          li.append(node("span", "explain-evidence", item.reason));
        }
        list.append(li);
      });
      section.append(list);
      fragment.append(section);
    }

    if (explanation.confidence) {
      fragment.append(node("p", "explain-confidence", `Confidence: ${explanation.confidence}`));
    }

    body.replaceChildren(fragment);
  };

  const requestExplanation = async (graphId, { force = false } = {}) => {
    const ctx = currentContext();
    const key = cacheKey(graphId, ctx);
    const existing = inflight.get(graphId);
    if (existing && existing.key === key && !force) {
      setOpen(graphId, true);
      return;
    }

    if (!force && cache.has(key)) {
      renderExplanation(graphId, cache.get(key));
      setOpen(graphId, true);
      return;
    }

    const token = ++requestSeq;
    inflight.set(graphId, { key, token });
    setOpen(graphId, true);
    renderLoading(graphId);

    try {
      const csrf = document.querySelector('meta[name="csrf-token"]')?.getAttribute("content") || "";
      const response = await fetch("/api/analytics/explain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          ...(csrf ? { "X-CSRF-Token": csrf } : {}),
        },
        body: JSON.stringify({ graphId, ...ctx }),
      });
      const payload = await response.json().catch(() => ({}));
      const current = inflight.get(graphId);
      if (!current || current.token !== token) return;
      if (!response.ok || !payload.ok || !payload.explanation) {
        inflight.delete(graphId);
        renderError(graphId, payload.error || "Could not explain this graph just now.");
        return;
      }
      cache.set(key, payload.explanation);
      inflight.delete(graphId);
      renderExplanation(graphId, payload.explanation);
    } catch (err) {
      const current = inflight.get(graphId);
      if (!current || current.token !== token) return;
      inflight.delete(graphId);
      renderError(graphId, "Could not reach the explanation service.");
    }
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const graphId = button.dataset.explainGraph;
      const panel = panelFor(graphId);
      if (!graphId || !panel) return;
      if (!panel.hidden) {
        setOpen(graphId, false);
        return;
      }
      requestExplanation(graphId);
    });
  });

  document.querySelectorAll("[data-explain-collapse]").forEach((button) => {
    button.addEventListener("click", () => setOpen(button.dataset.explainCollapse, false));
  });
})();
