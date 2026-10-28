(() => {
  const ready = async () => {
    try {
      if (document.fonts?.ready) await document.fonts.ready;
    } catch (_err) {
      /* continue without webfonts */
    }
    window.REPORT_CHARTS_READY = true;
  };

  const data = window.REPORT_CHARTS;
  if (!data || typeof Chart === "undefined" || !Array.isArray(data.labels) || !data.labels.length) {
    ready();
    return;
  }

  Chart.defaults.color = "#5c6570";
  Chart.defaults.borderColor = "#d8d4cc";
  Chart.defaults.font.family = "Manrope, Segoe UI, system-ui, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.animation = false;
  Chart.defaults.devicePixelRatio = 2;

  const grid = { color: "#ece8e0" };
  const ticks = {
    color: "#5c6570",
    maxRotation: 0,
    minRotation: 0,
    autoSkip: true,
    maxTicksLimit: 6,
    padding: 4,
  };
  const layout = { padding: { top: 6, right: 10, bottom: 2, left: 4 } };
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    layout,
    plugins: {
      legend: {
        position: "top",
        align: "end",
        labels: { boxWidth: 10, boxHeight: 10, padding: 10 },
      },
    },
  };

  const rms = document.getElementById("reportRmsChart");
  if (rms) {
    new Chart(rms, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "RMS (g)",
            data: data.rms,
            borderColor: "#2a7d7a",
            backgroundColor: "#2a7d7a",
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0,
          },
          {
            label: "Peak (g)",
            data: data.peak,
            borderColor: "#1a2744",
            backgroundColor: "#1a2744",
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        ...baseOptions,
        scales: {
          x: { grid: { display: false }, ticks },
          y: { beginAtZero: true, title: { display: true, text: "Acceleration (g)" }, grid, ticks },
        },
      },
    });
  }

  const anomaly = document.getElementById("reportAnomalyChart");
  if (anomaly) {
    new Chart(anomaly, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Isolation Forest score",
            data: data.anomaly,
            borderColor: "#1a2744",
            backgroundColor: "#d7e4e3",
            fill: true,
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0,
          },
          {
            label: "Train early-life mean",
            data: data.ifBaseline == null ? [] : data.labels.map(() => data.ifBaseline),
            borderColor: "#2a7d7a",
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        ...baseOptions,
        scales: {
          x: { grid: { display: false }, ticks },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Anomaly score (−score_samples)" },
            grid,
            ticks,
          },
        },
      },
    });
  }

  const fail = document.getElementById("reportFailChart");
  if (fail) {
    new Chart(fail, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Failure probability",
            data: data.failProb,
            borderColor: "#1a2744",
            spanGaps: true,
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0,
          },
          {
            label: `Threshold ${data.classifierThreshold}`,
            data: data.classifierThreshold == null ? [] : data.labels.map(() => data.classifierThreshold),
            borderColor: "#8b2e2e",
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        ...baseOptions,
        scales: {
          x: { grid: { display: false }, ticks },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "Calibrated probability" },
            grid,
            ticks,
          },
        },
      },
    });
  }

  const shap = document.getElementById("reportShapChart");
  if (shap && data.shapLabels?.length) {
    new Chart(shap, {
      type: "bar",
      data: {
        labels: data.shapLabels,
        datasets: [
          {
            label: "Mean |SHAP|",
            data: data.shapValues,
            backgroundColor: "#2a7d7a",
            barThickness: 14,
          },
        ],
      },
      options: {
        ...baseOptions,
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, title: { display: true, text: "Mean |SHAP| (raw margin)" }, grid, ticks },
          y: { grid: { display: false }, ticks: { color: "#1a2744", font: { size: 10 } } },
        },
      },
    });
  }

  window.requestAnimationFrame(() => {
    window.setTimeout(ready, 250);
  });
})();
