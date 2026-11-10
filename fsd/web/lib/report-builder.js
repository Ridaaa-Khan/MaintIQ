import { getImsMeta } from "./ims-data.js";

const IMS_SAMPLING_HZ = 20000;
const IMS_SNAPSHOT_SAMPLES = 20480;
const CHART_POINTS = 60;

function round(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

function stats(values) {
  const nums = values.filter((v) => v != null && !Number.isNaN(Number(v))).map(Number);
  if (!nums.length) return { min: null, max: null, mean: null, std: null, n: 0 };
  const n = nums.length;
  const mean = nums.reduce((a, b) => a + b, 0) / n;
  const variance = nums.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  return {
    min: round(Math.min(...nums), 4),
    max: round(Math.max(...nums), 4),
    mean: round(mean, 4),
    std: round(Math.sqrt(variance), 4),
    n,
  };
}

function fmtDate(iso) {
  if (!iso) return "";
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function fmtStamp(iso) {
  if (!iso) return "";
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function pctChange(now, base) {
  if (now == null || base == null || !base) return null;
  return round(((now - base) / base) * 100, 1);
}

function present(value) {
  if (value == null) return false;
  const text = String(value).trim();
  return text !== "" && text !== "—" && text !== "Not stored" && text !== "Not configured" && text !== "Unknown";
}

function uniqueAlerts(alerts) {
  const seen = new Set();
  return (alerts || []).filter((alert) => {
    const key = alert.id || `${alert.title}|${alert.created}|${alert.severity}|${alert.assetId}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function sampleSeries(series, maxPoints = CHART_POINTS) {
  if (!series.length) return [];
  if (series.length <= maxPoints) return series;
  const sampled = [];
  const lastIndex = series.length - 1;
  for (let i = 0; i < maxPoints - 1; i += 1) {
    const index = Math.round((i * lastIndex) / (maxPoints - 1));
    if (!sampled.length || sampled[sampled.length - 1] !== series[index]) {
      sampled.push(series[index]);
    }
  }
  sampled.push(series[lastIndex]);
  return sampled;
}

function nextNumber(sections, key, show) {
  if (!show) return;
  const used = Object.values(sections).filter((value) => typeof value === "number");
  sections[key] = used.length + 1;
}

export function assembleMachineReport({
  reportId,
  title,
  machine,
  alerts = [],
  runs = [],
  activity = [],
  generatedBy = "Maint-IQ",
  generatedAt = new Date(),
} = {}) {
  const meta = getImsMeta();
  const series = Array.isArray(machine.series) ? machine.series : [];
  const threshold = meta.classifierThreshold;
  const ifBaseline = meta.ifBaselineMean;
  const generatedIso = generatedAt instanceof Date ? generatedAt.toISOString() : String(generatedAt);
  const generatedAtLabel = fmtStamp(generatedIso);
  const isIms = machine.source === "ims";

  const last = series.length ? series[series.length - 1] : null;
  const first = series[0] || null;
  const earlyRms = machine.earlyMean?.rms ?? null;
  const earlyPeak = machine.earlyMean?.peak ?? null;

  const rmsStats = stats(series.map((point) => point.rms));
  const peakStats = stats(series.map((point) => point.peak));
  const anomalyStats = stats(series.map((point) => point.anomaly));
  const failStats = stats(series.map((point) => point.failProb));

  const flaggedIf = series.filter((point) => point.flag);
  const flaggedFail = series.filter((point) => point.failProb != null && point.failProb >= threshold);
  const predictedFail = series.filter((point) => point.failPred === 1);
  const aboveEarlyRms = earlyRms != null ? series.filter((point) => point.rms != null && point.rms > earlyRms) : [];

  const peakAnomaly = series.reduce((best, point) => {
    if (point.anomaly == null) return best;
    if (!best || point.anomaly > best.anomaly) return point;
    return best;
  }, null);
  const peakFail = series.reduce((best, point) => {
    if (point.failProb == null) return best;
    if (!best || point.failProb > best.failProb) return point;
    return best;
  }, null);

  const latestRms = last?.rms ?? (present(machine.rms) ? Number(String(machine.rms).replace(/[^\d.+-eE]/g, "")) : null);
  const latestPeak = last?.peak ?? null;
  const latestAnomaly = last?.anomaly ?? machine.anomalyScoreNum ?? null;
  const latestFailProb = last?.failProb ?? machine.failProbNum ?? null;
  const latestRul = last?.rul ?? machine.rulHours ?? null;
  const rmsDelta = pctChange(last?.rms, earlyRms);
  const peakDelta = pctChange(last?.peak, earlyPeak);
  const hasFailChart = series.some((point) => point.failProb != null);
  const hasAnomaly = series.some((point) => point.anomaly != null) || latestAnomaly != null;
  const hasRul = latestRul != null;
  const hasVibration = rmsStats.n > 0 || peakStats.n > 0 || latestRms != null;
  const machineAlerts = uniqueAlerts(alerts.filter((alert) => alert.machineId === machine.id));
  const openAlerts = machineAlerts.filter((alert) => alert.status !== "Resolved");
  const resolvedAlerts = machineAlerts.filter((alert) => alert.status === "Resolved");
  const machineRuns = (runs || []).filter((run) => run.machineId === machine.id);
  const notes = String(machine.notes || "").trim();
  const hasMaintenance = Boolean(activity.length || notes || machine.upload);
  const hasPredictions = hasAnomaly || hasFailChart || hasRul || predictedFail.length > 0;
  const hasFindings = hasAnomaly || hasFailChart || hasRul || flaggedIf.length > 0;

  const missing = {
    rms: series.filter((point) => point.rms == null).length,
    peak: series.filter((point) => point.peak == null).length,
    anomaly: series.filter((point) => point.anomaly == null).length,
    failProb: series.filter((point) => point.failProb == null).length,
  };

  const periodLabel =
    machine.periodLabel ||
    (machine.start && machine.end ? `${fmtDate(machine.start)} – ${fmtDate(machine.end)}` : generatedAtLabel);

  const datasetId = machine.dataset || (isIms && machine.testId ? `NASA-IMS-${machine.testId}` : "");
  const runId = machine.runId || machineRuns[0]?.id || "";

  const overviewRows = [];
  overviewRows.push({ label: "Machine name", value: machine.name });
  overviewRows.push({ label: "Asset ID", value: machine.assetId });
  if (present(machine.type)) overviewRows.push({ label: "Equipment type", value: machine.type });
  if (isIms && present(machine.testId)) {
    overviewRows.push({
      label: "IMS identity",
      value: `${machine.testId}${machine.bearingId != null ? `, bearing ${machine.bearingId}` : ""}`,
    });
  }
  if (present(machine.location) && machine.location !== "—") {
    overviewRows.push({ label: "Location", value: machine.location });
  }
  if (present(machine.description)) overviewRows.push({ label: "Description", value: machine.description });
  if (present(machine.dataSource)) overviewRows.push({ label: "Data source", value: machine.dataSource });
  if (present(machine.health) && machine.health !== "Unknown") {
    overviewRows.push({ label: "Current operational status", value: machine.health });
  }
  if (present(machine.monitoring) && machine.monitoring !== "Not configured") {
    overviewRows.push({ label: "Monitoring", value: machine.monitoring });
  }
  if (machine.createdAtLabel && machine.createdAtLabel !== "—") {
    overviewRows.push({ label: "Registered in Maint-IQ", value: machine.createdAtLabel });
  }
  if (present(periodLabel)) overviewRows.push({ label: "Analysis window", value: periodLabel });
  if (machine.nWindows) {
    overviewRows.push({
      label: "Processed record count",
      value: `${Number(machine.nWindows).toLocaleString()} operational snapshots${
        machine.nBins ? ` · ${machine.nBins} plotted 12-hour means` : ""
      }`,
    });
  }
  if (isIms) {
    overviewRows.push({
      label: "Sampling",
      value: `${IMS_SAMPLING_HZ.toLocaleString()} Hz · ${IMS_SNAPSHOT_SAMPLES.toLocaleString()} samples / snapshot`,
    });
  }
  if (present(datasetId)) overviewRows.push({ label: "Dataset ID", value: datasetId });
  if (present(runId)) overviewRows.push({ label: "Analysis run ID", value: runId });
  if (present(machine.documentedDamage)) {
    overviewRows.push({ label: "Documented damage", value: machine.documentedDamage });
  }
  overviewRows.push({ label: "Report generated", value: `${generatedAtLabel} by ${generatedBy}` });

  const sensorLatest = [];
  if (latestRms != null) sensorLatest.push({ label: "Latest RMS", value: `${round(latestRms, 4)} g`, at: last?.t ? fmtStamp(last.t) : machine.lastData });
  if (latestPeak != null) sensorLatest.push({ label: "Latest peak", value: `${round(latestPeak, 4)} g`, at: last?.t ? fmtStamp(last.t) : "" });
  if (machine.lastData && machine.lastData !== "—") {
    sensorLatest.push({ label: "Last scored timestamp", value: machine.lastData });
  }

  const sections = {};
  nextNumber(sections, "summary", true);
  nextNumber(sections, "findings", hasFindings);
  nextNumber(sections, "overview", true);
  nextNumber(sections, "sensors", hasVibration);
  nextNumber(sections, "predictions", hasPredictions);
  nextNumber(sections, "maintenance", hasMaintenance);
  nextNumber(sections, "alerts", machineAlerts.length > 0);
  nextNumber(sections, "conclusion", true);

  const figures = {};
  if (hasVibration && series.length) nextNumber(figures, "rms", true);
  if (hasAnomaly && series.length) nextNumber(figures, "anomaly", true);
  if (hasFailChart) nextNumber(figures, "fail", true);

  const sampled = sampleSeries(series);
  const findings = {
    rmsStats,
    peakStats,
    anomalyStats,
    failStats,
    flaggedIfCount: flaggedIf.length,
    flaggedFailCount: flaggedFail.length,
    predictedFailCount: predictedFail.length,
    aboveEarlyRmsCount: aboveEarlyRms.length,
    peakAnomalyScore: peakAnomaly?.anomaly ?? null,
    peakAnomalyAt: peakAnomaly?.t ?? null,
    peakFailProb: peakFail?.failProb ?? null,
    peakFailAt: peakFail?.t ?? null,
    latestRms: last?.rms ?? null,
    latestPeak: last?.peak ?? null,
    latestAnomaly,
    latestFailProb,
    latestRulHours: latestRul,
    rmsDeltaPct: rmsDelta,
    peakDeltaPct: peakDelta,
    earlyRms,
    earlyPeak,
    hasFailChart,
    peakAnomalyLabel: peakAnomaly?.anomaly == null ? "" : Number(peakAnomaly.anomaly).toFixed(2),
    peakFailLabel: peakFail?.failProb == null ? "" : Number(peakFail.failProb).toFixed(2),
    rulLabel: latestRul == null ? "" : `${round(latestRul, 1)} h`,
    seriesCount: series.length,
    missing,
  };

  return {
    id: reportId,
    name: title,
    type: "Condition assessment",
    generatedAt: generatedIso,
    generatedAtLabel,
    generatedBy,
    datasetId,
    runId,
    machine: {
      id: machine.id,
      name: machine.name,
      assetId: machine.assetId,
      type: machine.type,
      location: present(machine.location) ? machine.location : "",
      health: machine.health,
      healthClass: machine.healthClass,
      monitoring: machine.monitoring,
      dataSource: machine.dataSource,
      testId: machine.testId,
      bearingId: machine.bearingId,
      documentedDamage: machine.documentedDamage || "",
      periodLabel,
      nWindows: machine.nWindows,
      nBins: series.length,
      source: machine.source,
    },
    sections,
    figures,
    overviewRows,
    sensorLatest,
    findings,
    thresholds: hasPredictions || hasAnomaly
      ? {
          classifier: threshold,
          ifContamination: 0.01,
          ifBaselineMean: ifBaseline,
        }
      : null,
    narrative: {
      execSummary: buildExecutiveSummary({
        machine,
        series,
        flaggedIf,
        flaggedFail,
        predictedFail,
        last,
        rmsDelta,
        threshold,
        periodLabel,
        openAlerts,
      }),
      vibrationInterp: hasVibration
        ? buildVibrationInterp({ machine, last, first, rmsStats, peakStats, rmsDelta, aboveEarlyRms, missing })
        : "",
      anomalyInterp: hasAnomaly
        ? buildAnomalyInterp({ flaggedIf, series, peakAnomaly, ifBaseline, last })
        : "",
      failureInterp: hasFailChart || predictedFail.length || latestFailProb != null
        ? buildFailureInterp({ machine, threshold, peakFail, flaggedFail, predictedFail, series })
        : "",
      conclusion: buildConclusion({
        machine,
        flaggedIf,
        predictedFail,
        last,
        threshold,
        openAlerts,
        hasVibration,
      }),
    },
    activity: activity.map((row) => ({
      action: row.action,
      summary: row.summary,
      when: row.when,
      person: row.person || "",
    })),
    notes,
    upload: machine.upload || null,
    runs: machineRuns.map((run) => ({
      id: run.id,
      status: run.status,
      dataset: run.dataset,
      periodLabel: run.periodLabel,
      updated: run.updated,
      windowLabel: run.windowLabel,
    })),
    alerts: {
      openCount: openAlerts.length,
      resolvedCount: resolvedAlerts.length,
      anomalyFlags: flaggedIf.length,
      items: machineAlerts.slice(0, 40).map((alert) => ({
        title: alert.title,
        type: alert.type,
        severity: alert.severity,
        status: alert.status,
        created: alert.created,
        value: alert.value,
        threshold: alert.threshold,
        why: alert.why,
      })),
    },
    charts: {
      labels: sampled.map((point) => fmtDate(point.t)),
      rms: sampled.map((point) => point.rms),
      peak: sampled.map((point) => point.peak),
      anomaly: sampled.map((point) => point.anomaly),
      failProb: sampled.map((point) => point.failProb),
      ifBaseline: hasAnomaly ? ifBaseline : null,
      classifierThreshold: hasFailChart ? threshold : null,
    },
  };
}

function buildExecutiveSummary({
  machine,
  series,
  flaggedIf,
  flaggedFail,
  predictedFail,
  last,
  rmsDelta,
  threshold,
  periodLabel,
  openAlerts,
}) {
  const parts = [`This report covers ${machine.name} (${machine.assetId})`];
  if (periodLabel) parts[0] += ` for ${periodLabel}`;
  parts[0] += ".";
  if (machine.nWindows) {
    parts.push(
      ` Stored scoring uses ${Number(machine.nWindows).toLocaleString()} operational snapshots${
        series.length ? ` aggregated into ${series.length} twelve-hour bins` : ""
      }.`
    );
  } else if (!series.length) {
    parts.push(" No scored vibration windows are stored for this machine.");
  }
  if (machine.documentedDamage) {
    parts.push(` IMS documentation records ${String(machine.documentedDamage).replace(/^Documented /i, "")}.`);
  } else if (machine.source === "ims") {
    parts.push(" IMS documentation does not list a failure for this bearing.");
  }
  if (rmsDelta != null && last?.rms != null && machine.earlyMean?.rms != null) {
    parts.push(
      ` Latest 12-hour mean RMS is ${last.rms} g, ${
        rmsDelta > 0 ? `${rmsDelta}% above` : `${Math.abs(rmsDelta)}% below`
      } the early-life mean of ${machine.earlyMean.rms} g.`
    );
  }
  if (series.length) {
    parts.push(` Isolation Forest flagged ${flaggedIf.length} of ${series.length} 12-hour bins in this window.`);
  }
  if (last?.failProb == null && flaggedFail.length === 0 && predictedFail.length === 0) {
    if (series.length) {
      parts.push(" Calibrated failure-within-24h probabilities are not stored for every bin of this bearing.");
    }
  } else {
    parts.push(
      ` The 24-hour classifier (threshold ${threshold}) marked ${predictedFail.length} bin${
        predictedFail.length === 1 ? "" : "s"
      } as risk and ${flaggedFail.length} bin${flaggedFail.length === 1 ? "" : "s"} at or above threshold.`
    );
  }
  if (openAlerts.length) {
    parts.push(` ${openAlerts.length} open alert${openAlerts.length === 1 ? "" : "s"} are stored for this asset.`);
  }
  if (present(machine.health) && machine.health !== "Unknown") {
    parts.push(` Overall stored condition is ${machine.health}.`);
  }
  parts.push(" These results warrant review; they do not independently confirm mechanical failure.");
  return parts.join("");
}

function buildVibrationInterp({ machine, last, first, rmsStats, peakStats, rmsDelta, aboveEarlyRms, missing }) {
  if (!last && !rmsStats.n) return "No scored vibration windows are available in this period.";
  const bits = [];
  if (aboveEarlyRms.length && machine.earlyMean?.rms != null) {
    bits.push(`${aboveEarlyRms.length} of the plotted bins have RMS above this run’s early-life mean.`);
  }
  if (peakStats.n) {
    bits.push(`Peak vibration in the series ranges from ${peakStats.min} g to ${peakStats.max} g (mean ${peakStats.mean} g).`);
  }
  if (rmsDelta != null && last?.rms != null) {
    bits.push(
      `RMS moved from ${first?.rms ?? "—"} g at the start of the plotted series to ${last.rms} g (${
        rmsDelta > 0 ? "+" : ""
      }${rmsDelta}% vs early-life mean ${machine.earlyMean.rms} g).`
    );
  }
  bits.push("Units are g; values are 12-hour means of processed snapshot features.");
  if (missing.rms || missing.peak) {
    bits.push(
      `Missing values in this series: RMS ${missing.rms} bin${missing.rms === 1 ? "" : "s"}, peak ${missing.peak} bin${
        missing.peak === 1 ? "" : "s"
      }.`
    );
  }
  bits.push("No temperature channel is stored for this machine, so none is reported.");
  return bits.join(" ");
}

function buildAnomalyInterp({ flaggedIf, series, peakAnomaly, ifBaseline, last }) {
  const bits = [
    "Scores are −score_samples from the selected 1% contamination detector (higher = more unlike the early-life baseline).",
    `${flaggedIf.length} of ${series.length} bins have anomaly_flag = true.`,
  ];
  if (ifBaseline != null) {
    bits.push(
      `A dashed reference shows the train early-life mean (${ifBaseline}); that mean is not a decision threshold.`
    );
  }
  bits.push("Flags come from the fitted detector.");
  if (peakAnomaly) {
    bits.push(`The highest Isolation Forest score in this window is ${peakAnomaly.anomaly} at ${fmtStamp(peakAnomaly.t)}.`);
  }
  if (last?.anomaly != null) bits.push(`Latest score: ${last.anomaly}.`);
  return bits.join(" ");
}

function buildFailureInterp({ machine, threshold, peakFail, flaggedFail, predictedFail, series }) {
  if (series.every((point) => point.failProb == null) && !peakFail) {
    return `No calibrated failure-within-24h probabilities are stored for ${machine.assetId}. The operating threshold remains ${threshold}, but crossings cannot be counted.`;
  }
  const bits = [
    "The classifier answers whether the operational endpoint is likely inside 24 hours.",
    `Threshold ${threshold} is the stored operating point.`,
    `${flaggedFail.length} bins are at or above threshold; ${predictedFail.length} bins have predicted_failure_within_24h = 1.`,
  ];
  if (peakFail) bits.push(`Peak probability is ${peakFail.failProb} at ${fmtStamp(peakFail.t)}.`);
  bits.push("This is a horizon flag, not a confirmed failure.");
  return bits.join(" ");
}

function buildConclusion({ machine, flaggedIf, predictedFail, last, threshold, openAlerts, hasVibration }) {
  if (!hasVibration && !openAlerts.length) {
    return `${machine.name} has a stored profile in Maint-IQ, but no scored vibration windows or alerts were available when this report was generated.`;
  }
  const follow =
    flaggedIf.length || predictedFail.length || openAlerts.length
      ? " Suggested follow-up is to review the flagged windows against the early-life baseline and the stored scores, then continue monitoring."
      : " Suggested follow-up is to keep the current monitoring window and re-score if new snapshots are added.";
  const rulBit =
    last?.rul == null
      ? " No RUL estimate is stored for this bearing (RUL was trained on failed trajectories only)."
      : ` The last stored RUL estimate is ${last.rul} h and remains experimental.`;
  const healthBit = present(machine.health) && machine.health !== "Unknown" ? ` Stored health is ${machine.health}.` : "";
  return `${machine.name} shows ${flaggedIf.length} Isolation Forest flags and ${predictedFail.length} classifier risk bin(s) at threshold ${threshold} in this historical run.${rulBit}${healthBit}${follow}`;
}
