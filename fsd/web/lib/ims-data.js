import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ANALYTICS_JS = path.join(__dirname, "..", "public", "js", "analytics-data.js");

function loadAnalytics() {
  const raw = fs.readFileSync(ANALYTICS_JS, "utf8");
  const json = raw.replace(/^\s*window\.ANALYTICS_DATA\s*=\s*/, "").replace(/;\s*$/, "");
  return JSON.parse(json);
}

function fmtShort(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDay(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function fmtPeriod(start, end) {
  return `${fmtDay(start)} – ${fmtDay(end)}`;
}

function round(value, digits = 4) {
  if (value == null || Number.isNaN(value)) return null;
  return Number(Number(value).toFixed(digits));
}

function initials(name) {
  return String(name || "M")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function deriveHealth(machine, threshold, ifBaseline) {
  const latest = machine.latest || {};
  const early = machine.earlyMean || {};
  const critical =
    latest.anomalyFlag === true ||
    latest.predictedFailure24h === 1 ||
    (latest.failureProbability != null && latest.failureProbability >= threshold);

  if (critical) {
    return { health: "Critical", healthClass: "critical" };
  }

  const elevatedAnomaly =
    latest.anomalyScore != null && ifBaseline != null && latest.anomalyScore > ifBaseline * 1.15;
  const elevatedRms =
    latest.rms != null && early.rms != null && early.rms > 0 && latest.rms > early.rms * 2;

  if (elevatedAnomaly || elevatedRms) {
    return { health: "Warning", healthClass: "warning" };
  }

  return { health: "Healthy", healthClass: "healthy" };
}

function buildMachine(id, raw, threshold, ifBaseline) {
  const series = raw.series || [];
  const latest = raw.latest || {};
  const { health, healthClass } = deriveHealth(raw, threshold, ifBaseline);

  const flaggedIf = series.filter((p) => p.flag).length;
  const flaggedFail = series.filter(
    (p) => p.failProb != null && p.failProb >= threshold
  ).length;
  const predictedFail = series.filter((p) => p.failPred === 1).length;
  const openAlerts = flaggedIf + flaggedFail;

  const failProb =
    latest.failureProbability == null ? null : round(latest.failureProbability, 4);
  const anomalyScore =
    latest.anomalyScore == null ? null : round(latest.anomalyScore, 4);

  const notesParts = [];
  if (raw.documentedDamage) notesParts.push(raw.documentedDamage + ".");
  notesParts.push(
    `Isolation Forest flagged ${flaggedIf} of ${series.length} twelve-hour bins; classifier marked ${predictedFail} risk bin(s) at threshold ${threshold}.`
  );
  if (failProb == null) {
    notesParts.push("Calibrated failure probability is not stored for every window of this bearing.");
  }

  const lastIso = latest.timestamp || raw.end;
  const thumb = initials(raw.name || id);

  return {
    id,
    name: raw.name,
    assetId: raw.assetId,
    type: "Bearing",
    location: `IMS ${raw.testId}`,
    description: raw.documentedDamage
      ? `NASA IMS ${raw.testId} bearing ${raw.bearingId} — ${raw.documentedDamage}`
      : `NASA IMS ${raw.testId} bearing ${raw.bearingId} (no documented failure label)`,
    health,
    healthClass,
    monitoring: "Active",
    monitoringClass: "active",
    anomalyScore: anomalyScore == null ? "—" : String(anomalyScore),
    anomalyScoreNum: anomalyScore,
    lastData: fmtShort(lastIso),
    lastIso,
    dataSource: `Uploaded NASA IMS dataset (${raw.testId})`,
    dataset: `NASA-IMS-${raw.testId}`,
    split: raw.split || null,
    rms: latest.rms == null ? "—" : `${round(latest.rms, 4)} g`,
    peak: latest.peak == null ? "—" : `${round(latest.peak, 4)} g`,
    failProb: failProb == null ? "Not stored" : String(failProb),
    failProbNum: failProb,
    rulHours: latest.predictedRulHours == null ? null : round(latest.predictedRulHours, 1),
    openAlerts: String(openAlerts),
    openAlertsNum: openAlerts,
    notes: notesParts.join(" "),
    testId: raw.testId,
    bearingId: raw.bearingId,
    documentedDamage: raw.documentedDamage,
    nWindows: raw.nWindows,
    nBins: series.length,
    periodLabel: fmtPeriod(raw.start, raw.end),
    start: raw.start,
    end: raw.end,
    earlyMean: raw.earlyMean,
    lateMean: raw.lateMean,
    latest: raw.latest || null,
    latestBands: Array.isArray(raw.latestBands) ? raw.latestBands : [],
    latestFaultFreq: raw.latestFaultFreq || null,
    series,
    recentWindows: raw.recentWindows || [],
    thumb,
    scoreSort: health === "Critical" ? 10 : health === "Warning" ? 50 : 90,
    runId: `SCORE-${raw.testId}-B${raw.bearingId}`,
  };
}

function buildAlerts(machines, threshold, ifBaseline) {
  const alerts = [];

  for (const machine of machines) {
    for (const point of machine.series) {
      if (point.flag) {
        const score = point.anomaly;
        alerts.push({
          id: `${machine.id}|if|${point.t}`,
          severity: score != null && score >= 0.7 ? "Critical" : "High",
          severityClass: score != null && score >= 0.7 ? "critical" : "high",
          type: "Anomaly detection",
          title: "Isolation Forest anomaly flag",
          trigger: `Score ${score} · train early-life mean ${ifBaseline} (reference, not a cut)`,
          machine: machine.name,
          assetId: machine.assetId,
          machineId: machine.id,
          machineHref: `/machines/${machine.id}`,
          status: "New",
          statusClass: "new",
          created: fmtShort(point.t),
          createdIso: point.t,
          resolved: "",
          resolvedLabel: "—",
          value: score == null ? "—" : String(score),
          threshold: String(ifBaseline),
          why: [
            `anomaly_flag = true on the 12-hour bin at ${fmtShort(point.t)}.`,
            `Isolation Forest score ${score}; dashed reference on charts is the train early-life mean (${ifBaseline}), not an investigation cut.`,
            "Flags mean the window looks unlike the early-life baseline — not a confirmed mechanical failure.",
          ].join(";"),
          search: `isolation forest ${machine.name} ${machine.assetId} anomaly`.toLowerCase(),
        });
      }

      if (point.failProb != null && point.failProb >= threshold) {
        alerts.push({
          id: `${machine.id}|clf|${point.t}`,
          severity: "Critical",
          severityClass: "critical",
          type: "Failure prediction",
          title: "Failure-within-24h probability at or above threshold",
          trigger: `Probability ${point.failProb} · threshold ${threshold}`,
          machine: machine.name,
          assetId: machine.assetId,
          machineId: machine.id,
          machineHref: `/machines/${machine.id}`,
          status: "New",
          statusClass: "new",
          created: fmtShort(point.t),
          createdIso: point.t,
          resolved: "",
          resolvedLabel: "—",
          value: String(point.failProb),
          threshold: String(threshold),
          why: [
            `Calibrated probability ${point.failProb} met or exceeded threshold ${threshold} at ${fmtShort(point.t)}.`,
            point.failPred === 1
              ? "Predicted class for this bin is risk (predicted_failure_within_24h = 1)."
              : "Probability crossed the configured operating point.",
            "This is a horizon flag from the stored classifier scores, not a confirmed failure.",
          ].join(";"),
          search: `failure probability ${machine.name} ${machine.assetId} classifier`.toLowerCase(),
        });
      }
    }
  }

  alerts.sort((a, b) => String(b.createdIso).localeCompare(String(a.createdIso)));
  return alerts;
}

function buildRuns(machines) {
  return machines.map((machine) => {
    const updatedIso = machine.end || machine.lastIso;
    return {
      id: machine.runId,
      machineId: machine.id,
      name: machine.name,
      assetId: machine.assetId,
      status: "Completed",
      statusClass: "resolved",
      dataset: machine.dataSource,
      stage: "Scoring complete",
      windowLabel: `${machine.nWindows.toLocaleString()} operational snapshots · ${machine.nBins} twelve-hour means`,
      progress: 100,
      updated: fmtShort(updatedIso),
      updatedIso,
      periodLabel: machine.periodLabel,
      search: `${machine.name} ${machine.assetId} ${machine.runId} ${machine.dataset}`.toLowerCase(),
      scope: "history",
      dateBucket: "month",
    };
  });
}

function chartSeries(machine, maxPoints = 48) {
  const series = machine.series || [];
  if (!series.length) {
    return { labels: [], anomaly: [], rms: [] };
  }
  const step = Math.max(1, Math.floor(series.length / maxPoints));
  const sampled = series.filter((_, i) => i % step === 0 || i === series.length - 1);
  return {
    labels: sampled.map((p) => fmtDay(p.t)),
    anomaly: sampled.map((p) => p.anomaly),
    rms: sampled.map((p) => p.rms),
  };
}

let cache = null;
let cacheMtimeMs = null;

function getBundle() {
  const mtimeMs = fs.statSync(ANALYTICS_JS).mtimeMs;
  if (cache && cacheMtimeMs === mtimeMs) return cache;
  cacheMtimeMs = mtimeMs;

  const analytics = loadAnalytics();
  const threshold = analytics.classifierThreshold;
  const ifBaseline = analytics.ifBaselineMean;
  const machines = Object.entries(analytics.machines || {}).map(([id, raw]) =>
    buildMachine(id, raw, threshold, ifBaseline)
  );
  const testOrder = { "1st_test": 1, "2nd_test": 2, "3rd_test": 3 };
  machines.sort((a, b) => {
    const ta = testOrder[a.testId] || 99;
    const tb = testOrder[b.testId] || 99;
    if (ta !== tb) return ta - tb;
    return a.bearingId - b.bearingId;
  });

  const alerts = buildAlerts(machines, threshold, ifBaseline);
  const runs = buildRuns(machines);

  const healthCounts = { Healthy: 0, Warning: 0, Critical: 0 };
  for (const m of machines) {
    healthCounts[m.health] = (healthCounts[m.health] || 0) + 1;
  }

  const primary = machines.find((m) => m.id === "brg-003") || machines[0];
  const trend = primary ? chartSeries(primary) : { labels: [], anomaly: [], rms: [] };

  const attention = machines
    .filter((m) => m.health === "Critical" || m.health === "Warning")
    .map((m) => ({
      id: m.id,
      name: m.name,
      assetId: m.assetId,
      health: m.health,
      healthClass: m.healthClass,
      detail: `Anomaly ${m.anomalyScore} · Fail. prob. ${m.failProb}`,
    }));

  const recentAlerts = alerts.slice(0, 5).map((a) => ({
    machine: a.machine,
    type: a.type,
    severity: a.severity,
    severityClass: a.severityClass,
    time: a.created,
    status: a.status,
    statusClass: a.statusClass,
  }));

  const activity = [
    ...machines.map((m) => ({
      text: `Historical scoring complete · ${m.name} (${m.runId})`,
      when: m.periodLabel,
    })),
    ...alerts.slice(0, 3).map((a) => ({
      text: `${a.title} · ${a.machine}`,
      when: a.created,
    })),
  ].slice(0, 6);

  const criticalAlerts = alerts.filter((a) => a.severity === "Critical").length;
  const unacked = alerts.filter((a) => a.status === "New").length;

  cache = {
    analytics,
    threshold,
    ifBaseline,
    machines,
    alerts,
    runs,
    healthCounts,
    dashboard: {
      totalMachines: machines.length,
      healthy: healthCounts.Healthy || 0,
      attention: (healthCounts.Warning || 0) + (healthCounts.Critical || 0),
      activeAlerts: alerts.length,
      criticalAlerts,
      unacknowledged: unacked,
      resolvedToday: 0,
      welcomeHint:
        machines.length === 0
          ? "No scored IMS machines are loaded."
          : `${machines.length} bearings from the uploaded NASA IMS dataset (1st / 2nd / 3rd_test) with stored scores. ${alerts.length} model flag${alerts.length === 1 ? "" : "s"} from the scoring run.`,
      attentionList: attention,
      recentAlerts,
      activity,
      healthCounts,
      trend,
      ifBaseline,
      primaryMachine: primary
        ? { id: primary.id, name: primary.name, assetId: primary.assetId }
        : null,
    },
  };

  return cache;
}

export function listMachines() {
  return getBundle().machines;
}

export function getMachine(id) {
  return getBundle().machines.find((m) => m.id === id) || null;
}

export function listAlerts() {
  return getBundle().alerts;
}

export function alertCount() {
  return getBundle().alerts.length;
}

export function listMonitoringRuns() {
  return getBundle().runs;
}

export function getDashboard() {
  return getBundle().dashboard;
}

export function getImsMeta() {
  const { threshold, ifBaseline, analytics } = getBundle();
  return {
    source: analytics.source,
    classifierThreshold: threshold,
    ifBaselineMean: ifBaseline,
    ifContamination: analytics.ifContamination ?? 0.01,
    metrics: analytics.metrics || null,
  };
}
