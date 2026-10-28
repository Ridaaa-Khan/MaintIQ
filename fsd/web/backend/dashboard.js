import { listMachineRecords } from "./machines.js";
import { listAlertRecords } from "./alerts.js";

function sampleSeries(series, maxPoints = 48) {
  if (!series?.length) return { labels: [], anomaly: [], rms: [] };
  const step = Math.max(1, Math.floor(series.length / maxPoints));
  const sampled = series.filter((_, index) => index % step === 0 || index === series.length - 1);
  return {
    labels: sampled.map((point) =>
      new Date(point.t).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })
    ),
    anomaly: sampled.map((point) => point.anomaly),
    rms: sampled.map((point) => point.rms),
  };
}

export async function getDashboardPage(userId) {
  const machines = await listMachineRecords(userId);
  const alerts = await listAlertRecords(userId);
  const healthCounts = { Healthy: 0, Warning: 0, Critical: 0, Unknown: 0 };
  for (const machine of machines) {
    healthCounts[machine.health] = (healthCounts[machine.health] || 0) + 1;
  }
  const scored = machines.filter((machine) => machine.series?.length);
  const trends = {};
  for (const machine of scored) {
    trends[machine.id] = sampleSeries(machine.series);
  }
  const attention = machines
    .filter((machine) => machine.health === "Critical" || machine.health === "Warning")
    .map((machine) => ({
      id: machine.id,
      name: machine.name,
      assetId: machine.assetId,
      health: machine.health,
      healthClass: machine.healthClass,
      detail: `Anomaly ${machine.anomalyScore} · Fail. prob. ${machine.failProb}`,
    }));
  const recentAlerts = alerts.slice(0, 8).map((alert) => ({
    machine: alert.machine,
    type: alert.type,
    severity: alert.severity,
    severityClass: alert.severityClass,
    time: alert.created,
    status: alert.status,
    statusClass: alert.statusClass,
  }));
  const activity = [
    ...machines
      .filter((machine) => machine.runId)
      .map((machine) => ({
        text: `Stored scores loaded · ${machine.name} (${machine.runId})`,
        when: machine.periodLabel,
      })),
    ...alerts.slice(0, 3).map((alert) => ({
      text: `${alert.title} · ${alert.machine}`,
      when: alert.created,
    })),
  ].slice(0, 6);

  return {
    totalMachines: machines.length,
    healthy: healthCounts.Healthy || 0,
    attention: (healthCounts.Warning || 0) + (healthCounts.Critical || 0),
    activeAlerts: alerts.filter((alert) => alert.status !== "Resolved").length,
    welcomeHint: `${machines.length} machines are on record. ${scored.length} have scored IMS windows. ${alerts.length} stored alert${alerts.length === 1 ? "" : "s"}.`,
    attentionList: attention,
    recentAlerts,
    activity,
    healthCounts,
    trendMachines: scored.map((machine) => ({
      id: machine.id,
      name: machine.name,
      assetId: machine.assetId,
    })),
    trends,
    defaultTrendId: scored.find((machine) => machine.id === "brg-003")?.id || scored[0]?.id || "",
  };
}
