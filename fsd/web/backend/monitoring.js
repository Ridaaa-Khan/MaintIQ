import { getPool } from "../db.js";
import { listMonitoringRuns, getMachine } from "../lib/ims-data.js";
import { syncImsMachines } from "./machines.js";

export const RUN_STATUSES = ["Queued", "Running", "Paused", "Completed", "Failed"];

const syncedOwners = new Set();

function fmtShort(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export async function syncImsRuns(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return;
  if (syncedOwners.has(ownerId)) return;
  await syncImsMachines(ownerId);
  const pool = await getPool();
  const [machines] = await pool.query(
    "SELECT id, public_id FROM machines WHERE source = 'ims' AND created_by = ?",
    [ownerId]
  );
  const byPublic = new Map(machines.map((row) => [row.public_id, row.id]));
  for (const run of listMonitoringRuns()) {
    const machineId = byPublic.get(run.machineId);
    if (!machineId) continue;
    const scored = getMachine(run.machineId);
    const [existing] = await pool.query(
      "SELECT id FROM analysis_runs WHERE run_code = ? AND machine_id = ? LIMIT 1",
      [run.id, machineId]
    );
    if (existing.length) continue;
    await pool.query(
      `INSERT INTO analysis_runs
        (machine_id, run_code, status, stage, window_label, progress, dataset_label, period_label,
         n_windows, n_bins, scope, started_at, updated_at, error_message)
       VALUES (?, ?, 'Completed', ?, ?, 100, ?, ?, ?, ?, 'history', ?, ?, NULL)`,
      [
        machineId,
        run.id,
        "Scoring stored",
        run.windowLabel,
        run.dataset,
        run.periodLabel,
        scored?.nWindows ?? null,
        scored?.nBins ?? null,
        scored?.start ? new Date(scored.start) : null,
        scored?.end ? new Date(scored.end) : null,
      ]
    );
  }
  syncedOwners.add(ownerId);
}

export async function listRunRecords(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return [];
  await syncImsRuns(ownerId);
  const pool = await getPool();
  const [rows] = await pool.query(
    `SELECT r.*, m.public_id, m.name, m.asset_id, m.machine_type, m.test_id, m.bearing_id, m.notes
     FROM analysis_runs r
     JOIN machines m ON m.id = r.machine_id
     WHERE m.created_by = ?
     ORDER BY r.updated_at DESC, r.id DESC`,
    [ownerId]
  );
  return rows.map((row) => {
    const scored = getMachine(row.public_id);
    const updatedIso = row.updated_at ? new Date(row.updated_at).toISOString() : "";
    const year = updatedIso ? updatedIso.slice(0, 4) : "";
    return {
      id: row.run_code,
      machineId: row.public_id,
      name: row.name,
      assetId: row.asset_id,
      status: row.status,
      statusClass: row.status === "Failed" ? "critical" : row.status === "Paused" ? "paused" : row.status === "Running" || row.status === "Queued" ? "active" : "resolved",
      dataset: row.dataset_label,
      stage: row.stage,
      windowLabel: row.window_label,
      progress: row.progress == null ? null : Number(row.progress),
      updated: fmtShort(row.updated_at),
      updatedIso,
      periodLabel: row.period_label,
      year,
      search: `${row.name} ${row.asset_id} ${row.run_code} ${row.dataset_label}`.toLowerCase(),
      scope: row.scope,
      nWindows: row.n_windows,
      nBins: row.n_bins,
      error: row.error_message || "",
      testId: row.test_id,
      bearingId: row.bearing_id,
      anomalyScore: scored?.anomalyScore ?? null,
      rms: scored?.rms ?? null,
      failProb: scored?.failProb ?? null,
      rulHours: scored?.rulHours ?? null,
      documentedDamage: scored?.documentedDamage ?? null,
      health: scored?.health ?? null,
    };
  });
}

export function summarizeRuns(runs) {
  const latest = [...runs].sort((a, b) => String(b.updatedIso).localeCompare(String(a.updatedIso)))[0];
  return {
    running: runs.filter((r) => r.status === "Running" || r.status === "Queued").length,
    paused: runs.filter((r) => r.status === "Paused").length,
    completed: runs.filter((r) => r.status === "Completed").length,
    failed: runs.filter((r) => r.status === "Failed").length,
    lastRunLabel: latest ? latest.updated : "—",
    years: [...new Set(runs.map((r) => r.year).filter(Boolean))].sort(),
  };
}
