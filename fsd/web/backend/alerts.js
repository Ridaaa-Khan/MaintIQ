import { getPool } from "../db.js";
import { isTesterUser } from "../auth.js";
import { listAlerts as listScoredAlerts } from "../lib/ims-data.js";
import { syncImsMachines } from "./machines.js";

export const ALERT_SEVERITIES = ["Critical", "High", "Medium", "Low"];
export const ALERT_STATUSES = ["New", "Acknowledged", "Investigating", "Resolved"];

const SEVERITY_CLASS = { Critical: "critical", High: "high", Medium: "medium", Low: "low" };
const STATUS_CLASS = { New: "new", Acknowledged: "ack", Investigating: "warning", Resolved: "resolved" };

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

export async function syncImsAlerts(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return;
  if (!(await isTesterUser(ownerId))) return;
  if (syncedOwners.has(ownerId)) return;
  await syncImsMachines(ownerId);
  const pool = await getPool();
  const [machines] = await pool.query(
    "SELECT id, public_id, name, asset_id FROM machines WHERE created_by = ? AND source = 'ims'",
    [ownerId]
  );
  const byPublic = new Map(machines.map((row) => [row.public_id, row]));
  for (const alert of listScoredAlerts()) {
    const machine = byPublic.get(alert.machineId);
    if (!machine) continue;
    await pool.query(
      `INSERT INTO alerts
        (alert_key, machine_id, severity, alert_type, title, trigger_text, status, occurred_at, value_text, threshold_text, why_text)
       VALUES (?, ?, ?, ?, ?, ?, 'New', ?, ?, ?, ?)
       ON DUPLICATE KEY UPDATE alert_key = alert_key`,
      [
        alert.id,
        machine.id,
        alert.severity,
        alert.type,
        alert.title,
        alert.trigger,
        alert.createdIso ? new Date(alert.createdIso) : new Date(),
        alert.value,
        alert.threshold,
        alert.why,
      ]
    );
  }
  syncedOwners.add(ownerId);
}

export async function listAlertRecords(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return [];
  await syncImsAlerts(ownerId);
  const pool = await getPool();
  const tester = await isTesterUser(ownerId);
  const [rows] = await pool.query(
    tester
      ? `SELECT a.*, m.public_id, m.name AS machine_name, m.asset_id
         FROM alerts a
         JOIN machines m ON m.id = a.machine_id
         WHERE m.created_by = ?
         ORDER BY a.occurred_at DESC, a.id DESC`
      : `SELECT a.*, m.public_id, m.name AS machine_name, m.asset_id
         FROM alerts a
         JOIN machines m ON m.id = a.machine_id
         WHERE m.created_by = ? AND m.source <> 'ims'
         ORDER BY a.occurred_at DESC, a.id DESC`,
    [ownerId]
  );
  return rows.map((row) => ({
    id: row.alert_key,
    severity: row.severity,
    severityClass: SEVERITY_CLASS[row.severity] || "high",
    type: row.alert_type,
    title: row.title,
    trigger: row.trigger_text,
    machine: row.machine_name,
    assetId: row.asset_id,
    machineId: row.public_id,
    machineHref: `/machines/${row.public_id}`,
    status: row.status,
    statusClass: STATUS_CLASS[row.status] || "new",
    created: fmtShort(row.occurred_at),
    createdIso: row.occurred_at instanceof Date ? row.occurred_at.toISOString() : String(row.occurred_at),
    createdDay: (() => {
      const occurred = row.occurred_at instanceof Date ? row.occurred_at : new Date(row.occurred_at);
      const month = String(occurred.getMonth() + 1).padStart(2, "0");
      const day = String(occurred.getDate()).padStart(2, "0");
      return `${occurred.getFullYear()}-${month}-${day}`;
    })(),
    resolved: row.resolved_at ? fmtShort(row.resolved_at) : "",
    resolvedIso: row.resolved_at ? new Date(row.resolved_at).toISOString() : "",
    resolvedLabel: row.resolved_at ? fmtShort(row.resolved_at) : "—",
    value: row.value_text,
    threshold: row.threshold_text,
    why: row.why_text,
    search: `${row.title} ${row.machine_name} ${row.asset_id} ${row.alert_type}`.toLowerCase(),
  }));
}

export async function countOpenAlerts(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return 0;
  await syncImsAlerts(ownerId);
  const pool = await getPool();
  const tester = await isTesterUser(ownerId);
  const [rows] = await pool.query(
    tester
      ? `SELECT COUNT(*) AS n
         FROM alerts a
         JOIN machines m ON m.id = a.machine_id
         WHERE m.created_by = ? AND a.status <> 'Resolved'`
      : `SELECT COUNT(*) AS n
         FROM alerts a
         JOIN machines m ON m.id = a.machine_id
         WHERE m.created_by = ? AND a.status <> 'Resolved' AND m.source <> 'ims'`,
    [ownerId]
  );
  return Number(rows[0].n);
}

export async function updateAlertStatus(alertKey, status, userId) {
  if (!ALERT_STATUSES.includes(status)) return { ok: false, error: "Unknown status." };
  const ownerId = Number(userId);
  if (!ownerId) return { ok: false, error: "Alert not found." };
  await syncImsAlerts(ownerId);
  const pool = await getPool();
  const resolvedAt = status === "Resolved" ? new Date() : null;
  let result;
  if (status === "Resolved") {
    [result] = await pool.query(
      `UPDATE alerts a
       JOIN machines m ON m.id = a.machine_id
       SET a.status = 'Resolved', a.resolved_at = COALESCE(a.resolved_at, NOW())
       WHERE a.alert_key = ? AND m.created_by = ?`,
      [alertKey, ownerId]
    );
  } else {
    [result] = await pool.query(
      `UPDATE alerts a
       JOIN machines m ON m.id = a.machine_id
       SET a.status = ?, a.resolved_at = NULL
       WHERE a.alert_key = ? AND m.created_by = ?`,
      [status, alertKey, ownerId]
    );
  }
  if (!result.affectedRows) return { ok: false, error: "Alert not found." };
  return { ok: true, status, resolvedAt: status === "Resolved" ? resolvedAt : null, userId: ownerId };
}

export function summarizeAlerts(alerts) {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  return {
    active: alerts.filter((a) => a.status !== "Resolved").length,
    critical: alerts.filter((a) => a.severity === "Critical" && a.status !== "Resolved").length,
    unacknowledged: alerts.filter((a) => a.status === "New").length,
    resolvedToday: alerts.filter((a) => {
      if (!a.resolvedIso || a.status !== "Resolved") return false;
      const when = new Date(a.resolvedIso);
      return !Number.isNaN(when.getTime()) && when >= startOfToday;
    }).length,
  };
}
