import { randomBytes } from "crypto";
import { getPool } from "../db.js";
import { isTesterUser } from "../auth.js";
import { assembleMachineReport } from "../lib/report-builder.js";
import { getMachineRecord, listMachineRecords } from "./machines.js";
import { listAlertRecords } from "./alerts.js";
import { listRunRecords } from "./monitoring.js";

const TITLE_MAX = 120;

function publicReportId() {
  return `miq-rpt-${randomBytes(4).toString("hex")}`;
}

function cleanTitle(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, TITLE_MAX);
}

function mapListRow(row) {
  return {
    id: row.public_id,
    name: row.title,
    type: row.report_type,
    machineKey: row.machine_public_id,
    machineName: row.machine_name,
    assetId: row.asset_id,
    period: row.period_label || "—",
    datasetId: row.dataset_id || "—",
    runId: row.run_id || "—",
    createdBy: [row.first_name, row.last_name].filter(Boolean).join(" ") || "Maint-IQ",
    createdAt: row.created_at,
  };
}

export async function listReportMachines(userId) {
  const machines = await listMachineRecords(userId);
  return machines.map((machine) => ({
    id: machine.id,
    name: machine.name,
    assetId: machine.assetId,
  }));
}

export async function listReports(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return [];
  try {
    const pool = await getPool();
    const tester = await isTesterUser(ownerId);
    const [rows] = await pool.query(
      `SELECT r.public_id, r.title, r.report_type, r.period_label, r.dataset_id, r.run_id, r.created_at,
              m.public_id AS machine_public_id, m.name AS machine_name, m.asset_id,
              u.first_name, u.last_name
       FROM generated_reports r
       JOIN machines m ON m.id = r.machine_id
       LEFT JOIN users u ON u.id = r.created_by
       WHERE r.created_by = ? AND m.created_by = ?
         AND (? = 1 OR m.source <> 'ims')
       ORDER BY r.created_at DESC, r.id DESC`,
      [ownerId, ownerId, tester ? 1 : 0]
    );
    return rows.map(mapListRow);
  } catch (err) {
    console.error(err);
    throw err;
  }
}

export async function getStoredReport(publicId, userId) {
  const ownerId = Number(userId);
  if (!ownerId) return null;
  try {
    const pool = await getPool();
    const tester = await isTesterUser(ownerId);
    const [rows] = await pool.query(
      `SELECT r.payload_json
       FROM generated_reports r
       JOIN machines m ON m.id = r.machine_id
       WHERE r.public_id = ? AND r.created_by = ? AND m.created_by = ?
         AND (? = 1 OR m.source <> 'ims')
       LIMIT 1`,
      [String(publicId || ""), ownerId, ownerId, tester ? 1 : 0]
    );
    if (!rows.length) return null;
    return JSON.parse(rows[0].payload_json);
  } catch (err) {
    console.error(err);
    return null;
  }
}

export async function createGeneratedReport({ machinePublicId, title, generatedBy, userId }) {
  const clean = cleanTitle(title);
  const machineId = String(machinePublicId || "").trim();
  const errors = [];
  if (!machineId) errors.push("Choose a machine.");
  if (!clean) errors.push("Enter a report title.");
  if (clean.length > TITLE_MAX) errors.push("Title must be 120 characters or fewer.");
  if (errors.length) return { ok: false, errors };

  try {
    const machine = await getMachineRecord(machineId, userId);
    if (!machine) return { ok: false, errors: ["That machine is not in the register."] };

    const pool = await getPool();
    const [activityRows] = await pool.query(
      `SELECT a.action, a.summary, a.created_at, u.first_name, u.last_name
       FROM machine_activity a
       LEFT JOIN users u ON u.id = a.user_id
       WHERE a.machine_id = ?
       ORDER BY a.id DESC
       LIMIT 40`,
      [machine.dbId]
    );
    const activity = activityRows.map((row) => ({
      action: row.action,
      summary: row.summary,
      when: row.created_at
        ? new Date(row.created_at).toLocaleString("en-GB", {
            day: "numeric",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })
        : "",
      person: [row.first_name, row.last_name].filter(Boolean).join(" "),
    }));

    const alerts = (await listAlertRecords(userId)).filter((alert) => alert.machineId === machine.id);
    const runs = (await listRunRecords(userId)).filter((run) => run.machineId === machine.id);
    const reportId = publicReportId();
    const payload = assembleMachineReport({
      reportId,
      title: clean,
      machine,
      alerts,
      runs,
      activity,
      generatedBy,
      generatedAt: new Date(),
    });

    await pool.query(
      `INSERT INTO generated_reports
        (public_id, machine_id, title, report_type, period_label, dataset_id, run_id, payload_json, created_by)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        reportId,
        machine.dbId,
        clean,
        payload.type,
        payload.machine.periodLabel || "",
        payload.datasetId || "",
        payload.runId || "",
        JSON.stringify(payload),
        userId || null,
      ]
    );

    return { ok: true, id: reportId, report: payload };
  } catch (err) {
    console.error(err);
    return { ok: false, errors: ["The report could not be generated. Please try again."] };
  }
}

export async function deleteGeneratedReport(publicId, userId) {
  const ownerId = Number(userId);
  if (!ownerId) return false;
  try {
    const pool = await getPool();
    const tester = await isTesterUser(ownerId);
    const [result] = await pool.query(
      `DELETE r FROM generated_reports r
       JOIN machines m ON m.id = r.machine_id
       WHERE r.public_id = ? AND r.created_by = ? AND m.created_by = ?
         AND (? = 1 OR m.source <> 'ims')`,
      [String(publicId || ""), ownerId, ownerId, tester ? 1 : 0]
    );
    return Boolean(result.affectedRows);
  } catch (err) {
    console.error(err);
    return false;
  }
}

export async function buildReport(reportId, userId) {
  return getStoredReport(reportId, userId);
}
