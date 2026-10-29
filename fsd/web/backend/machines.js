import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { getPool } from "../db.js";
import { isTesterUser } from "../auth.js";
import { getMachine as getScoredMachine, listMachines as listScoredMachines } from "../lib/ims-data.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STORAGE_DIR = path.join(__dirname, "..", "storage", "uploads");
const VALIDATOR = path.join(__dirname, "validate_dataset.py");
const PROJECT_ROOT = path.join(__dirname, "..", "..", "..");
const VENV_PYTHON = path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe");

const ASSET_RE = /^[A-Za-z0-9][A-Za-z0-9-]{1,39}$/;
const ALLOWED_EXT = new Set([".csv", ".zip", ".txt"]);

const catalogReady = new Set();

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

function blankMachine(row) {
  return {
    id: row.public_id,
    dbId: row.id,
    name: row.name,
    assetId: row.asset_id,
    type: row.machine_type,
    location: row.location || "—",
    description: row.description || "",
    health: "Unknown",
    healthClass: "unknown",
    monitoring: "Not configured",
    monitoringClass: "none",
    anomalyScore: "—",
    anomalyScoreNum: null,
    lastData: "—",
    lastIso: "",
    dataSource: "Uploaded file",
    dataset: "",
    rms: "—",
    peak: "—",
    failProb: "Not stored",
    failProbNum: null,
    rulHours: null,
    openAlerts: "0",
    openAlertsNum: 0,
    notes: row.notes || "",
    testId: row.test_id,
    bearingId: row.bearing_id,
    documentedDamage: null,
    nWindows: null,
    nBins: null,
    periodLabel: "",
    start: null,
    end: null,
    series: [],
    thumb: String(row.name || "M")
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0])
      .join("")
      .toUpperCase(),
    scoreSort: 0,
    runId: null,
    source: row.source,
    createdAt: row.created_at || null,
    createdAtLabel: fmtShort(row.created_at),
  };
}

function withMonitoring(machine, row) {
  const stored = row?.monitoring_status;
  if (stored === "Inactive") {
    return { ...machine, monitoring: "Inactive", monitoringClass: "none" };
  }
  if (stored === "Active") {
    return { ...machine, monitoring: "Active", monitoringClass: "active" };
  }
  return machine;
}

function overlayProfile(scored, row) {
  if (!row) return scored;
  return {
    ...scored,
    dbId: row.id,
    name: row.name,
    assetId: row.asset_id,
    type: row.machine_type,
    location: row.location || scored.location,
    description: row.description || scored.description,
    notes: row.notes != null ? row.notes : scored.notes,
    source: row.source,
    createdAt: row.created_at || scored.createdAt || null,
    createdAtLabel: row.created_at ? fmtShort(row.created_at) : scored.createdAtLabel,
  };
}

async function recordActivity(pool, machineId, userId, action, summary) {
  await pool.query(
    `INSERT INTO machine_activity (machine_id, user_id, action, summary) VALUES (?, ?, ?, ?)`,
    [machineId, userId || null, action, summary.slice(0, 500)]
  );
  await pool.query(`UPDATE machines SET last_action = ?, updated_by = ? WHERE id = ?`, [
    action,
    userId || null,
    machineId,
  ]);
}

export async function syncImsMachines(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return;
  if (!(await isTesterUser(ownerId))) return;
  if (catalogReady.has(ownerId)) return;
  const pool = await getPool();
  const scored = listScoredMachines();
  for (const machine of scored) {
    const [existing] = await pool.query(
      "SELECT id FROM machines WHERE public_id = ? AND created_by = ? LIMIT 1",
      [machine.id, ownerId]
    );
    if (existing.length) continue;
    const [result] = await pool.query(
      `INSERT INTO machines
        (public_id, asset_id, name, machine_type, location, description, notes, source, test_id, bearing_id, last_action, created_by, updated_by)
       VALUES (?, ?, ?, ?, ?, ?, ?, 'ims', ?, ?, 'add', ?, ?)`,
      [
        machine.id,
        machine.assetId,
        machine.name,
        machine.type,
        machine.location,
        machine.description,
        machine.notes,
        machine.testId,
        machine.bearingId,
        ownerId,
        ownerId,
      ]
    );
    await recordActivity(
      pool,
      result.insertId,
      ownerId,
      "add",
      `Loaded from NASA IMS scored artifacts (${machine.testId} bearing ${machine.bearingId}).`
    );
  }
  catalogReady.add(ownerId);
}

export async function ensureUserWorkspace(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return;
  if (!(await isTesterUser(ownerId))) return;
  await syncImsMachines(ownerId);
  const { syncImsAlerts } = await import("./alerts.js");
  const { syncImsRuns } = await import("./monitoring.js");
  await syncImsAlerts(ownerId);
  await syncImsRuns(ownerId);
}

async function openAlertCounts(pool, userId) {
  const tester = await isTesterUser(userId);
  const [rows] = await pool.query(
    tester
      ? `SELECT m.public_id AS publicId, COUNT(a.id) AS n
         FROM machines m
         LEFT JOIN alerts a ON a.machine_id = m.id AND a.status <> 'Resolved'
         WHERE m.created_by = ?
         GROUP BY m.public_id`
      : `SELECT m.public_id AS publicId, COUNT(a.id) AS n
         FROM machines m
         LEFT JOIN alerts a ON a.machine_id = m.id AND a.status <> 'Resolved'
         WHERE m.created_by = ? AND m.source <> 'ims'
         GROUP BY m.public_id`,
    [userId]
  );
  return new Map(rows.map((row) => [row.publicId, Number(row.n)]));
}

function withAlertCount(machine, counts) {
  if (!counts.has(machine.id)) return machine;
  const n = counts.get(machine.id);
  return { ...machine, openAlerts: String(n), openAlertsNum: n };
}

export async function listMachineRecords(userId) {
  const ownerId = Number(userId);
  if (!ownerId) return [];
  await ensureUserWorkspace(ownerId);
  const pool = await getPool();
  const tester = await isTesterUser(ownerId);
  const [rows] = await pool.query(
    tester
      ? "SELECT * FROM machines WHERE created_by = ? ORDER BY source DESC, test_id, bearing_id, asset_id"
      : "SELECT * FROM machines WHERE created_by = ? AND source <> 'ims' ORDER BY source DESC, test_id, bearing_id, asset_id",
    [ownerId]
  );
  const counts = await openAlertCounts(pool, ownerId);
  const testOrder = { "1st_test": 1, "2nd_test": 2, "3rd_test": 3 };
  const machines = rows.map((row) => {
    const scored = tester ? getScoredMachine(row.public_id) : null;
    const merged = scored ? overlayProfile(scored, row) : blankMachine(row);
    return withAlertCount(withMonitoring(merged, row), counts);
  });
  machines.sort((a, b) => {
    const ta = testOrder[a.testId] || 99;
    const tb = testOrder[b.testId] || 99;
    if (ta !== tb) return ta - tb;
    return String(a.assetId).localeCompare(String(b.assetId));
  });
  return machines;
}

export async function getMachineRecord(publicId, userId) {
  const machines = await listMachineRecords(userId);
  const machine = machines.find((item) => item.id === publicId) || null;
  if (!machine) return null;
  const pool = await getPool();
  const [files] = await pool.query(
    `SELECT original_name, extension, validation_status, validation_message, row_count, column_count, created_at
     FROM dataset_files WHERE machine_id = ? ORDER BY id DESC LIMIT 1`,
    [machine.dbId]
  );
  const [activity] = await pool.query(
    `SELECT action, summary, created_at FROM machine_activity WHERE machine_id = ? ORDER BY id DESC LIMIT 8`,
    [machine.dbId]
  );
  return {
    ...machine,
    upload: files[0]
      ? {
          name: files[0].original_name,
          extension: files[0].extension,
          status: files[0].validation_status,
          message: files[0].validation_message,
          rows: files[0].row_count,
          columns: files[0].column_count,
          uploadedAt: fmtShort(files[0].created_at),
        }
      : null,
    activity: activity.map((item) => ({
      action: item.action,
      summary: item.summary,
      when: fmtShort(item.created_at),
    })),
  };
}

function cleanText(value, max) {
  return String(value || "").trim().slice(0, max);
}

export function validateMachineInput(body, { requireFile = false, file = null } = {}) {
  const errors = [];
  const assetId = cleanText(body.assetId, 40);
  const name = cleanText(body.name, 80);
  const type = cleanText(body.type, 40);
  const location = cleanText(body.location, 80);
  const description = cleanText(body.description, 500);
  if (!ASSET_RE.test(assetId)) {
    errors.push("Asset ID must be 2–40 letters, numbers, or hyphens and start with a letter or number.");
  }
  if (!name) errors.push("Machine name is required.");
  if (!type) errors.push("Machine type is required.");
  if (!location) errors.push("Location is required.");
  if (requireFile) {
    if (!file) errors.push("Upload a CSV, ZIP, or TXT file.");
    else {
      const ext = path.extname(file.originalname || "").toLowerCase();
      if (!ALLOWED_EXT.has(ext)) errors.push("Only CSV, ZIP, and TXT files are accepted.");
    }
  }
  return { ok: errors.length === 0, errors, values: { assetId, name, type, location, description } };
}

async function assetTaken(pool, assetId, userId, ignorePublicId) {
  const [rows] = await pool.query(
    "SELECT public_id FROM machines WHERE asset_id = ? AND created_by = ? LIMIT 1",
    [assetId, userId]
  );
  if (!rows.length) return false;
  return rows[0].public_id !== ignorePublicId;
}

export async function updateMachine(publicId, body, userId) {
  const checked = validateMachineInput(body);
  if (!checked.ok) return checked;
  const ownerId = Number(userId);
  if (!ownerId) return { ok: false, errors: ["Machine not found."] };
  await syncImsMachines(ownerId);
  const pool = await getPool();
  const [rows] = await pool.query(
    "SELECT * FROM machines WHERE public_id = ? AND created_by = ? LIMIT 1",
    [publicId, ownerId]
  );
  if (!rows.length) return { ok: false, errors: ["Machine not found."] };
  const current = rows[0];
  if (await assetTaken(pool, checked.values.assetId, ownerId, publicId)) {
    return { ok: false, errors: ["This Asset ID is already used."] };
  }
  const changes = [];
  const fields = [
    ["name", "name", current.name],
    ["assetId", "asset_id", current.asset_id],
    ["type", "machine_type", current.machine_type],
    ["location", "location", current.location],
    ["description", "description", current.description],
  ];
  for (const [key, column, previous] of fields) {
    if (checked.values[key] !== (previous || "")) changes.push(column);
  }
  await pool.query(
    `UPDATE machines
     SET name = ?, asset_id = ?, machine_type = ?, location = ?, description = ?, updated_by = ?
     WHERE id = ?`,
    [
      checked.values.name,
      checked.values.assetId,
      checked.values.type,
      checked.values.location,
      checked.values.description,
      userId,
      current.id,
    ]
  );
  if (changes.length) {
    await recordActivity(pool, current.id, userId, "edit", `Edited ${changes.join(", ")}.`);
  }
  return { ok: true, errors: [] };
}

export async function updateMachineNotes(publicId, notes, userId) {
  const text = cleanText(notes, 2000);
  const ownerId = Number(userId);
  if (!ownerId) return { ok: false, errors: ["Machine not found."] };
  await syncImsMachines(ownerId);
  const pool = await getPool();
  const [rows] = await pool.query(
    "SELECT id FROM machines WHERE public_id = ? AND created_by = ? LIMIT 1",
    [publicId, ownerId]
  );
  if (!rows.length) return { ok: false, errors: ["Machine not found."] };
  await pool.query("UPDATE machines SET notes = ?, updated_by = ? WHERE id = ?", [
    text,
    userId,
    rows[0].id,
  ]);
  await recordActivity(pool, rows[0].id, userId, "note", "Updated notes.");
  return { ok: true, errors: [], notes: text };
}

export async function setMachineMonitoring(publicId, status, userId) {
  const next = status === "Inactive" ? "Inactive" : status === "Active" ? "Active" : "";
  if (!next) return { ok: false, error: "Invalid monitoring status." };
  const ownerId = Number(userId);
  if (!ownerId) return { ok: false, error: "Machine not found." };
  const pool = await getPool();
  const tester = await isTesterUser(ownerId);
  const [result] = await pool.query(
    tester
      ? "UPDATE machines SET monitoring_status = ?, updated_by = ? WHERE public_id = ? AND created_by = ?"
      : "UPDATE machines SET monitoring_status = ?, updated_by = ? WHERE public_id = ? AND created_by = ? AND source <> 'ims'",
    [next, ownerId, publicId, ownerId]
  );
  if (!result.affectedRows) return { ok: false, error: "Machine not found." };
  return {
    ok: true,
    monitoring: next,
    monitoringClass: next === "Active" ? "active" : "none",
  };
}

function pythonCommand() {
  if (fs.existsSync(VENV_PYTHON)) return VENV_PYTHON;
  return "python";
}

export function validateDatasetFile(filePath) {
  return new Promise((resolve) => {
    const child = spawn(pythonCommand(), [VALIDATOR, filePath], {
      cwd: PROJECT_ROOT,
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", () => {
      resolve({
        ok: false,
        status: "rejected",
        message: "Dataset validator could not be started.",
        rows: null,
        columns: null,
      });
    });
    child.on("close", () => {
      try {
        const parsed = JSON.parse(stdout.trim().split(/\r?\n/).pop());
        resolve(parsed);
      } catch {
        resolve({
          ok: false,
          status: "rejected",
          message: stderr.trim().slice(0, 400) || "The file could not be parsed.",
          rows: null,
          columns: null,
        });
      }
    });
  });
}

export async function addMachine(body, file, userId) {
  const checked = validateMachineInput(body, { requireFile: true, file });
  if (!checked.ok) return checked;
  const ownerId = Number(userId);
  if (!ownerId) return { ok: false, errors: ["Sign in required."] };
  await syncImsMachines(ownerId);
  const pool = await getPool();
  if (await assetTaken(pool, checked.values.assetId, ownerId, null)) {
    return { ok: false, errors: ["This Asset ID is already used."] };
  }

  const ext = path.extname(file.originalname).toLowerCase();
  const publicId = `upl-${Date.now().toString(36)}`;
  const dir = path.join(STORAGE_DIR, publicId);
  fs.mkdirSync(dir, { recursive: true });
  const storedName = `dataset${ext}`;
  const storedPath = path.join(dir, storedName);
  fs.writeFileSync(storedPath, file.buffer);

  const report = await validateDatasetFile(storedPath);
  const validationStatus = report.ok ? "parsed" : "rejected";
  const runStatus = report.ok ? "Completed" : "Failed";

  const [created] = await pool.query(
    `INSERT INTO machines
      (public_id, asset_id, name, machine_type, location, description, notes, source, last_action, created_by, updated_by)
     VALUES (?, ?, ?, ?, ?, ?, ?, 'upload', 'add', ?, ?)`,
    [
      publicId,
      checked.values.assetId,
      checked.values.name,
      checked.values.type,
      checked.values.location,
      checked.values.description,
      "",
      userId,
      userId,
    ]
  );

  const [dataset] = await pool.query(
    `INSERT INTO dataset_files
      (machine_id, owner_user_id, original_name, stored_path, extension, byte_size, validation_status, validation_message, row_count, column_count)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      created.insertId,
      userId,
      path.basename(file.originalname).slice(0, 255),
      storedPath,
      ext.slice(1),
      file.size,
      validationStatus,
      String(report.message || "").slice(0, 1000),
      report.rows,
      report.columns,
    ]
  );

  const windowLabel =
    report.rows == null
      ? String(report.message || "").slice(0, 255)
      : `${report.rows} parsed rows${report.columns ? ` · ${report.columns} columns` : ""}`;

  await pool.query(
    `INSERT INTO analysis_runs
      (machine_id, dataset_file_id, run_code, status, stage, window_label, progress, dataset_label, scope, started_at, updated_at, error_message)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW(), NOW(), ?)`,
    [
      created.insertId,
      dataset.insertId,
      `UPLOAD-${publicId}`,
      runStatus,
      report.ok ? "File parsed" : "Validation failed",
      windowLabel,
      report.ok ? 100 : 0,
      path.basename(file.originalname).slice(0, 255),
      "history",
      report.ok ? null : String(report.message || "").slice(0, 1000),
    ]
  );

  await recordActivity(
    pool,
    created.insertId,
    userId,
    "add",
    `Added machine ${checked.values.assetId} with ${path.basename(file.originalname)} (${validationStatus}).`
  );

  return { ok: true, errors: [], id: publicId, validation: report };
}

export function machineFilterOptions(machines) {
  return {
    health: [...new Set(machines.map((m) => m.health))],
    types: [...new Set(machines.map((m) => m.type))],
    monitoring: [...new Set(machines.map((m) => m.monitoring))],
  };
}

export async function getAnalyticsPage(requestedId, userId) {
  const machines = (await listMachineRecords(userId))
    .filter((machine) => machine.series?.length)
    .map((machine) => ({
      id: machine.id,
      name: machine.name,
      assetId: machine.assetId,
    }));
  const selectedId = machines.some((machine) => machine.id === requestedId)
    ? requestedId
    : machines[0]?.id || "";
  return { machines, selectedId };
}
