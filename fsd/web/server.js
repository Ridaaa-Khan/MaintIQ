import "./load-env.js";
import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { getPool } from "./db.js";
import { logActivity } from "./lib/activity-looger.js";
import {
  consumeAuthAttempt,
  createSession,
  csrfTokensMatch,
  destroyAllSessions,
  destroySession,
  ensureTesterAccount,
  hashPassword,
  isTesterEmail,
  loadSession,
  readCsrfToken,
  validatePasswordChange,
  validateProfileUpdate,
  validateSignin,
  validateSignup,
  verifyPassword,
} from "./auth.js";
import {
  buildReport,
  createGeneratedReport,
  deleteGeneratedReport,
  listReportMachines,
  listReports,
} from "./backend/reports.js";
import { countOpenAlerts, listAlertRecords, summarizeAlerts, updateAlertStatus, ALERT_SEVERITIES, ALERT_STATUSES } from "./backend/alerts.js";
import { explainGraph } from "./backend/explain.js";
import { getDashboardPage } from "./backend/dashboard.js";
import {
  addMachine,
  deleteMachine,
  ensureUserWorkspace,
  getAnalyticsPage,
  getMachineRecord,
  listMachineRecords,
  machineFilterOptions,
  setMachineMonitoring,
  updateMachine,
  updateMachineNotes,
} from "./backend/machines.js";
import { listRunRecords, RUN_STATUSES, summarizeRuns } from "./backend/monitoring.js";
import multer from "multer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));
app.use(express.urlencoded({ extended: false }));
app.use(express.json({ limit: "32kb" }));

const CSRF_SKIP = new Set(["/signin", "/signup", "/tester"]);

app.use(async (req, res, next) => {
  try {
    req.session = await loadSession(req);
    res.locals.csrfToken = req.session?.csrfToken || "";
    return next();
  } catch (err) {
    console.error(err);
    return res.status(500).send("Server error.");
  }
});

app.use((req, res, next) => {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(req.method)) return next();
  if (CSRF_SKIP.has(req.path)) return next();
  if (!req.session) return next();
  if (String(req.headers["content-type"] || "").includes("multipart/form-data")) return next();
  if (csrfTokensMatch(req.session.csrfToken, readCsrfToken(req))) return next();
  const wantsJson = String(req.headers.accept || "").includes("application/json") || req.path.startsWith("/api/");
  if (wantsJson) return res.status(403).json({ ok: false, error: "Session expired. Refresh and try again." });
  return res.status(403).send("Session expired. Refresh the page and try again.");
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter(_req, file, cb) {
    const ext = path.extname(file.originalname || "").toLowerCase();
    if (![".csv", ".zip", ".txt"].includes(ext)) {
      cb(new Error("Only CSV, ZIP, and TXT files are accepted."));
      return;
    }
    cb(null, true);
  },
});

async function loadAccount(userId) {
  const pool = await getPool();
  const [rows] = await pool.query(
    "SELECT id, first_name, last_name, email, phone, created_at FROM users WHERE id = ? LIMIT 1",
    [userId]
  );
  return rows[0] || null;
}

async function requireUser(req, res, next) {
  try {
    if (!req.session?.userId) return res.redirect("/signin");
    const user = await loadAccount(req.session.userId);
    if (!user) {
      await destroySession(req, res);
      return res.redirect("/signin");
    }
    req.user = user;
    res.locals.isTester = isTesterEmail(user.email);
    try {
      await ensureUserWorkspace(user.id);
      res.locals.alertCount = await countOpenAlerts(user.id);
    } catch (countErr) {
      console.error(countErr);
      res.locals.alertCount = 0;
    }
    return next();
  } catch (err) {
    console.error(err);
    return res.status(500).send("Server error. Please try again.");
  }
}

async function requireApiUser(req, res, next) {
  try {
    if (!req.session?.userId) return res.status(401).json({ ok: false, error: "Sign in required." });
    const user = await loadAccount(req.session.userId);
    if (!user) {
      await destroySession(req, res);
      return res.status(401).json({ ok: false, error: "Sign in required." });
    }
    req.user = user;
    res.locals.isTester = isTesterEmail(user.email);
    try {
      await ensureUserWorkspace(user.id);
    } catch (workspaceErr) {
      console.error(workspaceErr);
    }
    return next();
  } catch (err) {
    console.error(err);
    return res.status(500).json({ ok: false, error: "Server error." });
  }
}

app.get("/", (_req, res) => {
  res.render("landing");
});

app.get("/signin", (req, res) => {
  if (req.session?.userId) return res.redirect("/dashboard");
  res.render("signin", { error: null, values: { email: "" } });
});

app.get("/signup", (req, res) => {
  if (req.session?.userId) return res.redirect("/dashboard");
  res.render("signup", {
    errors: [],
    values: { firstName: "", lastName: "", email: "", phone: "" },
  });
});

app.post("/signup", async (req, res) => {
  if (!consumeAuthAttempt(req)) {
    return res.status(429).render("signup", {
      errors: ["Too many attempts. Wait a few minutes and try again."],
      values: {
        firstName: String(req.body?.firstName || ""),
        lastName: String(req.body?.lastName || ""),
        email: String(req.body?.email || ""),
        phone: String(req.body?.phone || ""),
      },
    });
  }
  const checked = validateSignup(req.body);
  if (!checked.ok) {
    return res.status(400).render("signup", {
      errors: checked.errors,
      values: {
        firstName: checked.values.firstName,
        lastName: checked.values.lastName,
        email: checked.values.email,
        phone: checked.values.phone || "",
      },
    });
  }

  try {
    const pool = await getPool();
    const passwordHash = await hashPassword(checked.values.password);
    const [result] = await pool.query(
      `INSERT INTO users (first_name, last_name, email, phone, password_hash)
       VALUES (?, ?, ?, ?, ?)`,
      [
        checked.values.firstName,
        checked.values.lastName,
        checked.values.email,
        checked.values.phone,
        passwordHash,
      ]
    );
    await createSession(req, res, result.insertId);

    await logActivity({
      userId: result.insertId,
      action: "USER_REGISTERED",
      entityType: "user",
      entityId: result.insertId,
      summary: "A new user account was registered.",
    });

    return res.redirect("/dashboard");
  } catch (err) {
    if (err && err.code === "ER_DUP_ENTRY") {
      return res.status(400).render("signup", {
        errors: ["An account with this email already exists."],
        values: {
          firstName: checked.values.firstName,
          lastName: checked.values.lastName,
          email: checked.values.email,
          phone: checked.values.phone || "",
        },
      });
    }
    console.error(err);
    return res.status(500).render("signup", {
      errors: ["An error occurred. Please try again in a little while."],
      values: {
        firstName: checked.values.firstName,
        lastName: checked.values.lastName,
        email: checked.values.email,
        phone: checked.values.phone || "",
      },
    });
  }
});

app.post("/signin", async (req, res) => {
  if (!consumeAuthAttempt(req)) {
    return res.status(429).render("signin", {
      error: "Too many attempts. Wait a few minutes and try again.",
      values: { email: String(req.body?.email || "") },
    });
  }
  const checked = validateSignin(req.body);
  if (!checked.ok) {
    return res.status(400).render("signin", {
      error: checked.errors[0],
      values: { email: checked.values.email },
    });
  }

  try {
    const pool = await getPool();
    const [rows] = await pool.query(
      "SELECT id, password_hash FROM users WHERE email = ? LIMIT 1",
      [checked.values.email]
    );

    const user = rows[0];
    const match = user ? await verifyPassword(checked.values.password, user.password_hash) : false;
    if (!user || !match) {
      return res.status(401).render("signin", {
        error: "Invalid email or password.",
        values: { email: checked.values.email },
      });
    }
    if (!user || !match) {
      await logActivity({
        userId: user?.id ?? null,
        action: "USER_LOGIN_FAILED",
        entityType: "user",
        status: "failed",
        summary: "A sign-in attempt failed.",
      });

      return res.render("signin", {
        error: "Invalid email or password",
      });
    }

    await destroyAllSessions(user.id);
    await createSession(req, res, user.id);
    return res.redirect("/dashboard");
  } catch (err) {
    console.error(err);
    return res.status(500).render("signin", {
      error: "An error occurred. Please try again in a little while.",
      values: { email: checked.values.email },
    });
  }
});

app.post("/tester", async (req, res) => {
  if (!consumeAuthAttempt(req)) {
    return res.status(429).render("landing");
  }
  try {
    const userId = await ensureTesterAccount();
    await destroyAllSessions(userId);
    await createSession(req, res, userId);

    await logActivity({
      userId,
      action: "USER_LOGIN",
      entityType: "user",
      entityId: userId,
      summary: "Tester signed in successfully.",
    });

    return res.redirect("/dashboard");
  } catch (err) {
    console.error(err);
    return res.status(500).render("landing");
  }
});

app.get("/dashboard", requireUser, async (req, res) => {
  const dashboard = await getDashboardPage(req.user.id);
  res.render("dashboard", { user: req.user, dashboard });
});

app.get("/machines", requireUser, async (req, res) => {
  const machines = await listMachineRecords(req.user.id);
  res.render("machines", {
    user: req.user,
    machines,
    filters: machineFilterOptions(machines),
    formErrors: [],
  });
});

app.post("/machines", requireApiUser, (req, res) => {
  upload.single("dataset")(req, res, async (err) => {
    if (err) {
      const message =
        err.code === "LIMIT_FILE_SIZE"
          ? "That file is larger than the 25 MB upload limit."
          : err.message || "Upload failed.";
      return res.status(400).json({ ok: false, errors: [message] });
    }
    try {
      if (!csrfTokensMatch(req.session?.csrfToken, readCsrfToken(req))) {
        return res.status(403).json({ ok: false, errors: ["Session expired. Refresh and try again."] });
      }
      const result = await addMachine(req.body, req.file, req.user.id);

      if (!result.ok) return res.status(400).json(result);

      // Log machine creation
      await logActivity({
        userId: req.user.id,
        action: "MACHINE_CREATED",
        entityType: "machine",
        entityId: result.id,
        summary: "A new machine was added.",
      });

      return res.json({ ok: true, redirect: `/machines/${result.id}` });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ ok: false, errors: ["Could not save the machine."] });
    }
  });
});

app.post("/machines/:id/monitoring", requireApiUser, async (req, res) => {
  const result = await setMachineMonitoring(req.params.id, String(req.body?.status || ""), req.user.id);
  return res.status(result.ok ? 200 : 400).json(result);
});

app.post("/machines/:id/delete", requireApiUser, async (req, res) => {
  try {
    const result = await deleteMachine(req.params.id, req.user.id);
    if (!result.ok) return res.status(404).json(result);
    return res.json(result);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ ok: false, error: "Could not delete the machine." });
  }
});

app.post("/machines/:id/notes", requireUser, async (req, res) => {
  const result = await updateMachineNotes(req.params.id, req.body.notes, req.user.id);
  if (!result.ok) return res.redirect("/machines");
  return res.redirect(`/machines/${req.params.id}`);
});

app.post("/machines/:id", requireUser, async (req, res) => {
  const result = await updateMachine(req.params.id, req.body, req.user.id);
  const back = req.body.returnTo === "list" ? "/machines" : `/machines/${req.params.id}`;
  if (!result.ok) {
    const machine = await getMachineRecord(req.params.id, req.user.id);
    if (!machine || req.body.returnTo === "list") {
      const machines = await listMachineRecords(req.user.id);
      return res.status(400).render("machines", {
        user: req.user,
        machines,
        filters: machineFilterOptions(machines),
        formErrors: result.errors,
      });
    }
    return res.status(400).render("machine-detail", {
      user: req.user,
      machine,
      editErrors: result.errors,
    });
  }
  return res.redirect(back);
});

app.get("/machines/:id", requireUser, async (req, res) => {
  const machine = await getMachineRecord(req.params.id, req.user.id);
  if (!machine) return res.redirect("/machines");
  res.render("machine-detail", { user: req.user, machine, editErrors: [] });
});

app.get("/monitoring", requireUser, async (req, res) => {
  const runs = await listRunRecords(req.user.id);
  const machines = await listMachineRecords(req.user.id);
  res.render("monitoring", {
    user: req.user,
    runs,
    machines,
    statuses: RUN_STATUSES,
    monitoringStats: summarizeRuns(runs),
  });
});

app.get("/alerts", requireUser, async (req, res) => {
  const alerts = await listAlertRecords(req.user.id);
  res.render("alerts", {
    user: req.user,
    alerts,
    severities: ALERT_SEVERITIES,
    statuses: ALERT_STATUSES,
    alertStats: summarizeAlerts(alerts),
  });
});

app.post("/alerts/status", requireApiUser, async (req, res) => {
  const result = await updateAlertStatus(String(req.body.alertKey || ""), String(req.body.status || ""), req.user.id);
  return res.status(result.ok ? 200 : 400).json(result);
});

app.get("/analytics", requireUser, async (req, res) => {
  const page = await getAnalyticsPage(String(req.query.machine || "").trim(), req.user.id);
  res.render("analytics", { user: req.user, machines: page.machines, selectedId: page.selectedId });
});

app.post("/api/analytics/explain", requireApiUser, async (req, res) => {
  try {
    const result = await explainGraph(req.body || {}, req.user.id);
    return res.json(result);
  } catch (err) {
    const status = Number(err.status) || 500;
    if (!err.public && err.code !== "LLM" && err.code !== "NO_KEY") {
      console.error(err);
    } else {
      console.error(err.message);
    }
    return res.status(status).json({
      ok: false,
      error: err.public || err.code === "LLM" || err.code === "NO_KEY"
        ? err.message
        : "Could not explain this graph just now.",
    });
  }
});

app.get("/reports", requireUser, async (req, res) => {
  try {
    const [reports, machines] = await Promise.all([listReports(req.user.id), listReportMachines(req.user.id)]);
    res.render("reports", {
      user: req.user,
      reports,
      machines,
      selectedMachine: String(req.query.machine || ""),
      reportTitle: "",
      notice: String(req.query.deleted || "") === "1" ? "Report deleted." : "",
      error: "",
    });
  } catch (err) {
    console.error(err);
    res.status(500).render("reports", {
      user: req.user,
      reports: [],
      machines: [],
      selectedMachine: "",
      reportTitle: "",
      notice: "",
      error: "Reports could not be loaded. Check the database connection.",
    });
  }
});

app.post("/reports/generate", requireUser, async (req, res) => {
  const generatedBy = [req.user.first_name, req.user.last_name].filter(Boolean).join(" ") || req.user.email;
  const machineId = String(req.body?.machineId || "").trim();
  const title = String(req.body?.title || "").trim();
  const result = await createGeneratedReport({
    machinePublicId: machineId,
    title,
    generatedBy,
    userId: req.user.id,
  });
  if (!result.ok) {
    try {
      const [reports, machines] = await Promise.all([listReports(req.user.id), listReportMachines(req.user.id)]);
      return res.status(400).render("reports", {
        user: req.user,
        reports,
        machines,
        selectedMachine: machineId,
        reportTitle: title.slice(0, 120),
        notice: "",
        error: result.errors.join(" "),
      });
    } catch (err) {
      console.error(err);
      return res.status(500).render("reports", {
        user: req.user,
        reports: [],
        machines: [],
        selectedMachine: machineId,
        reportTitle: title.slice(0, 120),
        notice: "",
        error: "The report could not be generated. Please try again.",
      });
    }
  }
  return res.redirect(`/reports/preview?id=${encodeURIComponent(result.id)}`);
});

app.post("/reports/:id/delete", requireUser, async (req, res) => {
  const ok = await deleteGeneratedReport(req.params.id, req.user.id);
  return res.redirect(ok ? "/reports?deleted=1" : "/reports");
});

async function reportContext(req) {
  const id = String(req.query.id || "");
  if (!id) return null;
  return buildReport(id, req.user.id);
}

function renderView(name, locals) {
  return new Promise((resolve, reject) => {
    app.render(name, locals, (err, html) => {
      if (err) reject(err);
      else resolve(html);
    });
  });
}

app.get("/reports/document", requireUser, async (req, res) => {
  const report = await reportContext(req);
  if (!report) return res.redirect("/reports");
  res.render("report-document", {
    report,
    exportPdf: String(req.query.export || "") === "pdf",
  });
});

app.get("/reports/preview", requireUser, async (req, res) => {
  const report = await reportContext(req);
  if (!report) return res.redirect("/reports");
  res.render("report-preview", { user: req.user, report });
});

app.get("/reports/pdf", requireUser, async (req, res) => {
  const report = await reportContext(req);
  if (!report) return res.redirect("/reports");

  try {
    const puppeteer = (await import("puppeteer")).default;
    const origin = `http://127.0.0.1:${PORT}`;
    const html = await renderView("report-document", { report, exportPdf: true });
    const htmlWithBase = html.replace("<head>", `<head><base href="${origin}/">`);
    const chromePath = [
      process.env.PUPPETEER_EXECUTABLE_PATH,
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ].find((candidate) => candidate && fs.existsSync(candidate));
    const browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
      ...(chromePath ? { executablePath: chromePath } : { channel: "chrome" }),
    });
    try {
      const page = await browser.newPage();
      await page.setContent(htmlWithBase, { waitUntil: "load", timeout: 45000 });
      await page.waitForFunction("window.REPORT_CHARTS_READY === true", { timeout: 20000 }).catch(() => { });
      await page.evaluate(() => (document.fonts ? document.fonts.ready : null));
      await new Promise((resolve) => setTimeout(resolve, 300));
      const pdf = await page.pdf({
        format: "A4",
        printBackground: true,
        displayHeaderFooter: true,
        headerTemplate: `<div style="font-size:9px;color:#5c6570;width:100%;padding:0 16mm;font-family:Segoe UI,sans-serif;display:flex;justify-content:space-between;"><span>Maint-IQ · ${report.id.toUpperCase()}</span><span>${report.generatedAtLabel}</span></div>`,
        footerTemplate: `<div style="font-size:9px;color:#5c6570;width:100%;padding:0 16mm;font-family:Segoe UI,sans-serif;display:flex;justify-content:space-between;"><span>${report.datasetId || report.machine.assetId}${report.runId ? ` · ${report.runId}` : ""}</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`,
        margin: { top: "16mm", bottom: "16mm", left: "14mm", right: "14mm" },
      });
      const rawName = String(report.name || report.id || "report")
        .replace(/[<>:"/\\|?*\x00-\x1f]/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 100);
      const asciiName = (rawName.replace(/[^\x20-\x7E]/g, "").trim() || report.id || "report").replace(/\.+$/, "");
      const utf8Name = encodeURIComponent(`${rawName || report.id}.pdf`);
      res.setHeader("Content-Type", "application/pdf");
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="${asciiName}.pdf"; filename*=UTF-8''${utf8Name}`
      );
      return res.send(Buffer.from(pdf));
    } finally {
      await browser.close();
    }
  } catch (err) {
    console.error("PDF export failed:", err.message);
    const safeId = encodeURIComponent(report.id);
    return res.status(501).type("html").send(
      `<p>PDF export is not available. Open <a href="/reports/document?id=${safeId}">the report</a> and use Print → Save as PDF.</p>`
    );
  }
});

app.get("/settings", requireUser, (req, res) => {
  const rawSection = String(req.query.section || "profile");
  const section =
    rawSection === "password" || rawSection === "security" ? "password" : "profile";
  const editProfile = section === "profile" && String(req.query.edit || "") === "1";
  res.render("settings", {
    user: req.user,
    settingsSection: section,
    editProfile,
    errors: [],
    passwordErrors: [],
    passwordSuccess: String(req.query.passwordUpdated || "") === "1",
    profileSuccess: String(req.query.profileUpdated || "") === "1",
  });
});

app.post("/settings/profile", requireUser, async (req, res) => {
  const checked = validateProfileUpdate(req.body);
  if (!checked.ok) {
    return res.status(400).render("settings", {
      user: {
        ...req.user,
        first_name: checked.values.firstName,
        last_name: checked.values.lastName,
        email: checked.values.email,
        phone: checked.values.phone,
      },
      settingsSection: "profile",
      editProfile: true,
      errors: checked.errors,
      passwordErrors: [],
      passwordSuccess: false,
      profileSuccess: false,
    });
  }

  try {
    const pool = await getPool();
    await pool.query(
      `UPDATE users
       SET first_name = ?, last_name = ?, email = ?, phone = ?
       WHERE id = ?`,
      [
        checked.values.firstName,
        checked.values.lastName,
        checked.values.email,
        checked.values.phone,
        req.user.id,
      ]
    );
    return res.redirect("/settings?profileUpdated=1");
  } catch (err) {
    if (err && err.code === "ER_DUP_ENTRY") {
      return res.status(400).render("settings", {
        user: {
          ...req.user,
          first_name: checked.values.firstName,
          last_name: checked.values.lastName,
          email: checked.values.email,
          phone: checked.values.phone,
        },
        settingsSection: "profile",
        editProfile: true,
        errors: ["An account with this email already exists."],
        passwordErrors: [],
        passwordSuccess: false,
        profileSuccess: false,
      });
    }
    console.error(err);
    return res.status(500).render("settings", {
      user: req.user,
      settingsSection: "profile",
      editProfile: true,
      errors: ["Could not update profile. Please try again."],
      passwordErrors: [],
      passwordSuccess: false,
      profileSuccess: false,
    });
  }
});

app.post("/settings/password", requireUser, async (req, res) => {
  const checked = validatePasswordChange(req.body);
  if (!checked.ok) {
    return res.status(400).render("settings", {
      user: req.user,
      settingsSection: "password",
      editProfile: false,
      errors: [],
      passwordErrors: checked.errors,
      passwordSuccess: false,
      profileSuccess: false,
    });
  }

  try {
    const pool = await getPool();
    const [rows] = await pool.query(
      "SELECT password_hash FROM users WHERE id = ? LIMIT 1",
      [req.user.id]
    );
    if (!rows.length) {
      await destroySession(req, res);
      return res.redirect("/signin");
    }

    const match = await verifyPassword(checked.values.currentPassword, rows[0].password_hash);
    if (!match) {
      return res.status(401).render("settings", {
        user: req.user,
        settingsSection: "password",
        editProfile: false,
        errors: [],
        passwordErrors: ["Wrong password, please try again."],
        passwordSuccess: false,
        profileSuccess: false,
      });
    }

    const passwordHash = await hashPassword(checked.values.newPassword);
    await pool.query("UPDATE users SET password_hash = ? WHERE id = ?", [
      passwordHash,
      req.user.id,
    ]);
    await destroyAllSessions(req.user.id);
    await createSession(req, res, req.user.id);
    return res.redirect("/settings?section=password&passwordUpdated=1");
  } catch (err) {
    console.error(err);
    return res.status(500).render("settings", {
      user: req.user,
      settingsSection: "password",
      editProfile: false,
      errors: [],
      passwordErrors: ["Could not update password. Please try again."],
      passwordSuccess: false,
      profileSuccess: false,
    });
  }
});

app.post("/signout", async (req, res) => {
  const userId = req.session?.userId ?? null;

  await destroySession(req, res);

  await logActivity({
    userId,
    action: "USER_LOGOUT",
    entityType: "user",
    entityId: userId,
    summary: "User signed out.",
  });

  res.redirect("/signin");
});

async function start() {
  try {
    await getPool();
    try {
      await ensureTesterAccount();
    } catch (testerErr) {
      console.error(testerErr);
    }
    console.log("MySQL ready (database: maintiq, table: users)");
  } catch (err) {
    console.error("MySQL connection failed — start MySQL and check config.js / env vars.");
    console.error(err.message);
  }

  app.listen(PORT, () => {
    console.log(`Maint-IQ → http://localhost:${PORT}`);
  });
}

start();
