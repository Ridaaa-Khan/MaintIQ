import fs from "fs";
import path from "path";
import { randomBytes } from "crypto";
import { fileURLToPath } from "url";
import { getPool } from "../fsd/web/db.js";
import { listReports } from "../fsd/web/backend/reports.js";
import { getImsMeta } from "../fsd/web/lib/ims-data.js";
import { getMachineRecord, listMachineRecords } from "../fsd/web/backend/machines.js";
import { listAlertRecords } from "../fsd/web/backend/alerts.js";
import { listRunRecords } from "../fsd/web/backend/monitoring.js";
import { AVAILABLE_ROUTES, SYSTEM_PROMPT } from "./prompt.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KNOWLEDGE_DIR = path.join(__dirname, "knowledge");
const SIMPLE_FOLLOW_UP = "Explain it in a simpler way";
const FRIENDLY_FAIL = "I'm having trouble right now. Please try again in a moment.";
const CHITCHAT_FALLBACK =
  "Hi — I'm the MaintIQ Assistant. I can explain our models, look up your assets, or take you to a page. What do you need?";

/** Local LLM settings (not wired through fsd while the assistant feature is deferred). */
const llmConfig = {
  provider: process.env.LLM_PROVIDER || "google",
  apiKey: String(process.env.LLM_API_KEY || "").trim(),
  model: process.env.LLM_MODEL || "gemini-3.8-flash",
  baseUrl: process.env.LLM_BASE_URL || "https://generativelanguage.googleapis.com/v1beta",
};

const HISTORY_TURNS = 6;

const TOOLS = [
  {
    type: "function",
    function: {
      name: "find_asset",
      description: "Find assets by name, asset ID, or public id. Case-insensitive; returns matches or not_found.",
      parameters: {
        type: "object",
        properties: { name: { type: "string" } },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_asset_details",
      description: "Get one asset record by public id, asset ID, or name.",
      parameters: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_alerts",
      description: "Get recent alerts for one asset or the whole permitted fleet.",
      parameters: {
        type: "object",
        properties: {
          asset_id: { type: "string" },
          limit: { type: "number" },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_prediction",
      description: "Get stored vibration scores, anomaly, failure probability, and RUL for one asset.",
      parameters: {
        type: "object",
        properties: { asset_id: { type: "string" } },
        required: ["asset_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_monitoring_runs",
      description: "Get analysis-run status, optionally for one asset.",
      parameters: { type: "object", properties: { asset_id: { type: "string" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "search_reports",
      description: "Find stored reports by title, machine, or id.",
      parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_model_information",
      description: "Retrieve MaintIQ documentation about Isolation Forest, XGBoost, RUL, features, SHAP, IMS, or pages.",
      parameters: {
        type: "object",
        properties: { topic: { type: "string" } },
        required: ["topic"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_fleet_summary",
      description: "Fleet-level counts: total machines, health breakdown, open alerts, and which assets have open alerts.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "request_navigation",
      description: "Ask the backend to open a permitted page. The backend builds the URL. Do not invent routes.",
      parameters: {
        type: "object",
        properties: {
          page: { type: "string", enum: Object.keys(AVAILABLE_ROUTES) },
          asset_id: { type: "string" },
          report_id: { type: "string" },
        },
        required: ["page"],
      },
    },
  },
];

let knowledgeCache = null;

function loadKnowledge() {
  if (knowledgeCache) return knowledgeCache;
  if (!fs.existsSync(KNOWLEDGE_DIR)) {
    knowledgeCache = [];
    return knowledgeCache;
  }
  knowledgeCache = fs
    .readdirSync(KNOWLEDGE_DIR)
    .filter((name) => name.endsWith(".md"))
    .map((name) => {
      const text = fs.readFileSync(path.join(KNOWLEDGE_DIR, name), "utf8");
      return { id: name.replace(/\.md$/, ""), text };
    });
  return knowledgeCache;
}

const KNOWLEDGE_ALIASES = {
  isolation: ["isolation-forest"],
  forest: ["isolation-forest"],
  anomaly: ["isolation-forest", "model-outputs"],
  xgboost: ["failure-classifier", "rul-regressor"],
  classifier: ["failure-classifier"],
  failure: ["failure-classifier", "model-outputs"],
  rul: ["rul-regressor"],
  remaining: ["rul-regressor"],
  shap: ["shap"],
  choice: ["model-choices"],
  why: ["model-choices", "isolation-forest", "failure-classifier"],
  versus: ["model-choices"],
  alternative: ["model-choices"],
  vibration: ["vibration-features"],
  rms: ["vibration-features", "glossary"],
  kurtosis: ["vibration-features", "glossary"],
  fft: ["vibration-features"],
  ims: ["nasa-ims"],
  nasa: ["nasa-ims"],
  evaluation: ["evaluation-and-limitations"],
  dashboard: ["maintiq-app"],
  analytics: ["maintiq-app"],
  project: ["maintiq-app"],
  maintiq: ["maintiq-app"],
  glossary: ["glossary"],
  alert: ["glossary", "maintiq-app"],
  lstm: ["model-choices"],
  svm: ["model-choices"],
  autoencoder: ["model-choices"],
};

const KNOWLEDGE_STOP = new Set([
  "the",
  "and",
  "for",
  "are",
  "you",
  "what",
  "this",
  "that",
  "about",
  "how",
  "does",
  "did",
  "not",
  "with",
  "using",
  "from",
  "into",
]);

function queryTerms(query) {
  return String(query || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 2 && !KNOWLEDGE_STOP.has(t));
}

function retrieveKnowledge(query, limit = 2) {
  const docs = loadKnowledge();
  const terms = queryTerms(query);
  if (!terms.length) return [];
  const df = (term) => docs.filter((d) => `${d.id} ${d.text}`.toLowerCase().includes(term)).length;
  const scored = docs.map((doc) => {
    const hay = `${doc.id} ${doc.text}`.toLowerCase();
    let score = 0;
    for (const term of terms) {
      const idf = Math.log(1 + docs.length / Math.max(1, df(term)));
      if (hay.includes(term)) score += idf;
      if ((KNOWLEDGE_ALIASES[term] || []).includes(doc.id)) score += 4;
      if (doc.id.includes(term)) score += 3;
    }
    if (/\b(why not|versus|instead of|and not)\b/i.test(String(query || "")) && doc.id === "model-choices") {
      score += 8;
    }
    return { ...doc, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((doc) => doc.score >= 1).slice(0, limit);
}

function pickKnowledgeExcerpt(doc, query) {
  const terms = queryTerms(query);
  const blocks = String(doc?.text || "").split(/\n(?=##\s)|(?:\n\s*\n)/);
  let best = { score: -1, text: "" };
  for (const block of blocks) {
    const body = block.replace(/^#+\s+/, "").trim();
    if (body.length < 50) continue;
    const hay = block.toLowerCase();
    let score = body.length / 400;
    for (const term of terms) {
      if (hay.includes(term)) score += 3;
    }
    if (score > best.score) best = { score, text: body };
  }
  return (best.text || String(doc?.text || "").replace(/^#+\s+/, "")).trim().slice(0, 900);
}

function publicId() {
  return `chat-${randomBytes(8).toString("hex")}`;
}

function compactMachine(machine) {
  if (!machine) return null;
  return {
    machineId: machine.id,
    assetId: machine.assetId,
    name: machine.name,
    type: machine.type,
    location: machine.location,
    health: machine.health,
    monitoring: machine.monitoring,
    documentedDamage: machine.documentedDamage || null,
    openAlerts: machine.openAlertsNum ?? Number(machine.openAlerts) ?? 0,
  };
}

function compactPrediction(machine) {
  const latest = machine.series?.length ? machine.series[machine.series.length - 1] : null;
  const meta = getImsMeta();
  return {
    machineId: machine.id,
    assetId: machine.assetId,
    name: machine.name,
    health: machine.health,
    classifierThreshold: meta.classifierThreshold,
    ifBaselineMean: meta.ifBaselineMean,
    latest: {
      timestamp: latest?.t || machine.lastIso || null,
      rms: latest?.rms ?? (machine.rms === "—" ? null : machine.rms),
      peak: latest?.peak ?? null,
      kurtosis: latest?.kurtosis ?? null,
      anomalyScore: latest?.anomaly ?? machine.anomalyScoreNum ?? null,
      anomalyFlag: latest?.flag ?? null,
      failureProbability: latest?.failProb ?? machine.failProbNum ?? null,
      predictedRulHours: latest?.rul ?? machine.rulHours ?? null,
    },
  };
}

function normalizeQuery(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function uniqueAlerts(alerts) {
  const seen = new Set();
  return (alerts || []).filter((alert) => {
    const key = alert.id || `${alert.title}|${alert.created}|${alert.severity}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

async function findAssets(query) {
  const q = normalizeQuery(query);
  const machines = await listMachineRecords();
  if (!q) return machines;
  const parts = q.split(/\s+/).filter((part) => part.length > 1);
  return machines
    .map((machine) => {
      const hay = normalizeQuery(
        [machine.id, machine.assetId, machine.name, machine.location, machine.testId, `bearing ${machine.bearingId}`]
          .filter(Boolean)
          .join(" ")
      );
      let score = 0;
      if (normalizeQuery(machine.assetId) === q || normalizeQuery(machine.id) === q || normalizeQuery(machine.name) === q) {
        score = 100;
      } else if (hay.includes(q)) score = 80;
      else if (parts.length && parts.every((part) => hay.includes(part))) score = 60;
      else if (parts.some((part) => hay.includes(part) && part.length > 2)) score = 30;
      return { machine, score };
    })
    .filter((row) => row.score >= 30)
    .sort((a, b) => b.score - a.score)
    .map((row) => row.machine);
}

async function resolveAsset(query) {
  const matches = await findAssets(query);
  if (matches.length === 1) return { found: true, machine: matches[0] };
  if (!matches.length) return { found: false, status: "not_found", query };
  const q = normalizeQuery(query);
  const exact = matches.filter(
    (machine) =>
      normalizeQuery(machine.name) === q ||
      normalizeQuery(machine.assetId) === q ||
      normalizeQuery(machine.id) === q
  );
  if (exact.length === 1) return { found: true, machine: exact[0] };
  return { found: false, status: "ambiguous", matches: matches.slice(0, 8).map(compactMachine) };
}

function buildRoute(page, { machineId, reportId } = {}) {
  const spec = AVAILABLE_ROUTES[page];
  if (!spec) return { ok: false, error: "That page is not permitted." };
  if (page === "machine-details") {
    if (!machineId) return { ok: false, error: "machine-details needs an asset." };
    return {
      ok: true,
      action: {
        type: "navigate",
        route: page,
        href: `/machines/${encodeURIComponent(machineId)}`,
        label: spec.label,
        machineId,
      },
    };
  }
  if (page === "analytics") {
    const href = machineId ? `/analytics?machine=${encodeURIComponent(machineId)}` : "/analytics";
    return {
      ok: true,
      action: { type: "navigate", route: page, href, label: spec.label, machineId: machineId || null },
    };
  }
  if (page === "report-preview" || (page === "reports" && reportId)) {
    const href = reportId ? `/reports/preview?id=${encodeURIComponent(reportId)}` : "/reports";
    return {
      ok: true,
      action: { type: "navigate", route: reportId ? "report-preview" : "reports", href, label: spec.label, reportId: reportId || null },
    };
  }
  return { ok: true, action: { type: "navigate", route: page, href: spec.path, label: spec.label } };
}

async function executeTool(name, rawArgs) {
  const args = rawArgs && typeof rawArgs === "object" ? rawArgs : {};
  const query = String(args.name || args.id || args.asset_id || args.query || args.topic || "").trim();

  if (name === "find_asset") {
    const matches = await findAssets(query);
    if (!matches.length) return { status: "not_found", query };
    return { status: "ok", count: matches.length, assets: matches.slice(0, 10).map(compactMachine) };
  }

  if (name === "get_asset_details") {
    const found = await resolveAsset(query);
    if (!found.found) return found;
    const full = await getMachineRecord(found.machine.id);
    return { status: "ok", asset: compactMachine(full || found.machine) };
  }

  if (name === "get_prediction") {
    const found = await resolveAsset(query);
    if (!found.found) return found;
    const machine = (await getMachineRecord(found.machine.id)) || found.machine;
    if (!machine.series?.length && machine.source === "upload") {
      return { status: "ok", asset: compactMachine(machine), note: "No stored model scores for this upload yet." };
    }
    return { status: "ok", prediction: compactPrediction(machine) };
  }

  if (name === "get_alerts") {
    const limit = Math.min(20, Math.max(1, Number(args.limit) || 8));
    let alerts = await listAlertRecords();
    if (query) {
      const found = await findAssets(query);
      if (!found.length) return { status: "not_found", query, alerts: [] };
      const ids = new Set(found.map((machine) => machine.id));
      const assets = new Set(found.map((machine) => String(machine.assetId).toLowerCase()));
      alerts = alerts.filter(
        (alert) => ids.has(alert.machineId) || assets.has(String(alert.assetId).toLowerCase())
      );
    }
    const mapped = uniqueAlerts(
      alerts.map((alert) => ({
        id: alert.id,
        severity: alert.severity,
        status: alert.status,
        type: alert.type,
        title: alert.title,
        machine: alert.machine,
        assetId: alert.assetId,
        created: alert.created,
        value: alert.value,
        threshold: alert.threshold,
        why: alert.why,
      }))
    );
    return { status: "ok", count: mapped.length, alerts: mapped.slice(0, limit) };
  }

  if (name === "get_monitoring_runs") {
    let runs = await listRunRecords();
    if (query) {
      const found = await findAssets(query);
      const ids = new Set(found.map((machine) => machine.id));
      runs = runs.filter((run) => ids.has(run.machineId));
    }
    return {
      status: "ok",
      count: runs.length,
      runs: runs.slice(0, 10).map((run) => ({
        runId: run.id,
        machineId: run.machineId,
        name: run.name,
        assetId: run.assetId,
        status: run.status,
        health: run.health,
        anomalyScore: run.anomalyScore,
        failureProbability: run.failProb,
        predictedRulHours: run.rulHours,
      })),
    };
  }

  if (name === "get_fleet_summary") {
    const machines = await listMachineRecords();
    const open = (await listAlertRecords()).filter((a) => a.status !== "Resolved");
    const byHealth = {};
    machines.forEach((m) => {
      byHealth[m.health] = (byHealth[m.health] || 0) + 1;
    });
    const perAsset = {};
    open.forEach((a) => {
      perAsset[a.assetId] = (perAsset[a.assetId] || 0) + 1;
    });
    return {
      status: "ok",
      machineCount: machines.length,
      byHealth,
      openAlertCount: open.length,
      assetsWithOpenAlerts: Object.entries(perAsset)
        .map(([assetId, count]) => ({ assetId, openAlerts: count }))
        .slice(0, 20),
    };
  }

  if (name === "get_model_information") {
    const docs = retrieveKnowledge(query || "model", 2);
    if (!docs.length) return { status: "not_found", message: "No documentation matched that topic." };
    return { status: "ok", sections: docs.map((doc) => ({ id: doc.id, excerpt: doc.text.slice(0, 2800) })) };
  }

  if (name === "search_reports") {
    const q = query.toLowerCase();
    const reports = (await listReports()).filter((report) =>
      `${report.id} ${report.name} ${report.machineName} ${report.assetId} ${report.machineKey}`.toLowerCase().includes(q)
    );
    if (!reports.length) return { status: "not_found", query };
    return {
      status: "ok",
      reports: reports.map((report) => ({
        id: report.id,
        name: report.name,
        machineId: report.machineKey,
        machineName: report.machineName,
        assetId: report.assetId,
      })),
    };
  }

  if (name === "request_navigation") {
    const page = String(args.page || "");
    let machineId = String(args.asset_id || args.machineId || "").trim();
    const reportId = String(args.report_id || args.reportId || "").trim();
    if (machineId) {
      const found = await resolveAsset(machineId);
      if (!found.found) return found;
      machineId = found.machine.id;
    }
    const resolved = buildRoute(page, { machineId, reportId });
    if (resolved.ok && machineId) {
      const machine = await getMachineRecord(machineId);
      if (!machine) return { status: "not_found", query: machineId };
      resolved.action.label = `Open ${machine.name}`;
    }
    return resolved.ok ? { status: "ok", ...resolved } : resolved;
  }

  return { status: "error", error: "Unknown tool." };
}

function isSimpleGreeting(text) {
  return /^(hi|hello|hey|yo|hiya|howdy|sup|thanks|thank you|cheers|good (morning|afternoon|evening))([!.\s]*)$/i.test(
    String(text || "").trim()
  );
}

function classifyIntent(text) {
  const q = String(text || "").trim();
  if (
    /^(hi|hello|hey|yo|hiya|howdy|sup|thanks|thank you|cheers|good (morning|afternoon|evening)|how are you|how's it going|what's up)[!.?\s]*$/i.test(
      q
    )
  ) {
    return "chitchat";
  }
  return "general";
}

function guessIntentForFallback(text) {
  const q = String(text || "").trim();
  if (/\b(show me|open|take me|go to|navigate|bring me)\b/i.test(q) && !/\b(what is|why|how does|explain)\b/i.test(q)) {
    return "navigate";
  }
  if (
    /\b(asset\s*id|where is|locat(ed|ion)|health|alerts?|anomaly score|fail(ure)?\s*prob|rul|how many|registered|brg-|t\d-b\d|ims\s+\d|ims bearing|latest)\b/i.test(
      q
    )
  ) {
    return "data";
  }
  if (
    /\b(isolation\s*forest|xgboost|classifier|shap|vibration|predictive maintenance|remaining useful|kurtosis|fft|rms|why (did|do|are)|what is|how does|nasa ims|maintiq)\b/i.test(
      q
    )
  ) {
    return "explain";
  }
  if (/\b(this project|the (app|platform)|who are you|what can you)\b/i.test(q)) return "explain";
  if (extractAssetHint(q) || /^(xyz-\d+|pump\s+\d+|motor\s+\d+|bearing\s+\d+)$/i.test(q)) return "data";
  return "chitchat";
}

function extractAssetHint(text) {
  const match =
    String(text || "").match(/\b(BRG-\d+|T\d-B\d|brg-\d+|t\d-b\d)\b/i) ||
    String(text || "").match(/\b((?:IMS\s+)?(?:\d+(?:st|nd|rd|th)_+test\s+)?Bearing\s+\d+)\b/i);
  return match ? match[1] : "";
}

function lastAssetFromHistory(history) {
  const rows = Array.isArray(history) ? history : [];
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    const found = extractAssetHint(rows[i]?.content || "");
    if (found) return found;
  }
  return "";
}

function parseFollowUps(text) {
  const source = String(text || "");
  const split = source.split(/\nFOLLOW_UPS:\s*\n/i);
  const reply = (split[0] || "").trim();
  const followUps = [];
  if (split[1]) {
    for (const line of split[1].split("\n")) {
      const item = line.replace(/^\s*[-*]\s*/, "").trim();
      if (item && !followUps.includes(item)) followUps.push(item.slice(0, 120));
    }
  }
  while (followUps.length < 2) {
    const next = ["Which machines have active alerts?", "What is Isolation Forest?"].find(
      (item) => !followUps.includes(item)
    );
    if (!next) break;
    followUps.push(next);
  }
  return { reply: reply || source.trim(), followUps: followUps.slice(0, 3) };
}

function stripAction(text) {
  const source = String(text || "");
  const match = source.match(/<action>\s*([\s\S]*?)\s*<\/action>/i);
  let action = null;
  if (match) {
    try {
      action = JSON.parse(match[1]);
    } catch {
      action = null;
    }
  }
  return { text: source.replace(/<action>[\s\S]*?<\/action>/gi, "").trim(), action };
}

function wrapBlock(tag, value) {
  if (value == null || value === "" || (Array.isArray(value) && !value.length)) return "";
  const body = typeof value === "string" ? value : JSON.stringify(value);
  return `\n<${tag}>\n${body}\n</${tag}>\n`;
}

function geminiDeclarations() {
  return TOOLS.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }));
}

function toGeminiPayload(messages, { useTools = false, temperature = 0.4 } = {}) {
  let systemText = "";
  const contents = [];
  const nameByCallId = new Map();

  for (const message of messages) {
    if (message.role === "system") {
      systemText += (systemText ? "\n\n" : "") + String(message.content || "");
      continue;
    }
    if (message.role === "tool") {
      let responseObj = {};
      try {
        responseObj = JSON.parse(message.content || "{}");
      } catch {
        responseObj = { result: message.content || "" };
      }
      const name = message.name || nameByCallId.get(message.tool_call_id) || "tool";
      const part = { functionResponse: { name, response: responseObj } };
      const last = contents[contents.length - 1];
      if (last?.role === "user" && last.parts.every((p) => p.functionResponse)) last.parts.push(part);
      else contents.push({ role: "user", parts: [part] });
      continue;
    }
    if (message.role === "assistant" && message.tool_calls?.length) {
      for (const call of message.tool_calls) nameByCallId.set(call.id, call.function.name);
      if (message.rawParts?.length) {
        contents.push({ role: "model", parts: message.rawParts });
        continue;
      }
      const parts = [];
      if (message.content) parts.push({ text: String(message.content) });
      for (const call of message.tool_calls) {
        let args = {};
        try {
          args = JSON.parse(call.function.arguments || "{}");
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
      contents.push({ role: "model", parts });
      continue;
    }
    if (message.role === "assistant") {
      contents.push({ role: "model", parts: [{ text: String(message.content || "") }] });
      continue;
    }
    contents.push({ role: "user", parts: [{ text: String(message.content || "") }] });
  }

  const payload = {
    systemInstruction: systemText ? { parts: [{ text: systemText }] } : undefined,
    contents: contents.length ? contents : [{ role: "user", parts: [{ text: "Hello" }] }],
    generationConfig: {
      temperature,
      maxOutputTokens: 4096,
      thinkingConfig: { thinkingLevel: "low" },
    },
  };
  if (useTools) payload.tools = [{ functionDeclarations: geminiDeclarations() }];
  return payload;
}

function fromGeminiResponse(payload) {
  if (!payload.candidates?.length) {
    const blocked = payload.promptFeedback?.blockReason || payload.error?.message;
    const error = new Error(blocked || "Empty model response.");
    error.code = "LLM";
    error.retryable = false;
    throw error;
  }
  const candidate = payload.candidates[0];
  const parts = candidate?.content?.parts || [];
  const textParts = [];
  const toolCalls = [];
  for (const part of parts) {
    if (part.text && !part.thought) textParts.push(part.text);
    if (part.functionCall?.name) {
      toolCalls.push({
        id: `call_${randomBytes(6).toString("hex")}`,
        type: "function",
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args || {}),
        },
      });
    }
  }
  if (candidate.finishReason && candidate.finishReason !== "STOP") {
    console.warn("Gemini finishReason:", candidate.finishReason);
  }
  return {
    content: textParts.join("\n").trim(),
    tool_calls: toolCalls.length ? toolCalls : undefined,
    rawParts: parts,
  };
}

async function callGemini(messages, options = {}) {
  const model = llmConfig.model || "gemini-3.8-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const send = async () => {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": llmConfig.apiKey,
      },
      body: JSON.stringify(
        toGeminiPayload(messages, {
          useTools: options.useTools === true,
          temperature: options.temperature ?? 0.4,
        })
      ),
      signal: AbortSignal.timeout(options.timeoutMs || 25000),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload.error?.message || `Gemini request failed (${response.status}).`;
      console.error("Gemini error", response.status, JSON.stringify(payload.error || payload).slice(0, 800));
      const error = new Error(message);
      error.code = "LLM";
      error.status = response.status;
      error.retryable =
        response.status === 503 ||
        (response.status === 429 && !/per day|daily quota|free tier/i.test(message));
      throw error;
    }
    return fromGeminiResponse(payload);
  };

  try {
    return await send();
  } catch (err) {
    if (!err.retryable) throw err;
    console.warn("Gemini retry after", err.status || err.message);
    await new Promise((resolve) => setTimeout(resolve, 800));
    return send();
  }
}

async function callLlm(messages, options = {}) {
  if (!llmConfig.apiKey) {
    const error = new Error("LLM_API_KEY is empty.");
    error.code = "NO_KEY";
    throw error;
  }
  return callGemini(messages, options);
}

function mapRow(row) {
  return {
    id: row.public_id,
    title: row.title,
    pinned: Boolean(row.pinned),
    updatedAt: new Date(row.updated_at).getTime(),
    createdAt: new Date(row.created_at).getTime(),
  };
}

export async function listSessions(userId) {
  const pool = await getPool();
  const [rows] = await pool.query(
    `SELECT public_id, title, pinned, created_at, updated_at
     FROM chat_sessions WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC`,
    [userId]
  );
  return rows.map(mapRow);
}

export async function createSession(userId, title = "New chat") {
  const pool = await getPool();
  const id = publicId();
  await pool.query(`INSERT INTO chat_sessions (public_id, user_id, title, pinned) VALUES (?, ?, ?, 0)`, [
    id,
    userId,
    String(title).slice(0, 80) || "New chat",
  ]);
  const [rows] = await pool.query(
    "SELECT public_id, title, pinned, created_at, updated_at FROM chat_sessions WHERE public_id = ? AND user_id = ?",
    [id, userId]
  );
  return { ...mapRow(rows[0]), messages: [] };
}

async function ownedSession(pool, userId, sessionPublicId) {
  const [rows] = await pool.query("SELECT * FROM chat_sessions WHERE public_id = ? AND user_id = ? LIMIT 1", [
    sessionPublicId,
    userId,
  ]);
  return rows[0] || null;
}

export async function getSession(userId, sessionPublicId) {
  const pool = await getPool();
  const session = await ownedSession(pool, userId, sessionPublicId);
  if (!session) return null;
  const [messages] = await pool.query(
    `SELECT role, content, follow_ups, navigation_json, created_at
     FROM chat_messages WHERE session_id = ? AND role IN ('user','assistant') ORDER BY id ASC`,
    [session.id]
  );
  return {
    ...mapRow(session),
    messages: messages.map((row) => ({
      role: row.role,
      text: row.content,
      followUps: row.follow_ups ? JSON.parse(row.follow_ups) : [],
      navigation: row.navigation_json ? JSON.parse(row.navigation_json) : null,
    })),
  };
}

export async function updateSession(userId, sessionPublicId, patch) {
  const pool = await getPool();
  const session = await ownedSession(pool, userId, sessionPublicId);
  if (!session) return null;
  const title = patch.title != null ? String(patch.title).trim().slice(0, 80) || "New chat" : session.title;
  const pinned = patch.pinned == null ? session.pinned : patch.pinned ? 1 : 0;
  await pool.query("UPDATE chat_sessions SET title = ?, pinned = ? WHERE id = ?", [title, pinned, session.id]);
  const [rows] = await pool.query(
    "SELECT public_id, title, pinned, created_at, updated_at FROM chat_sessions WHERE id = ?",
    [session.id]
  );
  return mapRow(rows[0]);
}

export async function deleteSession(userId, sessionPublicId) {
  const pool = await getPool();
  const session = await ownedSession(pool, userId, sessionPublicId);
  if (!session) return false;
  await pool.query("DELETE FROM chat_sessions WHERE id = ?", [session.id]);
  return true;
}

function titleFromPrompt(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return "New chat";
  return clean.slice(0, 48) + (clean.length > 48 ? "…" : "");
}

function knowledgeBlock(question) {
  const docs = retrieveKnowledge(question, 2);
  if (!docs.length) return "";
  return wrapBlock(
    "knowledge_context",
    docs.map((doc) => `### ${doc.id}\n${doc.text.slice(0, 2200)}`).join("\n\n")
  );
}

async function runToolLoop(llmMessages, { temperature, pool, sessionId }) {
  let navigation = null;
  let replyText = "";
  for (let step = 0; step < 4; step += 1) {
    const messageOut = await callLlm(llmMessages, { useTools: step === 0 || true, temperature });
    if (messageOut.tool_calls?.length) {
      llmMessages.push({
        role: "assistant",
        content: messageOut.content || "",
        tool_calls: messageOut.tool_calls,
        rawParts: messageOut.rawParts,
      });
      for (const call of messageOut.tool_calls) {
        let args = {};
        try {
          args = JSON.parse(call.function.arguments || "{}");
        } catch {
          args = {};
        }
        const result = await executeTool(call.function.name, args);
        if (call.function.name === "request_navigation" && result?.ok && result.action) {
          navigation = result.action;
        }
        await pool.query(`INSERT INTO chat_messages (session_id, role, tool_name, content) VALUES (?, 'tool', ?, ?)`, [
          sessionId,
          call.function.name,
          JSON.stringify(result).slice(0, 4000),
        ]);
        llmMessages.push({
          role: "tool",
          name: call.function.name,
          tool_call_id: call.id,
          content: JSON.stringify(result).slice(0, 4000),
        });
      }
      continue;
    }
    replyText = String(messageOut.content || "").trim();
    break;
  }
  if (!replyText) {
    const last = await callLlm(llmMessages, { useTools: false, temperature });
    replyText = String(last.content || "").trim();
  }
  return { replyText, navigation };
}

export async function runAssistantChat(user, sessionPublicId, message, options = {}) {
  const text = String(message || "").trim().slice(0, 4000);
  if (!text) return { ok: false, error: "Type a question first." };

  const pool = await getPool();
  let session = sessionPublicId ? await ownedSession(pool, user.id, sessionPublicId) : null;
  if (!session) {
    const created = await createSession(user.id);
    session = await ownedSession(pool, user.id, created.id);
  }

  const replaceFrom = Number(options.replaceFrom);
  if (Number.isInteger(replaceFrom) && replaceFrom >= 0) {
    const [visible] = await pool.query(
      `SELECT id FROM chat_messages WHERE session_id = ? AND role IN ('user','assistant') ORDER BY id ASC`,
      [session.id]
    );
    if (replaceFrom < visible.length) {
      await pool.query("DELETE FROM chat_messages WHERE session_id = ? AND id >= ?", [
        session.id,
        visible[replaceFrom].id,
      ]);
    }
  }

  const [history] = await pool.query(
    `SELECT role, content, follow_ups FROM chat_messages
     WHERE session_id = ? AND role IN ('user','assistant')
     ORDER BY id DESC LIMIT 6`,
    [session.id]
  );
  history.reverse();
  const BAD = new Set([FRIENDLY_FAIL, CHITCHAT_FALLBACK]);
  const cleanHistory = history.filter((row) => !BAD.has(String(row.content || "").trim()));

  await pool.query(`INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'user', ?)`, [session.id, text]);
  if (session.title === "New chat") {
    await pool.query("UPDATE chat_sessions SET title = ? WHERE id = ?", [titleFromPrompt(text), session.id]);
  }

  const firstName = user.first_name || "there";
  const lastAsset = extractAssetHint(text) || lastAssetFromHistory(cleanHistory);
  const intent = classifyIntent(text);

  const finishReply = async (reply, followUpsOverride, navigation = null) => {
    const parsed = parseFollowUps(String(reply || ""));
    const finalReply = parsed.reply;
    const finalFollowUps = Array.isArray(followUpsOverride) ? followUpsOverride : parsed.followUps;
    await pool.query(
      `INSERT INTO chat_messages (session_id, role, content, follow_ups, navigation_json) VALUES (?, 'assistant', ?, ?, ?)`,
      [session.id, finalReply, JSON.stringify(finalFollowUps), navigation ? JSON.stringify(navigation) : null]
    );
    await pool.query("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", [session.id]);
    const [fresh] = await pool.query("SELECT title FROM chat_sessions WHERE id = ?", [session.id]);
    return {
      ok: true,
      sessionId: session.public_id,
      title: fresh[0]?.title || session.title,
      reply: finalReply,
      followUps: finalFollowUps,
      navigation,
      action: navigation
        ? { type: "navigate", route: navigation.href || navigation.route, label: navigation.label }
        : null,
    };
  };

  if (isSimpleGreeting(text)) {
    return finishReply(CHITCHAT_FALLBACK);
  }

  const routesBlock = wrapBlock(
    "available_routes",
    Object.entries(AVAILABLE_ROUTES).map(([key, spec]) => ({ page: key, path: spec.path }))
  );
  const knowledge = intent === "chitchat" ? "" : knowledgeBlock(text);
  const lastAssetNote = lastAsset ? `\nLast mentioned asset in this chat: ${lastAsset}. Resolve "it/this" to that asset.\n` : "";

  const llmMessages = [
    {
      role: "system",
      content:
        SYSTEM_PROMPT +
        `\nThe signed-in user is ${firstName}.` +
        lastAssetNote +
        (intent === "chitchat" ? "\nThis is small talk. Reply warmly in 1-2 sentences. Do not say you lack data.\n" : "") +
        knowledge +
        routesBlock,
    },
    ...cleanHistory.map((row) => ({
      role: row.role === "assistant" ? "assistant" : "user",
      content: String(row.content || "").slice(0, 800),
    })),
    { role: "user", content: text },
  ];

  const temperature = intent === "chitchat" ? 0.7 : 0.3;
  const useTools = intent !== "chitchat";

  try {
    if (process.env.ASSISTANT_SKIP_LLM === "1") {
      const skip = new Error("LLM skipped for fallback test.");
      skip.code = "LLM";
      skip.status = 429;
      throw skip;
    }
    if (useTools) {
      const { replyText, navigation } = await runToolLoop(llmMessages, {
        temperature,
        pool,
        sessionId: session.id,
      });
      const stripped = stripAction(replyText || "");
      let validated = navigation || stripped.action || null;
      if (validated && !validated.href && validated.route) {
        const built = buildRoute(validated.route, { machineId: validated.machineId, reportId: validated.reportId });
        validated = built.ok ? built.action : null;
      }
      if (guessIntentForFallback(text) === "navigate" && !validated) {
        const hint = extractAssetHint(text) || lastAsset;
        if (hint) {
          const found = await resolveAsset(hint);
          if (found.found) {
            const page = /\banalytics\b/i.test(text)
              ? "analytics"
              : /\balerts?\b/i.test(text)
                ? "alerts"
                : /\breports?\b/i.test(text)
                  ? "reports"
                  : "machine-details";
            const built = buildRoute(page, { machineId: found.machine.id });
            if (built.ok) validated = built.action;
          }
        }
      }
      if (!stripped.text && !validated) {
        return finishReply(intent === "chitchat" ? CHITCHAT_FALLBACK : FRIENDLY_FAIL, []);
      }
      const reply =
        stripped.text ||
        (validated ? `Opening ${validated.label || "that page"} in MaintIQ.` : FRIENDLY_FAIL);
      return finishReply(reply, undefined, validated);
    }

    const messageOut = await callLlm(llmMessages, {
      useTools: false,
      temperature,
      timeoutMs: intent === "chitchat" ? 10000 : 25000,
    });
    const replyText = String(messageOut.content || "").trim();
    if (!replyText) {
      return finishReply(intent === "chitchat" ? CHITCHAT_FALLBACK : FRIENDLY_FAIL, []);
    }
    return finishReply(replyText);
  } catch (err) {
    console.error("Assistant LLM error:", err.status, err.message, err.stack);
    const fb = guessIntentForFallback(text);
    if (intent === "chitchat" || fb === "chitchat") return finishReply(CHITCHAT_FALLBACK, []);
    if (err.code === "NO_KEY") {
      return finishReply("I am ready once an LLM API key is in the project `.env` file.", []);
    }
    if (/\b(acknowledge|resolve|close|dismiss)\b.{0,40}\balert/i.test(text)) {
      return finishReply(
        "I can't change alert status from chat. Open the Alerts page to acknowledge or resolve it.",
        undefined,
        buildRoute("alerts").action
      );
    }
    if (fb === "explain") {
      const docs = retrieveKnowledge(text, 1);
      if (docs.length) {
        const excerpt = pickKnowledgeExcerpt(docs[0], text);
        if (excerpt) return finishReply(excerpt);
      }
    }
    if (fb === "navigate") {
      const hint = extractAssetHint(text) || lastAsset;
      if (hint) {
        const found = await resolveAsset(hint);
        if (found.found) {
          const page = /\banalytics\b/i.test(text)
            ? "analytics"
            : /\balerts?\b/i.test(text)
              ? "alerts"
              : "machine-details";
          const built = buildRoute(page, { machineId: found.machine.id });
          if (built.ok) {
            return finishReply(`Opening **${found.machine.name}** (${found.machine.assetId}) in MaintIQ.`, undefined, built.action);
          }
        }
      }
      if (/\bdashboard\b/i.test(text)) return finishReply("Opening the dashboard.", undefined, buildRoute("dashboard").action);
      if (/\balerts?\b/i.test(text)) return finishReply("Opening alerts.", undefined, buildRoute("alerts").action);
      if (/\bmonitoring\b/i.test(text)) return finishReply("Opening monitoring.", undefined, buildRoute("monitoring").action);
    }
    if (fb === "data") {
      if (
        /\b(how many|number of|count of)\b.{0,40}\b(machines?|assets?|alerts?)\b/i.test(text) ||
        /\bwhich\b.{0,40}\b(machines?|assets?).{0,40}\balerts?\b/i.test(text) ||
        /\b(machines?|assets?) (have|with).{0,20}alerts?\b/i.test(text)
      ) {
        const summary = await executeTool("get_fleet_summary", {});
        if (/\bwhich\b.{0,40}\balerts?\b|\balerts?\b.{0,30}\b(machines?|assets?)\b/i.test(text)) {
          const list = (summary.assetsWithOpenAlerts || [])
            .map((row) => `**${row.assetId}** (${row.openAlerts})`)
            .join(", ");
          return finishReply(
            list
              ? `**${summary.openAlertCount}** open alerts across **${summary.machineCount}** machines. Assets with open alerts: ${list}.`
              : `MaintIQ has **${summary.machineCount}** machines and no open alerts.`
          );
        }
        return finishReply(
          `MaintIQ currently has **${summary.machineCount}** registered machines and **${summary.openAlertCount}** open alerts.`
        );
      }
      const hint = extractAssetHint(text) || lastAsset;
      if (hint && /\balert/i.test(text)) {
        const alertsOut = await executeTool("get_alerts", { asset_id: hint, limit: 3 });
        if (alertsOut.status === "ok" && alertsOut.alerts?.length) {
          const alert = alertsOut.alerts[0];
          const extra = [alert.why, alert.value != null ? `Value ${alert.value}` : "", alert.threshold != null ? `threshold ${alert.threshold}` : ""]
            .filter(Boolean)
            .join(". ")
            .replace(/[;]+/g, ". ")
            .replace(/\s+/g, " ")
            .replace(/\.\s*\./g, ".");
          return finishReply(
            `**${alert.machine || alert.assetId}** has a **${alert.severity}** alert: ${alert.title}.${extra ? ` ${extra}.` : ""} This is an investigation signal, not a repair order.`
          );
        }
      }
      const lookup = hint || text;
      const found = await resolveAsset(lookup);
      if (found.found) {
        const asset = compactMachine(found.machine);
        return finishReply(
          `**${asset.name}** (${asset.assetId}) is listed at **${asset.location}** with health **${asset.health}**.`
        );
      }
      if (found.status === "ambiguous") {
        const names = (found.matches || []).map((row) => `**${row.name}** (${row.assetId})`).join(", ");
        return finishReply(`Several assets match that. Did you mean ${names}?`);
      }
      if (hint || /^(xyz-\d+|pump\s+\d+|motor\s+\d+|bearing\s+\d+)$/i.test(text)) {
        return finishReply(`I couldn't find an asset named "${String(text).slice(0, 80)}". Check the name or asset ID.`);
      }
    }
    const debug = process.env.NODE_ENV !== "production" ? ` (debug: ${err.status || ""} ${err.message})` : "";
    return finishReply(FRIENDLY_FAIL + debug, []);
  }
}
