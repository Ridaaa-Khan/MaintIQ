import "./load-env.js";
import crypto from "crypto";
import { existsSync, readFileSync } from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let local = {};
const localPath = path.join(__dirname, "db.local.js");
if (existsSync(localPath)) {
  local = (await import(pathToFileURL(localPath).href)).default || {};
}

function pick(envValue, localValue, fallback = "") {
  if (envValue != null && String(envValue).length) return envValue;
  if (localValue != null && String(localValue).length) return localValue;
  return fallback;
}

function parseFlag(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "require", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off", "disable", "disabled"].includes(normalized)) return false;
  return null;
}

/** Render/Aiven env vars often store PEM with literal \n sequences. */
function normalizePem(value) {
  let pem = String(value).trim();
  if (
    (pem.startsWith('"') && pem.endsWith('"')) ||
    (pem.startsWith("'") && pem.endsWith("'"))
  ) {
    pem = pem.slice(1, -1);
  }
  return pem.replace(/\\n/g, "\n");
}

function loadMysqlCa() {
  const caInline = String(process.env.MYSQL_SSL_CA || local.sslCa || "").trim();
  if (caInline) {
    const ca = normalizePem(caInline);
    if (!ca.includes("BEGIN CERTIFICATE")) {
      throw new Error(
        "MYSQL_SSL_CA must be a PEM certificate (include -----BEGIN CERTIFICATE-----)."
      );
    }
    return ca;
  }

  const caPath = String(process.env.MYSQL_SSL_CA_PATH || local.sslCaPath || "").trim();
  if (!caPath) return "";

  const resolved = path.isAbsolute(caPath) ? caPath : path.join(__dirname, caPath);
  if (!existsSync(resolved)) {
    throw new Error(`MYSQL_SSL_CA_PATH file not found: ${resolved}`);
  }
  return readFileSync(resolved, "utf8");
}

function resolveMysqlSsl(host) {
  const production = process.env.NODE_ENV === "production";
  const isLocalHost = ["127.0.0.1", "localhost", "::1"].includes(String(host).toLowerCase());
  const explicit = parseFlag(process.env.MYSQL_SSL ?? local.ssl);

  // Explicit MYSQL_SSL wins. Dev + local MySQL defaults to no TLS.
  // Remote hosts (Aiven) default to TLS so certificate verification is used.
  const useSsl = explicit != null ? explicit : !isLocalHost;

  if (!useSsl) {
    if (production && !isLocalHost) {
      throw new Error(
        "MySQL SSL is required in production for remote hosts. Set MYSQL_SSL=1 and MYSQL_SSL_CA."
      );
    }
    return undefined;
  }

  const ca = loadMysqlCa();
  if (!ca) {
    if (production) {
      throw new Error(
        "MYSQL_SSL_CA is required in production. Paste the Aiven project CA PEM " +
          "(service Overview → Download CA certificate) into the MYSQL_SSL_CA env var. " +
          "Do not set rejectUnauthorized=false."
      );
    }
    throw new Error(
      "MYSQL_SSL is enabled but MYSQL_SSL_CA is missing. " +
        "Set MYSQL_SSL_CA to the Aiven CA PEM, or set MYSQL_SSL=0 for local MySQL without TLS."
    );
  }

  const ssl = {
    ca,
    // Always verify the server certificate against the Aiven project CA.
    rejectUnauthorized: true,
  };

  // Optional hostname check (Aiven VERIFY_IDENTITY).
  if (parseFlag(process.env.MYSQL_SSL_VERIFY_IDENTITY) === true) {
    ssl.verifyIdentity = true;
  }

  return ssl;
}

const host = pick(process.env.MYSQL_HOST, local.host, "127.0.0.1");

export const dbConfig = {
  host,
  port: Number(pick(process.env.MYSQL_PORT, local.port, 3306)),
  user: pick(process.env.MYSQL_USER, local.user, "root"),
  password: pick(process.env.MYSQL_PASSWORD, local.password, ""),
  database: pick(process.env.MYSQL_DATABASE, local.database, "maintiq"),
  ssl: resolveMysqlSsl(host),
};

const PLACEHOLDER_SECRET = "maintiq-dev-session-secret-change-me";
const configuredSecret = String(process.env.SESSION_SECRET || local.sessionSecret || "").trim();
const production = process.env.NODE_ENV === "production";

if (production && (!configuredSecret || configuredSecret === PLACEHOLDER_SECRET)) {
  throw new Error("SESSION_SECRET must be set to a strong unique value in production.");
}

export const sessionSecret =
  configuredSecret && configuredSecret !== PLACEHOLDER_SECRET
    ? configuredSecret
    : crypto.randomBytes(32).toString("hex");

if (!configuredSecret || configuredSecret === PLACEHOLDER_SECRET) {
  console.warn("SESSION_SECRET is not set; using an ephemeral development secret. Sessions reset on restart.");
}

export const llmConfig = {
  provider: process.env.LLM_PROVIDER || "google",
  apiKey: String(process.env.LLM_API_KEY || "").trim(),
  model: process.env.LLM_MODEL || "gemini-3.5-flash",
  baseUrl: process.env.LLM_BASE_URL || "https://generativelanguage.googleapis.com/v1beta",
};

export const NAME_MAX_LENGTH = 50;
export const PHONE_MAX_DIGITS = 10;
export const PASSWORD_MIN_LENGTH = 9;
