import "./load-env.js";
import crypto from "crypto";
import { existsSync } from "fs";
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

export const dbConfig = {
  host: pick(process.env.MYSQL_HOST, local.host, "127.0.0.1"),
  port: Number(pick(process.env.MYSQL_PORT, local.port, 3306)),
  user: pick(process.env.MYSQL_USER, local.user, "root"),
  password: pick(process.env.MYSQL_PASSWORD, local.password, ""),
  database: pick(process.env.MYSQL_DATABASE, local.database, "maintiq"),
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
  model: process.env.LLM_MODEL || "gemini-3.8-flash",
  baseUrl: process.env.LLM_BASE_URL || "https://generativelanguage.googleapis.com/v1beta",
};

export const NAME_MAX_LENGTH = 50;
export const PHONE_MAX_DIGITS = 10;
export const PASSWORD_MIN_LENGTH = 9;
