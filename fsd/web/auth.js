import crypto from "crypto";
import {
  NAME_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  PHONE_MAX_DIGITS,
  sessionSecret,
} from "./config.js";
import { getPool } from "./db.js";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const SPECIAL_RE = /[^A-Za-z0-9]/;
const SESSION_COOKIE = "maintiq_session";
const SESSION_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

function scryptAsync(password, salt) {
  return new Promise((resolve, reject) => {
    crypto.scrypt(password, salt, 64, (err, key) => {
      if (err) reject(err);
      else resolve(key);
    });
  });
}

export async function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString("hex");
  const derived = await scryptAsync(password, salt);
  return `${salt}:${derived.toString("hex")}`;
}

export async function verifyPassword(password, stored) {
  const [salt, hash] = String(stored || "").split(":");
  if (!salt || !hash) return false;
  const derived = await scryptAsync(password, salt);
  const a = Buffer.from(hash, "hex");
  const b = derived;
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

export function validateSignup({ firstName, lastName, email, phone, password }) {
  const errors = [];
  const values = {
    firstName: String(firstName || "").trim(),
    lastName: String(lastName || "").trim(),
    email: String(email || "").trim().toLowerCase(),
    phone: String(phone || "").trim(),
    password: String(password || ""),
  };

  if (!values.firstName) errors.push("First name is required.");
  else if (values.firstName.length > NAME_MAX_LENGTH) {
    errors.push(`First name must be at most ${NAME_MAX_LENGTH} characters.`);
  }

  if (!values.lastName) errors.push("Last name is required.");
  else if (values.lastName.length > NAME_MAX_LENGTH) {
    errors.push(`Last name must be at most ${NAME_MAX_LENGTH} characters.`);
  }

  if (!values.email) errors.push("Email ID is required.");
  else if (!EMAIL_RE.test(values.email) || values.email.length > 255) {
    errors.push("Email ID must be a valid email address.");
  }

  const phoneDigits = values.phone.replace(/\D/g, "");
  if (values.phone && phoneDigits.length === 0) {
    errors.push("Phone number must contain digits only.");
  } else if (phoneDigits.length > PHONE_MAX_DIGITS) {
    errors.push(`Phone number cannot be more than ${PHONE_MAX_DIGITS} digits.`);
  }
  values.phone = phoneDigits || null;

  if (!values.password) errors.push("Password is required.");
  else errors.push(...passwordRequirementErrors(values.password));

  return { ok: errors.length === 0, errors, values };
}

export const PASSWORD_REQUIREMENTS = [
  "More than 8 characters",
  "At least one uppercase letter",
  "At least one lowercase letter",
  "At least one digit",
  "At least one special character",
];

export function passwordRequirementErrors(password) {
  const errors = [];
  const value = String(password || "");
  if (value.length <= 8 || value.length < PASSWORD_MIN_LENGTH) {
    errors.push("Password must be more than 8 characters.");
  }
  if (!/[A-Z]/.test(value)) {
    errors.push("Password must include at least one uppercase letter.");
  }
  if (!/[a-z]/.test(value)) {
    errors.push("Password must include at least one lowercase letter.");
  }
  if (!/[0-9]/.test(value)) {
    errors.push("Password must include at least one digit.");
  }
  if (!SPECIAL_RE.test(value)) {
    errors.push("Password must include at least one special character.");
  }
  return errors;
}

export function validatePasswordChange({ currentPassword, newPassword, confirmPassword }) {
  const errors = [];
  const values = {
    currentPassword: String(currentPassword || ""),
    newPassword: String(newPassword || ""),
    confirmPassword: String(confirmPassword || ""),
  };

  if (!values.currentPassword) errors.push("Current password is required.");
  if (!values.newPassword) errors.push("New password is required.");
  else errors.push(...passwordRequirementErrors(values.newPassword));

  if (!values.confirmPassword) errors.push("Confirm new password is required.");
  else if (values.newPassword && values.newPassword !== values.confirmPassword) {
    errors.push("New password and confirmation do not match.");
  }

  return { ok: errors.length === 0, errors, values };
}

export function validateProfileUpdate({ firstName, lastName, email, phone }) {
  const errors = [];
  const values = {
    firstName: String(firstName || "").trim(),
    lastName: String(lastName || "").trim(),
    email: String(email || "").trim().toLowerCase(),
    phone: String(phone || "").trim(),
  };

  if (!values.firstName) errors.push("First name is required.");
  else if (values.firstName.length > NAME_MAX_LENGTH) {
    errors.push(`First name must be at most ${NAME_MAX_LENGTH} characters.`);
  }

  if (!values.lastName) errors.push("Last name is required.");
  else if (values.lastName.length > NAME_MAX_LENGTH) {
    errors.push(`Last name must be at most ${NAME_MAX_LENGTH} characters.`);
  }

  if (!values.email) errors.push("Email ID is required.");
  else if (!EMAIL_RE.test(values.email) || values.email.length > 255) {
    errors.push("Email ID must be a valid email address.");
  }

  const phoneDigits = values.phone.replace(/\D/g, "");
  if (values.phone && phoneDigits.length === 0) {
    errors.push("Phone number must contain digits only.");
  } else if (phoneDigits.length > PHONE_MAX_DIGITS) {
    errors.push(`Phone number cannot be more than ${PHONE_MAX_DIGITS} digits.`);
  }
  values.phone = phoneDigits || null;

  return { ok: errors.length === 0, errors, values };
}

export function validateSignin({ email, password }) {
  const errors = [];
  const values = {
    email: String(email || "").trim().toLowerCase(),
    password: String(password || ""),
  };

  if (!values.email) errors.push("Email ID is required.");
  else if (!EMAIL_RE.test(values.email)) {
    errors.push("Email ID must be a valid email address.");
  }
  if (!values.password) errors.push("Password is required.");

  return { ok: errors.length === 0, errors, values };
}

const AUTH_WINDOW_MS = 10 * 60 * 1000;
const AUTH_MAX_ATTEMPTS = 8;
const authHits = new Map();

function hashToken(token) {
  return crypto.createHmac("sha256", sessionSecret).update(token).digest("hex");
}

function cookieFlags(req, { clear = false } = {}) {
  const forwarded = String(req?.headers?.["x-forwarded-proto"] || "").split(",")[0].trim();
  const secure =
    process.env.COOKIE_SECURE === "1" ||
    Boolean(req?.secure) ||
    forwarded === "https";
  const maxAge = clear ? 0 : Math.floor(SESSION_MAX_AGE_MS / 1000);
  return `Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}${secure ? "; Secure" : ""}`;
}

export function parseCookies(req) {
  const header = req.headers.cookie || "";
  const out = {};
  header.split(";").forEach((part) => {
    const idx = part.indexOf("=");
    if (idx === -1) return;
    const key = part.slice(0, idx).trim();
    const val = part.slice(idx + 1).trim();
    if (!key) return;
    try {
      out[key] = decodeURIComponent(val);
    } catch {
      out[key] = val;
    }
  });
  return out;
}

function readSessionToken(req) {
  const cookies = parseCookies(req);
  const raw = cookies[SESSION_COOKIE];
  return raw ? String(raw) : "";
}

export async function loadSession(req) {
  const token = readSessionToken(req);
  if (!token) return null;
  const pool = await getPool();
  const [rows] = await pool.query(
    `SELECT s.id, s.user_id, s.csrf_token, s.expires_at
     FROM sessions s
     WHERE s.token_hash = ? LIMIT 1`,
    [hashToken(token)]
  );
  const row = rows[0];
  if (!row) return null;
  if (new Date(row.expires_at).getTime() <= Date.now()) {
    await pool.query("DELETE FROM sessions WHERE id = ?", [row.id]);
    return null;
  }
  return {
    id: row.id,
    userId: Number(row.user_id),
    csrfToken: row.csrf_token,
    token,
  };
}

export function clearSessionCookie(res, req) {
  res.setHeader("Set-Cookie", `${SESSION_COOKIE}=; ${cookieFlags(req, { clear: true })}`);
}

export async function createSession(req, res, userId) {
  const token = crypto.randomBytes(32).toString("base64url");
  const csrfToken = crypto.randomBytes(32).toString("hex");
  const expires = new Date(Date.now() + SESSION_MAX_AGE_MS);
  const pool = await getPool();
  await pool.query(
    `INSERT INTO sessions (token_hash, user_id, csrf_token, expires_at) VALUES (?, ?, ?, ?)`,
    [hashToken(token), Number(userId), csrfToken, expires]
  );
  res.setHeader(
    "Set-Cookie",
    `${SESSION_COOKIE}=${encodeURIComponent(token)}; ${cookieFlags(req)}`
  );
  return { userId: Number(userId), csrfToken, token };
}

export async function destroySession(req, res) {
  const token = readSessionToken(req);
  if (token) {
    const pool = await getPool();
    await pool.query("DELETE FROM sessions WHERE token_hash = ?", [hashToken(token)]);
  }
  clearSessionCookie(res, req);
}

export async function destroyAllSessions(userId, exceptSessionId = null) {
  const pool = await getPool();
  if (exceptSessionId) {
    await pool.query("DELETE FROM sessions WHERE user_id = ? AND id <> ?", [userId, exceptSessionId]);
    return;
  }
  await pool.query("DELETE FROM sessions WHERE user_id = ?", [userId]);
}

export function readCsrfToken(req) {
  const header = req.headers["x-csrf-token"];
  if (header) return String(header).trim();
  if (req.body && req.body._csrf) return String(req.body._csrf).trim();
  return "";
}

export function csrfTokensMatch(sessionToken, provided) {
  const a = Buffer.from(String(sessionToken || ""));
  const b = Buffer.from(String(provided || ""));
  if (!a.length || a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

export function consumeAuthAttempt(req) {
  const ip = String(req.socket?.remoteAddress || req.ip || "unknown");
  const now = Date.now();
  const rec = authHits.get(ip) || { n: 0, start: now };
  if (now - rec.start > AUTH_WINDOW_MS) {
    rec.n = 0;
    rec.start = now;
  }
  rec.n += 1;
  authHits.set(ip, rec);
  return rec.n <= AUTH_MAX_ATTEMPTS;
}

export async function getSessionUserId(req) {
  const session = await loadSession(req);
  return session ? session.userId : null;
}
