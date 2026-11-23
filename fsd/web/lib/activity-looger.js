
import { getPool } from "../db.js";

export async function logActivity({
  userId = null,
  action,
  entityType,
  entityId = null,
  status = "success",
  summary = "",
}) {
  const event = {
    timestamp: new Date().toISOString(),
    level: "INFO",
    type: "ACTIVITY",
    action,
    userId,
    entityType,
    entityId,
    status,
    summary,
  };

  // Print the event to the application console.
  // systemd will capture this output on EC2.
  console.log(JSON.stringify(event));

  // Also store the event permanently in MySQL.
  try {
    const pool = await getPool();

    await pool.query(
      `INSERT INTO activity_logs
       (user_id, action, entity_type, entity_id, status, summary)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [
        userId,
        action,
        entityType,
        entityId == null ? null : String(entityId),
        status,
        String(summary).slice(0, 500),
      ]
    );
  } catch (error) {
    console.error("[ACTIVITY_LOG_DB_ERROR]", error.message);
  }
}
