import mysql from "mysql2/promise";
import { dbConfig } from "./config.js";

let pool;

async function indexExists(pool, table, index) {
  const [rows] = await pool.query(
    `SELECT INDEX_NAME
     FROM information_schema.statistics
     WHERE table_schema = DATABASE() AND table_name = ? AND index_name = ?
     LIMIT 1`,
    [table, index]
  );
  return Boolean(rows.length);
}

async function dropIndexIfExists(pool, table, index) {
  if (await indexExists(pool, table, index)) {
    await pool.query(`ALTER TABLE \`${table}\` DROP INDEX \`${index}\``);
  }
}

async function addUniqueIfMissing(pool, table, index, columns) {
  if (await indexExists(pool, table, index)) return;
  await pool.query(`ALTER TABLE \`${table}\` ADD UNIQUE KEY \`${index}\` (${columns})`);
}

async function migrateOwnershipIndexes(pool) {
  await dropIndexIfExists(pool, "machines", "uq_machines_public_id");
  await dropIndexIfExists(pool, "machines", "uq_machines_asset_id");
  await addUniqueIfMissing(pool, "machines", "uq_machines_owner_public", "created_by, public_id");
  await addUniqueIfMissing(pool, "machines", "uq_machines_owner_asset", "created_by, asset_id");

  await dropIndexIfExists(pool, "alerts", "uq_alerts_key");
  await addUniqueIfMissing(pool, "alerts", "uq_alerts_machine_key", "machine_id, alert_key");

  await dropIndexIfExists(pool, "analysis_runs", "uq_analysis_runs_code");
  await addUniqueIfMissing(pool, "analysis_runs", "uq_analysis_runs_machine_code", "machine_id, run_code");
}

export async function getPool() {
  if (pool) return pool;

    pool = mysql.createPool({
    host: dbConfig.host,
    port: dbConfig.port,
    user: dbConfig.user,
    password: dbConfig.password,
    database: dbConfig.database,
    ssl: { rejectUnauthorized: true },
    waitForConnections: true,
    connectionLimit: 10,
  });

  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      first_name VARCHAR(50) NOT NULL,
      last_name VARCHAR(50) NOT NULL,
      email VARCHAR(255) NOT NULL,
      phone VARCHAR(10) NULL,
      password_hash VARCHAR(255) NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE KEY uq_users_email (email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS machines (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      public_id VARCHAR(40) NOT NULL,
      asset_id VARCHAR(40) NOT NULL,
      name VARCHAR(80) NOT NULL,
      machine_type VARCHAR(40) NOT NULL,
      location VARCHAR(80) NOT NULL DEFAULT '',
      description VARCHAR(500) NOT NULL DEFAULT '',
      notes TEXT NULL,
      source ENUM('ims','upload') NOT NULL DEFAULT 'upload',
      test_id VARCHAR(32) NULL,
      bearing_id INT NULL,
      last_action ENUM('add','edit','note') NOT NULL DEFAULT 'add',
      created_by INT UNSIGNED NULL,
      updated_by INT UNSIGNED NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uq_machines_owner_public (created_by, public_id),
          UNIQUE KEY uq_machines_owner_asset (created_by, asset_id),
          KEY idx_machines_created_by (created_by)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS machine_activity (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      machine_id INT UNSIGNED NOT NULL,
      user_id INT UNSIGNED NULL,
      action ENUM('add','edit','note') NOT NULL,
      summary VARCHAR(500) NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      KEY idx_machine_activity_machine (machine_id),
      CONSTRAINT fk_machine_activity_machine FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS dataset_files (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      machine_id INT UNSIGNED NOT NULL,
      owner_user_id INT UNSIGNED NULL,
      original_name VARCHAR(255) NOT NULL,
      stored_path VARCHAR(500) NOT NULL,
      extension VARCHAR(8) NOT NULL,
      byte_size INT UNSIGNED NOT NULL,
      validation_status ENUM('parsed','rejected') NOT NULL,
      validation_message VARCHAR(1000) NOT NULL,
      row_count INT UNSIGNED NULL,
      column_count INT UNSIGNED NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      KEY idx_dataset_files_machine (machine_id),
      CONSTRAINT fk_dataset_files_machine FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS analysis_runs (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      machine_id INT UNSIGNED NOT NULL,
      dataset_file_id INT UNSIGNED NULL,
      run_code VARCHAR(80) NOT NULL,
      status ENUM('Queued','Running','Paused','Completed','Failed') NOT NULL,
      stage VARCHAR(120) NOT NULL,
      window_label VARCHAR(255) NOT NULL DEFAULT '',
      progress TINYINT UNSIGNED NULL,
      dataset_label VARCHAR(255) NOT NULL DEFAULT '',
      period_label VARCHAR(120) NOT NULL DEFAULT '',
      n_windows INT UNSIGNED NULL,
      n_bins INT UNSIGNED NULL,
      error_message VARCHAR(1000) NULL,
      scope ENUM('active','history') NOT NULL DEFAULT 'history',
      started_at DATETIME NULL,
      updated_at DATETIME NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_analysis_runs_machine_code (machine_id, run_code),
      KEY idx_analysis_runs_machine (machine_id),
      CONSTRAINT fk_analysis_runs_machine FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS alerts (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      alert_key VARCHAR(120) NOT NULL,
      machine_id INT UNSIGNED NOT NULL,
      severity ENUM('Critical','High','Medium','Low') NOT NULL,
      alert_type VARCHAR(80) NOT NULL,
      title VARCHAR(180) NOT NULL,
      trigger_text VARCHAR(255) NOT NULL,
      status ENUM('New','Acknowledged','Investigating','Resolved') NOT NULL DEFAULT 'New',
      occurred_at DATETIME NOT NULL,
      resolved_at DATETIME NULL,
      value_text VARCHAR(40) NOT NULL DEFAULT '',
      threshold_text VARCHAR(40) NOT NULL DEFAULT '',
      why_text TEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_alerts_machine_key (machine_id, alert_key),
      KEY idx_alerts_machine (machine_id),
      KEY idx_alerts_status (status),
      CONSTRAINT fk_alerts_machine FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS generated_reports (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      public_id VARCHAR(40) NOT NULL,
      machine_id INT UNSIGNED NOT NULL,
      title VARCHAR(120) NOT NULL,
      report_type VARCHAR(80) NOT NULL DEFAULT 'Condition assessment',
      period_label VARCHAR(120) NOT NULL DEFAULT '',
      dataset_id VARCHAR(80) NOT NULL DEFAULT '',
      run_id VARCHAR(80) NOT NULL DEFAULT '',
      payload_json LONGTEXT NOT NULL,
      created_by INT UNSIGNED NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE KEY uq_generated_reports_public_id (public_id),
      KEY idx_generated_reports_machine (machine_id),
      KEY idx_generated_reports_created (created_at),
      CONSTRAINT fk_generated_reports_machine FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS sessions (
      id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
      token_hash CHAR(64) NOT NULL,
      user_id INT UNSIGNED NOT NULL,
      csrf_token CHAR(64) NOT NULL,
      expires_at DATETIME NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE KEY uq_sessions_token_hash (token_hash),
      KEY idx_sessions_user (user_id),
      KEY idx_sessions_expires (expires_at),
      CONSTRAINT fk_sessions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await migrateOwnershipIndexes(pool);

  return pool;
}
