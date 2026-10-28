/** Copy to db.local.js and fill in your MySQL password. db.local.js is gitignored. */
export default {
  host: "127.0.0.1",
  port: 3306,
  user: "root",
  password: "YOUR_MYSQL_PASSWORD",
  database: "maintiq",
  // Local default: no SSL. For Aiven locally, set ssl: true and sslCa / sslCaPath.
  // ssl: true,
  // sslCaPath: "./certs/aiven-ca.pem",
};
