# MaintIQ AI Assistant (deferred)

Standalone assistant package kept for a later release. It is **not** imported,
routed, or served by `fsd/web` right now.

| Path | Role |
|------|------|
| `assistant.js` | Chat sessions, tools, Gemini calls, and reply routing |
| `knowledge/` | Markdown notes for model/PM explanations |
| `views/assistant.ejs` | Assistant page template |
| `public/js/assistant.js` | Browser chat UI |

To re-enable later: restore Express routes/static mounts in `fsd/web`, recreate
`chat_sessions` / `chat_messages` tables, and set `LLM_*` env vars.
