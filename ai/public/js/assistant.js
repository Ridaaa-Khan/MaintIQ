(() => {
  const root = document.getElementById("assistant-workspace");
  if (!root) return;

  const pinnedEl = document.getElementById("assistant-pinned");
  const recentsEl = document.getElementById("assistant-recents");
  const pinnedGroup = document.getElementById("assistant-pinned-group");
  const emptyEl = document.getElementById("assistant-history-empty");
  const searchEl = document.getElementById("assistant-search");
  const logEl = document.getElementById("assistant-log");
  const titleWrap = document.querySelector(".assistant-chat-title-wrap");
  const inputEl = document.getElementById("assistant-input");
  const formEl = document.getElementById("assistant-form");
  const pinActiveBtn = document.getElementById("assistant-pin-active");
  const renameActiveBtn = document.getElementById("assistant-rename-active");
  const sendBtn = document.getElementById("assistant-send");

  const ALLOWED_HREF = /^\/(dashboard|machines|monitoring|alerts|analytics|reports|assistant)(\/|\?|$)/;
  const titleNode = () => document.getElementById("assistant-chat-title");
  const SEND_ICON =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
  const STOP_ICON =
    '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
  const COPY_ICON =
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
  const EDIT_ICON =
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>';

  let sessions = [];
  let activeId = "";
  let messages = [];
  let busy = false;
  let activeController = null;
  let editFromIndex = null;

  const isBlankChat = () => !messages.some((message) => message.role === "user");

  const api = async (url, options = {}) => {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || "Request failed.");
    }
    return payload;
  };

  const closeMenus = () => {
    root.querySelectorAll(".assistant-history-menu").forEach((menu) => {
      menu.hidden = true;
    });
  };

  const renderHistoryItem = (chat) => {
    const item = document.createElement("div");
    item.className = `assistant-history-item${chat.id === activeId ? " is-active" : ""}`;
    item.dataset.chatId = chat.id;
    item.innerHTML = `
      <button class="assistant-history-item-main" type="button" data-select-chat="${chat.id}">
        <span class="assistant-history-item-title"></span>
      </button>
      <div class="assistant-history-item-actions">
        <button class="btn btn-icon menu-trigger" type="button" aria-label="Chat actions" data-chat-menu="${chat.id}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="19" r="1.6"/></svg>
        </button>
        <div class="assistant-history-menu" hidden data-menu-for="${chat.id}">
          <button type="button" data-edit-title="${chat.id}">Edit title</button>
          <button type="button" data-toggle-pin="${chat.id}">${chat.pinned ? "Unpin" : "Pin"}</button>
          <button type="button" data-delete-chat="${chat.id}">Delete</button>
        </div>
      </div>
    `;
    item.querySelector(".assistant-history-item-title").textContent = chat.title;
    return item;
  };

  const renderHistory = () => {
    const query = (searchEl?.value || "").trim().toLowerCase();
    const matched = sessions.filter((chat) => !query || chat.title.toLowerCase().includes(query));
    const pinned = matched.filter((chat) => chat.pinned).sort((a, b) => b.updatedAt - a.updatedAt);
    const recents = matched.filter((chat) => !chat.pinned).sort((a, b) => b.updatedAt - a.updatedAt);
    pinnedEl.innerHTML = "";
    recentsEl.innerHTML = "";
    pinned.forEach((chat) => pinnedEl.appendChild(renderHistoryItem(chat)));
    recents.forEach((chat) => recentsEl.appendChild(renderHistoryItem(chat)));
    pinnedGroup.hidden = pinned.length === 0;
    emptyEl.hidden = matched.length > 0;
  };

  const escapeHtml = (value) =>
    String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const formatMarkdown = (raw) => {
    const escaped = escapeHtml(raw).replace(/\r\n/g, "\n");
    const lines = escaped.split("\n");
    const html = [];
    let inList = false;

    const closeList = () => {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
    };

    const inline = (text) =>
      text
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[\s(])\*(.+?)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");

    for (const line of lines) {
      const heading = line.match(/^###\s+(.+)$/);
      const bullet = line.match(/^\s*[-*]\s+(.+)$/);
      if (heading) {
        closeList();
        html.push(`<h3>${inline(heading[1])}</h3>`);
        continue;
      }
      if (bullet) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push(`<li>${inline(bullet[1])}</li>`);
        continue;
      }
      closeList();
      if (!line.trim()) {
        html.push("");
        continue;
      }
      html.push(`<p>${inline(line)}</p>`);
    }
    closeList();
    return html.filter((part, index, arr) => part !== "" || arr[index - 1] !== "").join("");
  };

  const appendBubble = (message, index) => {
    const row = document.createElement("div");
    row.className = `assistant-msg ${message.role === "user" ? "is-user" : "is-ai"}`;
    row.innerHTML = `
      <div class="assistant-msg-avatar" aria-hidden="true">${message.role === "user" ? "You" : "AI"}</div>
      <div class="assistant-msg-body">
        <div class="assistant-msg-bubble"></div>
      </div>
    `;
    const bubble = row.querySelector(".assistant-msg-bubble");
    if (message.role === "assistant") {
      bubble.classList.add("assistant-md");
      if (message.typing) {
        row.classList.add("is-typing");
        bubble.innerHTML = `<span class="assistant-typing" aria-label="Assistant is typing"><i></i><i></i><i></i></span>`;
      } else {
        bubble.innerHTML = formatMarkdown(message.text);
      }
    } else {
      bubble.textContent = message.text;
    }
    const body = row.querySelector(".assistant-msg-body");
    const actions = document.createElement("div");
    actions.className = "assistant-msg-actions";
    if (message.typing) {
      logEl.appendChild(row);
      return;
    }
    if (message.role === "user") {
      actions.innerHTML = `
        <button type="button" class="assistant-msg-action" data-copy-prompt="${index}" aria-label="Copy prompt" title="Copy">
          ${COPY_ICON}
        </button>
        <button type="button" class="assistant-msg-action" data-edit-prompt="${index}" aria-label="Edit prompt" title="Edit">
          ${EDIT_ICON}
        </button>
      `;
    } else {
      actions.innerHTML = `
        <button type="button" class="assistant-msg-action" data-copy-prompt="${index}" aria-label="Copy response" title="Copy">
          ${COPY_ICON}
        </button>
      `;
    }
    body.appendChild(actions);
    if (message.navigation?.href && ALLOWED_HREF.test(message.navigation.href)) {
      const link = document.createElement("a");
      link.className = "btn btn-primary assistant-nav";
      link.href = message.navigation.href;
      link.textContent = message.navigation.label || "Open page";
      body.appendChild(link);
    }
    if (message.followUps?.length) {
      const wrap = document.createElement("div");
      wrap.className = "assistant-followups";
      message.followUps.forEach((prompt) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn";
        btn.textContent = prompt;
        btn.dataset.assistantPrompt = prompt;
        wrap.appendChild(btn);
      });
      body.appendChild(wrap);
    }
    logEl.appendChild(row);
  };

  const renderMessages = () => {
    logEl.innerHTML = "";
    if (isBlankChat()) {
      const heading = document.createElement("div");
      heading.className = "assistant-empty";
      heading.innerHTML = `
        <h2 class="assistant-empty-title">Welcome to MaintIQ AI Assistant</h2>
        <p class="assistant-empty-copy">I can help you understand machine health, interpret analytics and failure predictions, investigate alerts, and navigate MaintIQ. Ask me anything about your equipment, maintenance insights, or how to use the platform.</p>
        <div class="assistant-suggestions">
          <button type="button" class="btn" data-assistant-prompt="Which machines have active alerts?">Which machines have active alerts?</button>
          <button type="button" class="btn" data-assistant-prompt="What is Isolation Forest, and why are we using it in MaintIQ?">Why Isolation Forest?</button>
          <button type="button" class="btn" data-assistant-prompt="What is the asset ID of IMS Bearing 3?">Asset ID of IMS Bearing 3</button>
          <button type="button" class="btn" data-assistant-prompt="Show me analytics for BRG-004">Open analytics for BRG-004</button>
        </div>
      `;
      logEl.appendChild(heading);
      return;
    }
    messages.forEach((message, index) => appendBubble(message, index));
    logEl.scrollTop = logEl.scrollHeight;
  };

  const renderActiveChrome = () => {
    const chat = sessions.find((item) => item.id === activeId);
    if (!chat) return;
    const titleEl = titleNode();
    if (titleEl && titleEl.tagName !== "INPUT") titleEl.textContent = chat.title;
    if (pinActiveBtn) {
      pinActiveBtn.classList.toggle("btn-primary", chat.pinned);
      pinActiveBtn.setAttribute("aria-label", chat.pinned ? "Unpin chat" : "Pin chat");
      pinActiveBtn.title = chat.pinned ? "Unpin chat" : "Pin chat";
    }
  };

  const renderAll = () => {
    renderHistory();
    renderActiveChrome();
    renderMessages();
  };

  const setBusy = (next) => {
    busy = next;
    if (!sendBtn) return;
    sendBtn.disabled = false;
    sendBtn.classList.toggle("is-stop", next);
    sendBtn.setAttribute("aria-label", next ? "Stop generating" : "Send");
    sendBtn.title = next ? "Stop" : "Send";
    sendBtn.type = next ? "button" : "submit";
    sendBtn.innerHTML = next ? STOP_ICON : SEND_ICON;
  };

  const stopGenerating = () => {
    if (!busy) return;
    if (activeController) {
      activeController.abort();
      activeController = null;
    }
  };

  const upsertSession = (session) => {
    const index = sessions.findIndex((item) => item.id === session.id);
    const row = {
      id: session.id,
      title: session.title,
      pinned: Boolean(session.pinned),
      updatedAt: session.updatedAt || Date.now(),
    };
    if (index >= 0) sessions[index] = { ...sessions[index], ...row };
    else sessions.unshift(row);
    activeId = row.id;
  };

  const loadSession = async (id) => {
    const payload = await api(`/api/assistant/sessions/${encodeURIComponent(id)}`);
    upsertSession(payload.session);
    messages = payload.session.messages || [];
    activeId = payload.session.id;
    renderAll();
  };

  const ensureSession = async () => {
    const payload = await api("/api/assistant/sessions");
    sessions = payload.sessions || [];
    if (!sessions.length) {
      const created = await api("/api/assistant/sessions", { method: "POST", body: "{}" });
      upsertSession(created.session);
      messages = [];
      renderAll();
      return;
    }
    activeId = sessions[0].id;
    await loadSession(activeId);
  };

  const requestNewChat = async () => {
    if (isBlankChat() && sessions.some((item) => item.id === activeId)) {
      inputEl?.focus();
      return;
    }
    const created = await api("/api/assistant/sessions", { method: "POST", body: "{}" });
    upsertSession(created.session);
    messages = [];
    renderAll();
    inputEl?.focus();
  };

  const sendPrompt = async (raw) => {
    const text = String(raw || "").trim();
    if (!text || busy) return;
    if (!activeId) await ensureSession();

    let replaceFrom;
    if (editFromIndex != null && editFromIndex >= 0 && editFromIndex < messages.length) {
      replaceFrom = editFromIndex;
      messages = messages.slice(0, editFromIndex);
      editFromIndex = null;
    }

    messages.push({ role: "user", text });
    if (inputEl) inputEl.value = "";
    renderMessages();
    setBusy(true);
    const thinking = { role: "assistant", text: "", typing: true };
    messages.push(thinking);
    renderMessages();

    const controller = new AbortController();
    activeController = controller;

    try {
      const payload = await api("/api/assistant/chat", {
        method: "POST",
        body: JSON.stringify({
          sessionId: activeId,
          message: text,
          ...(replaceFrom != null ? { replaceFrom } : {}),
        }),
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      messages = messages.filter((item) => item !== thinking);
      messages.push({
        role: "assistant",
        text: payload.reply,
        followUps: payload.followUps || [],
        navigation: payload.navigation || payload.action || null,
      });
      const current = sessions.find((item) => item.id === activeId);
      if (current) {
        current.title = payload.title || current.title;
        current.updatedAt = Date.now();
      }
      if (payload.sessionId && payload.sessionId !== activeId) {
        activeId = payload.sessionId;
      }
      renderAll();
    } catch (err) {
      messages = messages.filter((item) => item !== thinking);
      if (err.name === "AbortError" || controller.signal.aborted) {
        messages.push({
          role: "assistant",
          text: "Stopped.",
          followUps: [],
        });
      } else {
        messages.push({
          role: "assistant",
          text: err.message || "I could not answer just now.",
          followUps: [],
        });
      }
      renderAll();
    } finally {
      if (activeController === controller) activeController = null;
      setBusy(false);
    }
  };

  const copyPrompt = async (index, button) => {
    const message = messages[index];
    if (!message) return;
    try {
      await navigator.clipboard.writeText(message.text);
      if (button) {
        const previous = button.getAttribute("aria-label");
        button.setAttribute("aria-label", "Copied");
        button.title = "Copied";
        button.classList.add("is-copied");
        window.setTimeout(() => {
          button.setAttribute("aria-label", previous || "Copy prompt");
          button.title = "Copy";
          button.classList.remove("is-copied");
        }, 1200);
      }
    } catch {
      /* ignore clipboard failures */
    }
  };

  const editPrompt = (index) => {
    const message = messages[index];
    if (!message || message.role !== "user" || busy) return;
    editFromIndex = index;
    if (inputEl) {
      inputEl.value = message.text;
      inputEl.focus();
      inputEl.setSelectionRange(message.text.length, message.text.length);
    }
  };

  const startTitleEdit = (id) => {
    const chat = sessions.find((item) => item.id === id);
    if (!chat || !titleWrap) return;
    closeMenus();
    if (chat.id !== activeId) {
      loadSession(id).catch(() => {});
    }
    const current = titleNode();
    if (!current || current.tagName === "INPUT") return;
    const input = document.createElement("input");
    input.type = "text";
    input.value = chat.title;
    input.setAttribute("aria-label", "Chat title");
    let done = false;
    const finish = async (commit) => {
      if (done) return;
      done = true;
      if (commit) {
        const next = input.value.trim().slice(0, 80) || "New chat";
        try {
          const payload = await api(`/api/assistant/sessions/${encodeURIComponent(id)}`, {
            method: "PATCH",
            body: JSON.stringify({ title: next }),
          });
          upsertSession(payload.session);
        } catch {
          chat.title = next;
        }
      }
      const heading = document.createElement("h2");
      heading.id = "assistant-chat-title";
      input.replaceWith(heading);
      renderAll();
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      }
      if (event.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
    current.replaceWith(input);
    input.focus();
    input.select();
  };

  const togglePin = async (id) => {
    const chat = sessions.find((item) => item.id === id);
    if (!chat) return;
    const payload = await api(`/api/assistant/sessions/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ pinned: !chat.pinned }),
    });
    upsertSession(payload.session);
    closeMenus();
    renderAll();
  };

  const deleteChat = async (id) => {
    await api(`/api/assistant/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    sessions = sessions.filter((item) => item.id !== id);
    closeMenus();
    if (!sessions.length) {
      await requestNewChat();
      return;
    }
    if (activeId === id) await loadSession(sessions[0].id);
    else renderAll();
  };

  ensureSession()
    .then(() => {
      const params = new URLSearchParams(window.location.search);
      const incoming = params.get("prompt");
      if (incoming) {
        window.history.replaceState({}, "", "/assistant");
        return sendPrompt(incoming);
      }
      return null;
    })
    .catch(() => {
      logEl.innerHTML = `<p class="muted" style="padding:18px">Could not load chats. Refresh the page.</p>`;
    });

  document.querySelectorAll("[data-assistant-new]").forEach((btn) => {
    btn.addEventListener("click", () => requestNewChat().catch(() => {}));
  });

  searchEl?.addEventListener("input", () => {
    closeMenus();
    renderHistory();
  });

  formEl?.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy) {
      stopGenerating();
      return;
    }
    sendPrompt(inputEl?.value);
  });

  sendBtn?.addEventListener("click", (event) => {
    if (!busy) return;
    event.preventDefault();
    stopGenerating();
  });

  logEl.addEventListener("click", (event) => {
    const copyBtn = event.target.closest("[data-copy-prompt]");
    if (copyBtn) {
      event.preventDefault();
      copyPrompt(Number(copyBtn.dataset.copyPrompt), copyBtn);
      return;
    }
    const editBtn = event.target.closest("[data-edit-prompt]");
    if (editBtn) {
      event.preventDefault();
      editPrompt(Number(editBtn.dataset.editPrompt));
      return;
    }
    const btn = event.target.closest("[data-assistant-prompt]");
    if (btn) sendPrompt(btn.dataset.assistantPrompt);
  });

  pinActiveBtn?.addEventListener("click", () => {
    if (activeId) togglePin(activeId).catch(() => {});
  });

  renameActiveBtn?.addEventListener("click", () => {
    if (activeId) startTitleEdit(activeId);
  });

  root.addEventListener("click", (event) => {
    const select = event.target.closest("[data-select-chat]");
    if (select) {
      loadSession(select.dataset.selectChat).catch(() => {});
      return;
    }
    const menuBtn = event.target.closest("[data-chat-menu]");
    if (menuBtn) {
      event.stopPropagation();
      const id = menuBtn.dataset.chatMenu;
      const menu = root.querySelector(`[data-menu-for="${id}"]`);
      const wasOpen = menu && !menu.hidden;
      closeMenus();
      if (menu && !wasOpen) menu.hidden = false;
      return;
    }
    const edit = event.target.closest("[data-edit-title]");
    if (edit) {
      startTitleEdit(edit.dataset.editTitle);
      return;
    }
    const pin = event.target.closest("[data-toggle-pin]");
    if (pin) {
      togglePin(pin.dataset.togglePin).catch(() => {});
      return;
    }
    const remove = event.target.closest("[data-delete-chat]");
    if (remove) {
      deleteChat(remove.dataset.deleteChat).catch(() => {});
      return;
    }
    if (!event.target.closest(".assistant-history-item-actions")) closeMenus();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".assistant-history-item-actions")) closeMenus();
  });
})();
