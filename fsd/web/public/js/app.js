(() => {
  const csrfToken = () => document.querySelector('meta[name="csrf-token"]')?.getAttribute("content") || "";
  const csrfHeaders = (headers = {}) => {
    const token = csrfToken();
    if (!token) return headers;
    return { ...headers, "X-CSRF-Token": token };
  };

  const shell = document.getElementById("app-shell");
  if (!shell) return;

  const toggle = () => {
    if (window.matchMedia("(max-width: 860px)").matches) {
      shell.classList.toggle("is-mobile-open");
      return;
    }
    shell.classList.toggle("is-collapsed");
    localStorage.setItem("maintiq-sidebar", shell.classList.contains("is-collapsed") ? "collapsed" : "open");
  };

  if (localStorage.getItem("maintiq-sidebar") === "collapsed") {
    shell.classList.add("is-collapsed");
  }

  document.querySelectorAll("[data-sidebar-toggle]").forEach((el) => {
    el.addEventListener("click", toggle);
  });
  document.querySelectorAll("[data-sidebar-close]").forEach((el) => {
    el.addEventListener("click", () => shell.classList.remove("is-mobile-open"));
  });

  document.querySelectorAll("[data-sidebar-dropdown]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const root = btn.closest(".sidebar-dropdown");
      const panel = root?.querySelector(".sidebar-submenu");
      if (!root || !panel) return;
      const willOpen = panel.hidden;
      root.classList.toggle("is-open", willOpen);
      panel.hidden = !willOpen;
      btn.setAttribute("aria-expanded", String(willOpen));
    });
  });

  const avatarKey = (userId) => `maintiq-avatar-${userId}`;

  const closeAvatarMenus = () => {
    document.querySelectorAll("[data-avatar-controls]").forEach((root) => {
      root.classList.remove("is-open");
      const toggle = root.querySelector("[data-avatar-menu-toggle]");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
    });
  };

  const syncAvatarControls = (userId, hasPhoto) => {
    document.querySelectorAll(`[data-avatar-controls][data-user-id="${userId}"]`).forEach((root) => {
      const uploadBtn = root.querySelector("[data-avatar-upload-label]");
      const changeBtn = root.querySelector("[data-avatar-change-label]");
      const deleteBtn = root.querySelector("[data-avatar-delete]");
      if (uploadBtn) {
        uploadBtn.hidden = hasPhoto;
        uploadBtn.style.display = hasPhoto ? "none" : "";
      }
      if (changeBtn) {
        changeBtn.hidden = !hasPhoto;
        changeBtn.style.display = hasPhoto ? "" : "none";
      }
      if (deleteBtn) {
        deleteBtn.hidden = !hasPhoto;
        deleteBtn.style.display = hasPhoto ? "" : "none";
      }
      root.dataset.hasPhoto = hasPhoto ? "1" : "0";
    });
  };

  const applyAvatar = (userId, dataUrl) => {
    document.querySelectorAll(`[data-avatar-slot][data-user-id="${userId}"]`).forEach((slot) => {
      if (dataUrl) {
        slot.style.backgroundImage = `url("${dataUrl}")`;
        slot.classList.add("has-image");
      } else {
        slot.style.backgroundImage = "";
        slot.classList.remove("has-image");
      }
    });
    document.querySelectorAll(`[data-sidebar-avatar][data-user-id="${userId}"]`).forEach((avatar) => {
      if (dataUrl) {
        avatar.style.backgroundImage = `url("${dataUrl}")`;
        avatar.style.backgroundSize = "cover";
        avatar.style.backgroundPosition = "center";
        avatar.textContent = "";
      } else {
        avatar.style.backgroundImage = "";
        const initials = avatar.dataset.initials || avatar.getAttribute("data-initials") || "";
        if (initials) avatar.textContent = initials;
      }
    });
    syncAvatarControls(userId, Boolean(dataUrl));
  };

  document.querySelectorAll("[data-sidebar-avatar][data-user-id]").forEach((avatar) => {
    if (!avatar.dataset.initials) {
      avatar.dataset.initials = String(avatar.textContent || "").trim();
    }
  });

  document.querySelectorAll("[data-user-id]").forEach((el) => {
    const userId = el.dataset.userId;
    if (!userId) return;
    const stored = localStorage.getItem(avatarKey(userId));
    if (stored) applyAvatar(userId, stored);
    else syncAvatarControls(userId, false);
  });

  document.querySelectorAll("[data-avatar-menu-toggle]").forEach((toggle) => {
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const root = toggle.closest("[data-avatar-controls]");
      if (!root) return;
      const willOpen = !root.classList.contains("is-open");
      closeAvatarMenus();
      if (willOpen) {
        root.classList.add("is-open");
        toggle.setAttribute("aria-expanded", "true");
      }
    });
  });

  document.querySelectorAll("[data-avatar-controls]").forEach((root) => {
    root.addEventListener("mouseenter", () => {
      const userId = root.dataset.userId;
      const hasPhoto = localStorage.getItem(avatarKey(userId));
      syncAvatarControls(userId, Boolean(hasPhoto));
    });
  });

  document.querySelectorAll("[data-avatar-input]").forEach((input) => {
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      const userId = input.dataset.userId;
      if (!file || !userId) return;
      if (!/^image\/(png|jpeg)$/.test(file.type)) return;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = String(reader.result || "");
        localStorage.setItem(avatarKey(userId), dataUrl);
        applyAvatar(userId, dataUrl);
        input.value = "";
        closeAvatarMenus();
      };
      reader.readAsDataURL(file);
    });
  });

  document.querySelectorAll("[data-avatar-delete]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const userId = btn.dataset.userId;
      if (!userId) return;
      localStorage.removeItem(avatarKey(userId));
      applyAvatar(userId, null);
      const input = document.querySelector(`[data-avatar-input][data-user-id="${userId}"]`);
      if (input) input.value = "";
      closeAvatarMenus();
    });
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-avatar-controls]")) closeAvatarMenus();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAvatarMenus();
  });

  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "b") {
      event.preventDefault();
      toggle();
    }
  });

  document.querySelectorAll("[data-tabs]").forEach((root) => {
    const tabs = [...root.querySelectorAll(".tab")];
    const panels = [...root.querySelectorAll(".tab-panel")];
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.toggle("is-active", t === tab));
        panels.forEach((p) => p.classList.toggle("is-active", p.dataset.panel === tab.dataset.tab));
        window.setTimeout(() => {
          document.querySelectorAll("canvas").forEach((canvas) => {
            const chart = typeof Chart !== "undefined" ? Chart.getChart(canvas) : null;
            chart?.resize();
          });
        }, 50);
      });
    });
  });

  const content = document.getElementById("app-content");
  document.querySelectorAll("[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      content?.classList.toggle("is-cards", btn.dataset.view === "cards");
      document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("btn-primary", b === btn));
    });
  });

  const PAGE_SIZE = 8;
  const pagers = [];

  const paintPager = (root, page, pages, onPick) => {
    root.innerHTML = "";
    if (pages <= 1) {
      root.hidden = true;
      return;
    }
    root.hidden = false;
    root.classList.add("pager-nav");

    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "btn";
    prev.textContent = "Previous";
    prev.disabled = page <= 1;
    prev.addEventListener("click", () => {
      if (page > 1) onPick(page - 1);
    });

    const next = document.createElement("button");
    next.type = "button";
    next.className = "btn";
    next.textContent = "Next";
    next.disabled = page >= pages;
    next.addEventListener("click", () => {
      if (page < pages) onPick(page + 1);
    });

    root.append(prev, next);
  };

  const bindPager = (list, itemSelector, pager) => {
    if (!list || !pager) return { draw() {}, reset() {} };
    let page = 1;
    const draw = () => {
      const items = [...list.querySelectorAll(itemSelector)];
      const matched = items.filter((item) => item.dataset.match !== "0");
      const pages = Math.max(1, Math.ceil(matched.length / PAGE_SIZE));
      page = Math.min(page, pages);
      items.forEach((item) => {
        item.hidden = true;
      });
      matched.forEach((item, index) => {
        const start = (page - 1) * PAGE_SIZE;
        item.hidden = index < start || index >= start + PAGE_SIZE;
      });
      paintPager(pager, page, pages, (next) => {
        page = next;
        draw();
      });
    };
    const api = {
      draw,
      reset() {
        page = 1;
        draw();
      },
    };
    pagers.push(api);
    return api;
  };

  const matchesFilters = (el, q, filters) => {
    const hay = (el.dataset.search || el.textContent || "").toLowerCase();
    const matchQ = !q || hay.includes(q);
    const matchF = Object.entries(filters).every(([key, val]) => !val || el.dataset[key] === val);
    let matchDate = true;
    if (el.dataset.day) {
      const from = el.closest("body")?.querySelector("#alerts-from")?.value || document.querySelector("#alerts-from")?.value || "";
      const to = document.querySelector("#alerts-to")?.value || "";
      if (from && el.dataset.day < from) matchDate = false;
      if (to && el.dataset.day > to) matchDate = false;
    }
    return matchQ && matchF && matchDate;
  };

  const machineTablePager = bindPager(
    document.querySelector("#machine-table tbody"),
    "tr",
    document.querySelector("[data-pager-for='#machine-table']")
  );
  machineTablePager.listSelector = "#machine-table";
  const machineCardPager = bindPager(
    document.querySelector("#machine-cards"),
    ".machine-card",
    document.querySelector("[data-pager-for='#machine-cards']")
  );
  machineCardPager.listSelector = "#machine-cards";
  const alertPager = bindPager(
    document.querySelector("#alerts-table tbody"),
    "tr",
    document.querySelector("[data-pager-for='#alerts-table']")
  );
  alertPager.listSelector = "#alerts-table";
  const windowPager = bindPager(
    document.querySelector("#analytics-windows tbody"),
    "tr",
    document.querySelector("[data-pager-for='#analytics-windows']")
  );
  windowPager.listSelector = "#analytics-windows";
  window.refreshTablePager = (tableSelector) => {
    if (!tableSelector || tableSelector === "#analytics-windows") {
      windowPager.reset();
      return;
    }
    const found = pagers.find((pager) => pager.listSelector === tableSelector);
    found?.reset();
  };

  const applyFilters = (table) => {
    const q = (table.dataset.searchValue || "").toLowerCase();
    const filters = JSON.parse(table.dataset.filters || "{}");
    table.querySelectorAll("tbody tr").forEach((row) => {
      row.dataset.match = matchesFilters(row, q, filters) ? "1" : "0";
    });
    if (table.id === "machine-table") {
      document.querySelectorAll("#machine-cards .machine-card").forEach((card) => {
        card.dataset.match = matchesFilters(card, q, filters) ? "1" : "0";
      });
      machineTablePager.reset();
      machineCardPager.reset();
      return;
    }
    if (table.id === "alerts-table") alertPager.reset();
  };

  document.querySelectorAll("#machine-table tbody tr, #machine-cards .machine-card, #alerts-table tbody tr").forEach((el) => {
    el.dataset.match = "1";
  });
  machineTablePager.draw();
  machineCardPager.draw();
  alertPager.draw();

  document.querySelectorAll("[data-search]").forEach((input) => {
    input.addEventListener("input", () => {
      const table = document.querySelector(input.dataset.search);
      if (!table) return;
      table.dataset.searchValue = input.value;
      applyFilters(table);
    });
  });

  document.querySelectorAll("[data-filter]").forEach((select) => {
    select.addEventListener("change", () => {
      const table = document.querySelector(select.dataset.filter);
      if (!table) return;
      const filters = JSON.parse(table.dataset.filters || "{}");
      filters[select.dataset.key] = select.value;
      table.dataset.filters = JSON.stringify(filters);
      applyFilters(table);
    });
  });

  document.querySelectorAll("[data-sort]").forEach((select) => {
    select.addEventListener("change", () => {
      const table = document.querySelector(select.dataset.sort);
      if (!table) return;
      const rows = [...table.querySelectorAll("tbody tr")];
      const key = select.value;
      rows.sort((a, b) => {
        if (key === "created") return String(b.dataset.created || "").localeCompare(String(a.dataset.created || ""));
        if (key === "createdAsc") return String(a.dataset.created || "").localeCompare(String(b.dataset.created || ""));
        if (key === "severityRank") return Number(a.dataset.severityRank || 9) - Number(b.dataset.severityRank || 9);
        if (key === "score" || key === "last") {
          return String(b.dataset[key] || "").localeCompare(String(a.dataset[key] || ""), undefined, { numeric: true });
        }
        return String(a.dataset[key] || "").localeCompare(String(b.dataset[key] || ""), undefined, { numeric: true });
      });
      rows.forEach((row) => table.querySelector("tbody").appendChild(row));
      applyFilters(table);
    });
  });

  const open = (sel) => document.querySelector(sel)?.classList.add("is-open");
  const closeAll = () => {
    document.querySelectorAll(".overlay, .modal, .drawer").forEach((el) => el.classList.remove("is-open"));
  };

  const closeMenus = () => {
    document.querySelectorAll(".menu-panel").forEach((panel) => {
      panel.hidden = true;
    });
    document.querySelectorAll(".menu-trigger").forEach((btn) => {
      btn.setAttribute("aria-expanded", "false");
    });
  };

  document.querySelectorAll(".menu-trigger").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      const panel = btn.parentElement?.querySelector(".menu-panel");
      if (!panel) return;
      const willOpen = panel.hidden;
      closeMenus();
      panel.hidden = !willOpen;
      btn.setAttribute("aria-expanded", String(willOpen));
    });
  });

  document.addEventListener("click", () => closeMenus());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMenus();
      closeAll();
    }
  });

  const machineTable = document.querySelector("#machine-table");

  const existingAssetIds = () => {
    if (!machineTable) return [];
    return [...machineTable.querySelectorAll("tbody tr")]
      .map((row) => (row.dataset.asset || "").trim().toUpperCase())
      .filter(Boolean);
  };

  const setAssetError = (form, show) => {
    const err = form.querySelector("[data-asset-error]");
    if (err) err.hidden = !show;
    const input = form.querySelector("[data-asset-input]");
    if (input) input.setAttribute("aria-invalid", show ? "true" : "false");
  };

  const isAssetTaken = (assetId, ignoreId = "") => {
    const needle = assetId.trim().toUpperCase();
    if (!needle) return false;
    return existingAssetIds().some((id) => id === needle && id !== ignoreId.trim().toUpperCase());
  };

  const findMachineRow = (id) => {
    if (!machineTable || !id) return null;
    return machineTable.querySelector(`tbody tr[data-id="${id}"]`);
  };

  const fillEditForm = (source) => {
    if (!source) return;
    const form = document.querySelector("#edit-machine-form");
    if (!form) return;
    form.action = `/machines/${source.dataset.id}`;
    form.querySelector("#edit-name").value = source.dataset.name || "";
    form.querySelector("#edit-asset").value = source.dataset.asset || "";
    form.querySelector("#edit-type").value = source.dataset.type || "Bearing";
    form.querySelector("#edit-location").value = source.dataset.location || "";
    form.querySelector("#edit-description").value = source.dataset.description || "";
    setAssetError(form, false);
  };

  document.querySelectorAll("[data-edit-machine]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeMenus();
      const source = btn.closest("tr") || btn.closest(".machine-card") || findMachineRow(btn.dataset.id);
      fillEditForm(source);
      open(".overlay");
      open("#edit-machine");
    });
  });

  document.querySelectorAll("[data-delete-machine]").forEach((btn) => {
    btn.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeMenus();
      const source = btn.closest("tr") || btn.closest(".machine-card");
      const id = source?.dataset.id;
      const name = source?.dataset.name || "this machine";
      if (!id) return;
      if (!window.confirm(`Delete "${name}"? This cannot be undone.`)) return;
      try {
        const response = await fetch(`/machines/${encodeURIComponent(id)}/delete`, {
          method: "POST",
          headers: csrfHeaders({ Accept: "application/json" }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.ok) {
          window.alert(payload.error || "Could not delete the machine.");
          return;
        }
        window.location.reload();
      } catch {
        window.alert("Could not delete the machine.");
      }
    });
  });

  document.querySelectorAll("[data-open]").forEach((el) => {
    el.addEventListener("click", (event) => {
      event.preventDefault();
      const target = el.dataset.open;
      const node = document.querySelector(target);
      if (node?.classList.contains("modal") || node?.classList.contains("drawer") || target.includes("overlay")) {
        open(".overlay");
      }
      if (target === "#add-machine") {
        const form = document.querySelector("#add-machine-form");
        if (form) {
          form.reset();
          setAssetError(form, false);
        }
      }
      open(target);
    });
  });

  document.querySelectorAll("[data-close]").forEach((el) => {
    el.addEventListener("click", closeAll);
  });

  document.querySelectorAll(".drawer").forEach((drawer) => {
    drawer.addEventListener("click", (event) => {
      if (event.target === drawer) closeAll();
    });
  });

  document.querySelectorAll("form[data-ui-only]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const assetInput = form.querySelector("[data-asset-input]");
      if (assetInput) {
        const ignore =
          form.id === "edit-machine-form"
            ? form.querySelector("#edit-original-asset")?.value || ""
            : "";
        if (isAssetTaken(assetInput.value, ignore)) {
          setAssetError(form, true);
          assetInput.focus();
          return;
        }
        setAssetError(form, false);
      }
      closeAll();
    });

    const assetInput = form.querySelector("[data-asset-input]");
    assetInput?.addEventListener("input", () => setAssetError(form, false));
  });

  document.querySelector("#edit-machine-form")?.addEventListener("submit", (event) => {
    const form = event.currentTarget;
    const assetInput = form.querySelector("[data-asset-input]");
    const original = document.querySelector(`[data-id="${form.action.split("/").pop()}"]`)?.dataset.asset || "";
    if (assetInput && isAssetTaken(assetInput.value, original)) {
      event.preventDefault();
      setAssetError(form, true);
    }
  });

  const addForm = document.querySelector("#add-machine-form");
  const drop = document.querySelector("#dataset-drop");
  const fileInput = document.querySelector("#dataset-file");
  const fileName = document.querySelector("#dataset-file-name");
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
  const showFile = () => {
    const file = fileInput?.files?.[0];
    if (fileName) fileName.textContent = file ? file.name : "No file selected";
  };
  fileInput?.addEventListener("change", showFile);
  ["dragover", "dragenter"].forEach((name) => {
    drop?.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  });
  drop?.addEventListener("dragleave", () => drop.classList.remove("is-over"));
  drop?.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("is-over");
    if (fileInput && event.dataTransfer?.files?.length) {
      fileInput.files = event.dataTransfer.files;
      showFile();
    }
  });
  addForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const assetInput = addForm.querySelector("[data-asset-input]");
    if (assetInput && isAssetTaken(assetInput.value, "")) {
      setAssetError(addForm, true);
      return;
    }
    const error = document.querySelector("#upload-error");
    const progress = document.querySelector("#upload-progress");
    const bar = document.querySelector("#upload-bar");
    if (error) error.hidden = true;
    const selected = fileInput?.files?.[0];
    if (selected && selected.size > MAX_UPLOAD_BYTES) {
      if (error) {
        error.hidden = false;
        error.textContent = "That file is larger than the 25 MB upload limit.";
      }
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/machines");
    const token = csrfToken();
    if (token) xhr.setRequestHeader("X-CSRF-Token", token);
    xhr.upload.onprogress = (uploadEvent) => {
      if (!uploadEvent.lengthComputable || !bar || !progress) return;
      progress.hidden = false;
      bar.style.width = `${Math.round((uploadEvent.loaded / uploadEvent.total) * 100)}%`;
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText || "{}"); } catch { body = {}; }
      if (xhr.status >= 200 && xhr.status < 300 && body.redirect) {
        window.location = body.redirect;
        return;
      }
      if (error) {
        error.hidden = false;
        error.textContent = (body.errors || ["Could not add the machine."]).join(" ");
      }
    };
    xhr.onerror = () => {
      if (error) {
        error.hidden = false;
        error.textContent = "Upload failed.";
      }
    };
    xhr.send(new FormData(addForm));
  });

  document.querySelector("[data-edit-notes]")?.addEventListener("click", () => {
    const text = document.querySelector("#machine-notes-text");
    const form = document.querySelector("#notes-form");
    if (text) text.hidden = true;
    if (form) form.hidden = false;
  });
  document.querySelector("[data-cancel-notes]")?.addEventListener("click", () => {
    const text = document.querySelector("#machine-notes-text");
    const form = document.querySelector("#notes-form");
    if (text) text.hidden = false;
    if (form) form.hidden = true;
  });

  const syncDateField = (input) => {
    const field = input.closest(".date-field");
    const label = field?.querySelector("[data-date-placeholder]");
    if (!field || !label) return;
    const placeholder = input.getAttribute("aria-label") || "Date";
    if (input.value) {
      field.classList.add("has-value");
      const [y, m, d] = input.value.split("-");
      label.textContent = `${d}/${m}/${y}`;
    } else {
      field.classList.remove("has-value");
      label.textContent = placeholder;
    }
  };

  document.querySelectorAll("[data-date-input]").forEach((input) => {
    syncDateField(input);
    input.addEventListener("change", () => {
      syncDateField(input);
      const alertsTable = document.querySelector("#alerts-table");
      if (alertsTable && (input.id === "alerts-from" || input.id === "alerts-to")) applyFilters(alertsTable);
    });
    input.addEventListener("input", () => syncDateField(input));
  });

  document.querySelectorAll(".date-field").forEach((field) => {
    const input = field.querySelector("[data-date-input]");
    if (!input) return;
    field.addEventListener("click", () => {
      if (typeof input.showPicker === "function") {
        try {
          input.showPicker();
        } catch {
          input.focus();
        }
      } else {
        input.focus();
      }
    });
  });

  document.querySelector("[data-clear-filters]")?.addEventListener("click", () => {
    const toolbar = document.querySelector("#alerts-toolbar");
    if (!toolbar) return;
    toolbar.querySelectorAll("select").forEach((select) => {
      select.value = "";
      select.dispatchEvent(new Event("change"));
    });
    toolbar.querySelectorAll("input[type='search'], input[type='date']").forEach((input) => {
      input.value = "";
      input.dispatchEvent(new Event("input"));
      if (input.matches("[data-date-input]")) syncDateField(input);
    });
  });

  const alertsTable = document.querySelector("#alerts-table");
  if (alertsTable) {
    const sevClass = {
      Critical: "badge-critical",
      High: "badge-high",
      Medium: "badge-medium",
      Low: "badge-low",
    };
    const statusClass = {
      New: "badge-new",
      Acknowledged: "badge-ack",
      Investigating: "badge-warning",
      Resolved: "badge-resolved",
    };

    const fillAlertDrawer = (row) => {
      alertsTable.querySelectorAll("tbody tr").forEach((r) => r.classList.toggle("is-selected", r === row));
      const sev = row.dataset.severity || "";
      const status = row.dataset.status || "";
      const sevEl = document.querySelector("#alert-drawer-sev");
      const statusEl = document.querySelector("#alert-drawer-status");
      if (sevEl) {
        sevEl.className = `badge ${sevClass[sev] || ""}`;
        sevEl.textContent = sev;
      }
      if (statusEl) {
        statusEl.className = `badge ${statusClass[status] || ""}`;
        statusEl.textContent = status;
      }
      document.querySelector("#alert-drawer-title").textContent = row.dataset.title || "";
      document.querySelector("#alert-drawer-desc").textContent = row.dataset.trigger || "";
      const createdIso = row.dataset.createdIso || row.dataset.created || "";
      const createdAt = createdIso ? new Date(createdIso) : null;
      const createdLabel =
        createdAt && !Number.isNaN(createdAt.getTime())
          ? createdAt.toLocaleString(undefined, {
              day: "2-digit",
              month: "short",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })
          : "—";
      const meta = document.querySelector("#alert-drawer-meta");
      if (meta) {
        meta.replaceChildren();
        [
          ["Machine", `${row.dataset.machine || "—"} (${row.dataset.asset || "—"})`],
          ["Created", createdLabel],
          ["Resolved at", row.dataset.resolved || "—"],
          ["Alert type", row.dataset.type || "—"],
          ["Current value", row.dataset.value || "—"],
          ["Threshold", row.dataset.threshold || "—"],
        ].forEach(([label, value]) => {
          const wrap = document.createElement("div");
          const dt = document.createElement("dt");
          const dd = document.createElement("dd");
          dt.textContent = label;
          dd.textContent = value;
          wrap.append(dt, dd);
          meta.append(wrap);
        });
      }
      const whyList = document.querySelector("#alert-drawer-why");
      if (whyList) {
        whyList.replaceChildren();
        (row.dataset.why || "")
          .split(";")
          .map((item) => item.trim())
          .filter(Boolean)
          .forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item;
            whyList.append(li);
          });
      }
      const ackBtn = document.querySelector("#alert-drawer-ack");
      const resolveBtn = document.querySelector("#alert-drawer-resolve");
      if (ackBtn) {
        ackBtn.hidden = status !== "New";
        ackBtn.dataset.alertKey = row.dataset.alertKey || "";
      }
      if (resolveBtn) {
        resolveBtn.hidden = status === "Resolved";
        resolveBtn.dataset.alertKey = row.dataset.alertKey || "";
      }
      if (row.dataset.machineId) {
        window.updateAlertSparkChart?.(row.dataset.machineId);
      }
    };

    const setRowStatus = (row, status) => {
      if (!row) return;
      row.dataset.status = status;
      const badge = row.querySelector("td:nth-child(6) .badge");
      if (badge) {
        badge.className = `badge ${statusClass[status] || ""}`;
        badge.textContent = status;
      }
      if (status === "Resolved") {
        const now = new Date();
        const label = now.toLocaleString(undefined, {
          day: "2-digit",
          month: "short",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        });
        row.dataset.resolved = label;
        const resolvedCell = row.children[4];
        if (resolvedCell) {
          resolvedCell.textContent = label;
          resolvedCell.classList.remove("muted");
        }
      }
      const menu = row.querySelector(".menu-panel");
      if (menu) {
        menu.querySelector("[data-ack-alert]")?.remove();
        if (status === "Resolved") menu.querySelector("[data-resolve-alert]")?.remove();
      }
      const selected = row.classList.contains("is-selected");
      if (selected) fillAlertDrawer(row);
      applyFilters(alertsTable);
    };

    const postAlertStatus = async (row, status) => {
      if (!row?.dataset.alertKey) return;
      try {
        const response = await fetch("/alerts/status", {
          method: "POST",
          headers: csrfHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
          body: JSON.stringify({ alertKey: row.dataset.alertKey, status }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.ok) return;
        setRowStatus(row, status);
      } catch {
        /* keep current status */
      }
    };

    const alertAction = (event) => {
      const ack = event.target.closest("[data-ack-alert]");
      if (ack) {
        event.preventDefault();
        closeMenus();
        const row =
          ack.closest("tr") ||
          alertsTable.querySelector("tbody tr.is-selected");
        postAlertStatus(row, "Acknowledged");
        return;
      }
      const resolve = event.target.closest("[data-resolve-alert]");
      if (resolve) {
        event.preventDefault();
        closeMenus();
        const row =
          resolve.closest("tr") ||
          alertsTable.querySelector("tbody tr.is-selected");
        postAlertStatus(row, "Resolved");
      }
    };

    alertsTable.addEventListener("click", (event) => {
      const inspect = event.target.closest("[data-inspect-alert]");
      if (inspect) {
        event.preventDefault();
        closeMenus();
        const row = inspect.closest("tr");
        fillAlertDrawer(row);
        open("#alert-drawer");
        return;
      }
      alertAction(event);
    });

    document.querySelector("#alert-drawer")?.addEventListener("click", alertAction);
  }

  const runList = document.querySelector("#run-list");
  if (runList) {
    const runState = { q: "", status: "", year: "", sort: "updated" };

    const applyRunFilters = () => {
      const cards = [...runList.querySelectorAll(".run-card")];
      cards.forEach((card) => {
        const hay = (card.dataset.search || "").toLowerCase();
        const matchQ = !runState.q || hay.includes(runState.q);
        const matchStatus = !runState.status || card.dataset.status === runState.status;
        const matchYear = !runState.year || card.dataset.year === runState.year;
        card.hidden = !(matchQ && matchStatus && matchYear);
      });

      const visible = cards.filter((c) => !c.hidden);
      visible.sort((a, b) => {
        if (runState.sort === "name") return String(a.dataset.name || "").localeCompare(String(b.dataset.name || ""));
        if (runState.sort === "progress") return Number(b.dataset.progress || 0) - Number(a.dataset.progress || 0);
        return String(b.dataset.updated || "").localeCompare(String(a.dataset.updated || ""));
      });
      visible.forEach((card) => runList.appendChild(card));
    };

    document.querySelector("[data-run-search]")?.addEventListener("input", (e) => {
      runState.q = e.target.value.toLowerCase();
      applyRunFilters();
    });

    document.querySelectorAll("[data-run-filter]").forEach((select) => {
      select.addEventListener("change", () => {
        runState[select.dataset.runFilter] = select.value;
        applyRunFilters();
      });
    });

    document.querySelector("[data-run-sort]")?.addEventListener("change", (e) => {
      runState.sort = e.target.value;
      applyRunFilters();
    });

    document.querySelectorAll("[data-run-scope]").forEach((btn) => {
      btn.addEventListener("click", () => {
        runState.scope = btn.dataset.runScope;
        document.querySelectorAll("[data-run-scope]").forEach((b) => b.classList.toggle("btn-primary", b === btn));
        applyRunFilters();
      });
    });

    const fillInspect = (card) => {
      const status = card.dataset.status || "";
      document.querySelector("#inspect-title").textContent = `${card.dataset.name} · ${card.dataset.run}`;
      document.querySelector("#inspect-sub").textContent = `${card.dataset.asset} · ${card.dataset.dataset}`;

      const meta = [
        ["Machine", card.dataset.name],
        ["Asset ID", card.dataset.asset],
        ["Run ID", card.dataset.run],
        ["Dataset", card.dataset.dataset],
        ["Status", status],
        ["Stage", card.dataset.stage],
      ];
      if (card.dataset.test) meta.push(["IMS test", card.dataset.test]);
      if (card.dataset.windows) meta.push(["Operational snapshots", card.dataset.windows]);
      if (card.dataset.bins) meta.push(["12-hour bins", card.dataset.bins]);
      if (card.dataset.period) meta.push(["Period", card.dataset.period]);
      if (card.dataset.health) meta.push(["Latest health", card.dataset.health]);
      document.querySelector("#inspect-meta").innerHTML = meta
        .map(([label, value]) => `<div><dt>${label}</dt><dd>${value || "—"}</dd></div>`)
        .join("");

      const facts = [card.dataset.stage || "Recorded"];
      if (card.dataset.window) facts.push(card.dataset.window);
      if (card.dataset.damage) facts.push(card.dataset.damage);
      else if (card.dataset.test) facts.push("No documented IMS damage label for this bearing.");
      document.querySelector("#inspect-stages").innerHTML = facts
        .map((fact) => `<li class="is-done"><span>${fact}</span><span>${status}</span></li>`)
        .join("");

      const started = card.dataset.updated ? new Date(card.dataset.updated).toLocaleString() : "—";
      document.querySelector("#inspect-times").innerHTML = `
        <div><dt>Run end / update</dt><dd>${started}</dd></div>
        <div><dt>Window summary</dt><dd>${card.dataset.window || "—"}</dd></div>
      `;
      document.querySelector("#inspect-logs").textContent = [
        `Run ${card.dataset.run}`,
        `Asset ${card.dataset.asset}`,
        `Dataset: ${card.dataset.dataset || "—"}`,
        card.dataset.windows ? `Operational snapshots: ${card.dataset.windows}` : null,
        card.dataset.bins ? `12-hour bins: ${card.dataset.bins}` : null,
        `Status: ${status}`,
        card.dataset.error ? `Error: ${card.dataset.error}` : null,
      ].filter(Boolean).join("\n");
      document.querySelector("#inspect-errors").textContent = card.dataset.error || "None";
      const outputs = [];
      if (card.dataset.anomaly) outputs.push(["Latest anomaly score", card.dataset.anomaly]);
      if (card.dataset.rms) outputs.push(["Latest RMS", card.dataset.rms]);
      if (card.dataset.fail) outputs.push(["Latest failure probability", card.dataset.fail]);
      if (card.dataset.rul) outputs.push(["Latest predicted RUL (h)", card.dataset.rul]);
      document.querySelector("#inspect-outputs").innerHTML = outputs.length
        ? outputs.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("")
        : `<div><dt>Model scores</dt><dd>None stored for this run.</dd></div>`;
      const analyticsLink = document.querySelector("#inspect-analytics-link");
      if (analyticsLink) {
        analyticsLink.href = card.dataset.machineId
          ? `/analytics?machine=${encodeURIComponent(card.dataset.machineId)}`
          : "/analytics";
      }
    };

    runList.addEventListener("click", (event) => {
      const inspectBtn = event.target.closest("[data-inspect-run]");
      if (inspectBtn) {
        event.preventDefault();
        closeMenus();
        const card = inspectBtn.closest(".run-card");
        if (!card) return;
        fillInspect(card);
        open("#run-inspect");
        return;
      }

      const monitorBtn = event.target.closest("[data-toggle-monitoring]");
      if (monitorBtn) {
        event.preventDefault();
        const card = monitorBtn.closest(".run-card");
        const machineId = card?.dataset.machineId;
        if (!machineId) return;
        const next = card.dataset.monitoring === "Inactive" ? "Active" : "Inactive";
        fetch(`/machines/${encodeURIComponent(machineId)}/monitoring`, {
          method: "POST",
          headers: csrfHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
          body: JSON.stringify({ status: next }),
        })
          .then((response) => response.json().catch(() => ({})))
          .then((payload) => {
            if (!payload?.ok) return;
            runList.querySelectorAll(".run-card").forEach((item) => {
              if (item.dataset.machineId !== machineId) return;
              item.dataset.monitoring = payload.monitoring;
              const toggle = item.querySelector("[data-toggle-monitoring]");
              if (toggle) {
                toggle.textContent =
                  payload.monitoring === "Inactive" ? "Continue Monitoring" : "Stop Monitoring";
              }
            });
            closeMenus();
          })
          .catch(() => {});
      }
    });

    applyRunFilters();
  }

  const machineSelect = document.getElementById("report-machine");
  const titleInput = document.getElementById("report-title");
  const suggestReportTitle = () => {
    if (!machineSelect || !titleInput) return;
    const option = machineSelect.selectedOptions[0];
    if (!option?.value) return;
    if (titleInput.dataset.locked === "1") return;
    const name = option.dataset.name || option.textContent.trim();
    titleInput.value = `${name} health assessment`.slice(0, 120);
  };
  machineSelect?.addEventListener("change", suggestReportTitle);
  titleInput?.addEventListener("input", () => {
    titleInput.dataset.locked = titleInput.value.trim() ? "1" : "0";
  });
  if (titleInput?.value.trim()) titleInput.dataset.locked = "1";
  document.querySelectorAll("[data-open='#generate-report']").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (titleInput && !titleInput.value.trim()) {
        titleInput.dataset.locked = "0";
        suggestReportTitle();
      }
    });
  });

})();
