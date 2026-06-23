// App logic for LLM Proxy Router Dashboard

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initMobileDrawer();
    fetchProviders();
    fetchSettings();
    fetchLogs();
});

// Mobile Drawer Navigation
function initMobileDrawer() {
    const toggle = document.getElementById("mobile-nav-toggle");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    function openDrawer() {
        toggle.classList.add("open");
        toggle.setAttribute("aria-expanded", "true");
        sidebar.classList.add("drawer-open");
        overlay.classList.add("visible");
        overlay.setAttribute("aria-hidden", "false");
    }

    function closeDrawer() {
        toggle.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
        sidebar.classList.remove("drawer-open");
        overlay.classList.remove("visible");
        overlay.setAttribute("aria-hidden", "true");
    }

    toggle.addEventListener("click", () => {
        if (sidebar.classList.contains("drawer-open")) {
            closeDrawer();
        } else {
            openDrawer();
        }
    });

    overlay.addEventListener("click", closeDrawer);

    // Close drawer on Escape
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && sidebar.classList.contains("drawer-open")) {
            closeDrawer();
        }
    });

    // Auto-close drawer on tab switch (mobile convenience)
    document.querySelectorAll(".nav-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            if (window.innerWidth <= 768) {
                closeDrawer();
            }
        });
    });
}

// Tab Navigation Logic
function initTabs() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");

            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(pane => pane.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(`tab-${tabId}`).classList.add("active");

            if (tabId === "logs") {
                fetchLogs();
            } else if (tabId === "routing") {
                fetchRouting();
            } else if (tabId === "metrics") {
                updateMetricsCharts();
            }
        });
    });
}

// Suggestions for Endpoint URL depending on API format type
function suggestBaseUrl() {
    const apiType = document.getElementById("provider-api-type").value;
    const urlInput = document.getElementById("provider-endpoint-url");
    if (apiType === "anthropic") {
        urlInput.value = "https://api.anthropic.com/v1/messages";
    } else if (apiType === "gemini") {
        urlInput.value = "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions";
    } else if (apiType === "mistral") {
        urlInput.value = "https://api.mistral.ai/v1/chat/completions";
    } else if (apiType === "embedding") {
        urlInput.value = "https://generativelanguage.googleapis.com/v1beta/openai/embeddings";
    } else if (apiType === "embedding_nvidia_nim") {
        urlInput.value = "https://integrate.api.nvidia.com/v1/embeddings";
    } else if (apiType === "nvidia_nim") {
        urlInput.value = "https://integrate.api.nvidia.com/v1/chat/completions";
    } else {
        urlInput.value = "https://api.openai.com/v1/chat/completions";
    }
}

// Toggle visibility of the API key field
function toggleKeyVisibility() {
    const input = document.getElementById("provider-api-key");
    const toggleBtn = document.getElementById("toggle-key-visibility");
    if (input.type === "password") {
        input.type = "text";
        toggleBtn.innerText = "🔒";
    } else {
        input.type = "password";
        toggleBtn.innerText = "👁";
    }
}

// Fetch and Render Providers
async function fetchProviders() {
    const container = document.getElementById("providers-list");
    // Show shimmer while loading
    container.innerHTML = skeletonProviderCards(3);
    try {
        const [providersRes, metricsRes] = await Promise.all([
            fetch("/api/providers"),
            fetch("/api/metrics")
        ]);
        const providers = await providersRes.json();
        const metrics = await metricsRes.json();

        renderProviders(providers, metrics);

        // Update sidebar active providers
        const activeChat = providers.find(p => p.is_active && p.api_type !== "embedding");
        const activeEmbedding = providers.find(p => p.is_active_embedding);

        document.getElementById("sidebar-active-provider").innerText =
            activeChat ? `${activeChat.name} (${activeChat.model_name})` : "None Selected";
        document.getElementById("sidebar-active-embedding-provider").innerText =
            activeEmbedding ? `${activeEmbedding.name} (${activeEmbedding.model_name})` : "None Selected";

        const chatStatusText = document.getElementById("chat-active-provider");
        if (chatStatusText) {
            chatStatusText.innerText = activeChat
                ? `Connected to Active Provider: ${activeChat.name} [${activeChat.model_name}]`
                : "No Active Provider Selected. Configure one in Providers tab.";
        }
    } catch (err) {
        console.error("Error fetching providers/metrics:", err);
        container.innerHTML = `<div class="glass-panel" style="grid-column:1/-1;text-align:center;color:var(--text-secondary);">Failed to load providers.</div>`;
    }
}

function renderProviders(providers, metrics = []) {
    const container = document.getElementById("providers-list");
    container.innerHTML = "";

    if (providers.length === 0) {
        container.innerHTML = `<div class="glass-panel" style="grid-column: 1/-1; text-align: center; color: var(--text-secondary);">No providers configured. Click "Add Provider" to get started.</div>`;
        return;
    }

    providers.forEach(p => {
        const providerMetrics = metrics.find(m => m.provider_name === p.name) || {};
        const avgLatency = providerMetrics.avg_latency
            ? `${Math.round(providerMetrics.avg_latency)}ms`
            : 'N/A';

        const card = document.createElement("div");
        const isActive = p.api_type === "embedding" ? p.is_active_embedding : p.is_active;
        card.className = `provider-card glass-panel ${isActive ? 'active-card' : ''}`;

        const rateLimitText = p.rate_limit_tps
            ? `${p.rate_limit_tps} TPS`
            : 'Global';

        const maxTokensText = p.max_tokens
            ? `${p.max_tokens.toLocaleString()}`
            : 'Global';

        card.innerHTML = `
            <div class="card-header">
                <div>
                    <h3 class="provider-title">${p.name}</h3>
                    <span class="badge badge-api">${p.api_type}</span>
                </div>
                ${isActive ? `<span class="badge badge-active">${p.api_type === "embedding" ? "Active Embedding" : "Active"}</span>` : ''}
            </div>
            <div class="card-details">
                <div><span>Endpoint:</span> <span class="val">${p.endpoint_url}</span></div>
                <div><span>Routing ID:</span> <span class="val">${p.model_name}</span></div>
                <div><span>Rate Limit:</span> <span class="val">${rateLimitText}</span></div>
                <div><span>Max Tokens:</span> <span class="val">${maxTokensText}</span></div>
                <div><span>Avg Latency:</span> <span class="val">${avgLatency}</span></div>
                <div><span>API Key:</span> <span class="val">••••••••</span></div>
            </div>
            <div class="card-actions">
                ${!isActive ? `<button class="btn btn-primary" onclick="setActiveProvider(${p.id})">Set Active</button>` : ''}
                <button class="btn btn-secondary" onclick="openProviderModal(${JSON.stringify(p).replace(/"/g, '&quot;')})">Edit</button>
                <button class="btn btn-secondary" onclick="duplicateProvider(${JSON.stringify(p).replace(/"/g, '&quot;')})">Duplicate</button>
                <button class="btn btn-danger" onclick="deleteProvider(${p.id})">Delete</button>
            </div>
        `;
        container.appendChild(card);
    });
}

// Set Active Provider
async function setActiveProvider(id) {
    try {
        await fetch(`/api/providers/${id}/active`, { method: "POST" });
        fetchProviders();
        fetchLogs();
    } catch (err) {
        console.error(err);
    }
}

// Delete Provider
async function deleteProvider(id) {
    if (!confirm("Are you sure you want to delete this provider?")) return;
    try {
        await fetch(`/api/providers/${id}`, { method: "DELETE" });
        fetchProviders();
    } catch (err) {
        console.error(err);
    }
}

// Modal handling
function openProviderModal(provider = null) {
    const modal = document.getElementById("provider-modal");
    const form = document.getElementById("provider-form");
    const title = document.getElementById("modal-title");
    
    form.reset();
    
    // Default API Key field to password type and eye icon when opening
    const keyInput = document.getElementById("provider-api-key");
    keyInput.type = "password";
    document.getElementById("toggle-key-visibility").innerText = "👁";
    
    if (provider && provider.id) {
        title.innerText = "Edit Provider";
        document.getElementById("provider-id").value = provider.id;
        document.getElementById("provider-name").value = provider.name;
        document.getElementById("provider-api-type").value = provider.api_type;
        document.getElementById("provider-endpoint-url").value = provider.endpoint_url;
        document.getElementById("provider-api-key").value = provider.api_key;
        document.getElementById("provider-model-name").value = provider.model_name;
        document.getElementById("provider-rate-limit").value = provider.rate_limit_tps || "";
        document.getElementById("provider-max-tokens").value = provider.max_tokens || "";
        document.getElementById("provider-is-active").checked = provider.is_active === 1;
    } else if (provider) {
        title.innerText = "Add Provider";
        document.getElementById("provider-id").value = "";
        document.getElementById("provider-name").value = provider.name;
        document.getElementById("provider-api-type").value = provider.api_type;
        document.getElementById("provider-endpoint-url").value = provider.endpoint_url;
        document.getElementById("provider-api-key").value = provider.api_key;
        document.getElementById("provider-model-name").value = provider.model_name;
        document.getElementById("provider-rate-limit").value = provider.rate_limit_tps || "";
        document.getElementById("provider-max-tokens").value = provider.max_tokens || "";
        document.getElementById("provider-is-active").checked = false;
    } else {
        title.innerText = "Add Provider";
        document.getElementById("provider-id").value = "";
        suggestBaseUrl();
    }
    
    modal.classList.add("open");
}

function closeProviderModal() {
    document.getElementById("provider-modal").classList.remove("open");
}

function duplicateProvider(provider) {
    const copy = { ...provider, name: provider.name + " (copy)" };
    delete copy.id;
    openProviderModal(copy);
}

async function handleProviderSubmit(event) {
    event.preventDefault();
    
    const id = document.getElementById("provider-id").value;
    const provider = {
        name: document.getElementById("provider-name").value,
        api_type: document.getElementById("provider-api-type").value,
        endpoint_url: document.getElementById("provider-endpoint-url").value,
        api_key: document.getElementById("provider-api-key").value,
        model_name: document.getElementById("provider-model-name").value,
        rate_limit_tps: parseFloat(document.getElementById("provider-rate-limit").value) || null,
        max_tokens: parseInt(document.getElementById("provider-max-tokens").value) || null,
        is_active: document.getElementById("provider-is-active").checked ? 1 : 0
    };
    
    const url = id ? `/api/providers/${id}` : "/api/providers";
    const method = id ? "PUT" : "POST";
    
    try {
        const res = await fetch(url, {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(provider)
        });
        if (res.ok) {
            closeProviderModal();
            fetchProviders();
        }
    } catch (err) {
        console.error(err);
    }
}

// Fetch / Save settings
async function fetchSettings() {
    try {
        const res = await fetch("/api/settings");
        const settings = await res.json();
        document.getElementById("log-limit-select").value = settings.log_limit.toString();
        if (document.getElementById("global-rate-limit")) {
            document.getElementById("global-rate-limit").value = settings.rate_limit_tps;
        }
        if (document.getElementById("global-max-tokens")) {
            document.getElementById("global-max-tokens").value = settings.max_tokens;
        }
        return settings;
    } catch (err) {
        console.error(err);
    }
}

async function openGlobalSettingsModal() {
    await fetchSettings();
    document.getElementById("global-settings-modal").classList.add("open");
}

function closeGlobalSettingsModal() {
    document.getElementById("global-settings-modal").classList.remove("open");
}

async function saveGlobalRateLimit() {
    const tps = parseFloat(document.getElementById("global-rate-limit").value) || 0;
    const maxTokens = parseInt(document.getElementById("global-max-tokens").value) || 32000;
    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rate_limit_tps: tps, max_tokens: maxTokens })
        });
        if (res.ok) {
            closeGlobalSettingsModal();
        }
    } catch (err) {
        console.error(err);
    }
}

async function saveSettings() {
    const limit = parseInt(document.getElementById("log-limit-select").value);
    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ log_limit: limit })
        });
        if (res.ok) {
            alert("Settings saved successfully.");
            fetchLogs();
        }
    } catch (err) {
        console.error(err);
    }
}

// Logs fetching and inspection
let allLogs = [];

// ─── Shimmer skeleton helpers ───────────────────────────────────────────────

function skeletonLogEntries(count = 4) {
    return Array.from({ length: count }, () => `
        <div class="skeleton-log-entry">
            <div style="display:flex;justify-content:space-between;">
                <div class="skeleton-line short shimmer"></div>
                <div class="skeleton-line short shimmer" style="width:30%;"></div>
            </div>
            <div class="skeleton-line medium shimmer"></div>
            <div class="skeleton-line short shimmer" style="width:20%;"></div>
        </div>
    `).join("");
}

function skeletonProviderCards(count = 2) {
    return Array.from({ length: count }, () => `
        <div class="skeleton-provider-card">
            <div style="display:flex;justify-content:space-between;">
                <div class="skeleton-line medium shimmer" style="height:18px;"></div>
                <div class="skeleton-line shimmer" style="width:60px;height:18px;"></div>
            </div>
            <div class="skeleton-line long shimmer"></div>
            <div class="skeleton-line medium shimmer"></div>
            <div class="skeleton-line long shimmer"></div>
            <div class="skeleton-line medium shimmer"></div>
            <div style="display:flex;gap:8px;margin-top:4px;">
                <div class="skeleton-line shimmer" style="width:80px;height:32px;border-radius:6px;"></div>
                <div class="skeleton-line shimmer" style="width:60px;height:32px;border-radius:6px;"></div>
            </div>
        </div>
    `).join("");
}

function skeletonRoutingRows(count = 3) {
    return Array.from({ length: count }, () => `
        <tr class="skeleton-table-row">
            <td><div class="skeleton-line medium shimmer" style="height:14px;"></div></td>
            <td><div class="skeleton-line short shimmer" style="height:14px;"></div></td>
            <td><div class="skeleton-line shimmer" style="width:60px;height:28px;border-radius:6px;"></div></td>
        </tr>
    `).join("");
}

function skeletonMetrics() {
    return `
        <div class="skeleton-chart shimmer"></div>
    `;
}

// ─── fetchLogs with shimmer ──────────────────────────────────────────────────

async function fetchLogs() {
    const historyContainer = document.getElementById("logs-history");
    // Show shimmer while loading
    historyContainer.innerHTML = skeletonLogEntries(5);
    try {
        const res = await fetch("/api/logs");
        allLogs = await res.json();
        renderLogsList();
    } catch (err) {
        console.error(err);
        historyContainer.innerHTML = `<div style="text-align:center;color:var(--text-secondary);margin-top:20px;">Failed to load logs.</div>`;
    }
}

function renderLogsList() {
    const historyContainer = document.getElementById("logs-history");
    historyContainer.innerHTML = "";
    
    if (allLogs.length === 0) {
        historyContainer.innerHTML = `<div style="text-align: center; color: var(--text-secondary); margin-top: 20px;">No logs recorded.</div>`;
        document.getElementById("log-detail").innerHTML = `<h3>Log Inspector</h3><div class="no-log-selected">Select a log entry to view full details.</div>`;
        return;
    }
    
    allLogs.forEach((log, index) => {
        const time = new Date(log.timestamp).toLocaleTimeString();
        const date = new Date(log.timestamp).toLocaleDateString();
        const isPending = !log.response_status || log.response_status === 0;
        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.onclick = () => showLogDetail(index, entry);

        const stageCount = (log.events || []).length;
        const stageLabel = isPending
            ? `<span class="log-status timeline-status pending">⏳ stage ${stageCount}/4</span>`
            : `<span class="log-status ${log.response_status < 400 ? 'status-success' : 'status-error'}">${log.response_status}</span>`;

        entry.innerHTML = `
            <div class="log-meta">
                <span>${date} ${time}</span>
                <span>${log.provider_name || 'Unknown'}</span>
            </div>
            <div class="log-path">${log.request_method} ${log.request_path}</div>
            <div style="margin-top: 4px; display: flex; justify-content: space-between; align-items: center;">
                ${stageLabel}
            </div>
        `;
        historyContainer.appendChild(entry);
    });
}

function showLogDetail(index, element) {
    // Remove active class from previous
    document.querySelectorAll(".log-entry").forEach(el => el.classList.remove("active-entry"));
    element.classList.add("active-entry");

    const log = allLogs[index];
    const detailPane = document.getElementById("log-detail");

    // Pretty print json helper
    const formatJSON = (val) => {
        if (!val) return "(empty)";
        try {
            return JSON.stringify(JSON.parse(val), null, 2);
        } catch {
            return val;
        }
    };

    // Helper to create a collapsible pre block
    const collapsiblePre = (content) => {
        const wrapper = document.createElement("div");
        const pre = document.createElement("pre");
        pre.textContent = content;
        wrapper.appendChild(pre);

        // Check if content is long enough to warrant a toggle
        const lineCount = content.split("\n").length;
        if (lineCount > 10 || content.length > 500) {
            pre.style.maxHeight = "150px";
            const toggle = document.createElement("button");
            toggle.className = "log-pre-toggle";
            toggle.textContent = "Show more";
            toggle.addEventListener("click", () => {
                if (pre.style.maxHeight !== "none") {
                    pre.style.maxHeight = "none";
                    toggle.textContent = "Show less";
                } else {
                    pre.style.maxHeight = "150px";
                    toggle.textContent = "Show more";
                    pre.scrollIntoView({ behavior: "smooth", block: "nearest" });
                }
            });
            wrapper.appendChild(toggle);
        }

        return wrapper;
    };

    // Stage display metadata
    const STAGE_META = {
        router_received:   { label: "1 · Router Received",       icon: "R" },
        provider_request:  { label: "2 · Sent to Provider",      icon: "→" },
        provider_response: { label: "3 · Provider Response",     icon: "←" },
        client_response:   { label: "4 · Response to Client",    icon: "C" },
    };

    // Build detail content
    const contentDiv = document.createElement("div");
    contentDiv.className = "log-detail-content";
    contentDiv.style.marginTop = "16px";

    // General Details
    const isPending = !log.response_status || log.response_status === 0;
    const generalSection = document.createElement("div");
    generalSection.className = "detail-section";
    generalSection.innerHTML = `<h4>General Details</h4>
        <div class="card-details">
            <div><span>Timestamp:</span> <span class="val">${new Date(log.timestamp).toLocaleString()}</span></div>
            <div><span>Provider:</span> <span class="val">${log.provider_name}</span></div>
            <div><span>Method / Path:</span> <span class="val">${log.request_method} ${log.request_path}</span></div>
            <div><span>Status Code:</span> <span class="val">${isPending ? '⏳ in progress' : log.response_status}</span></div>
            ${log.latency_ms ? `<div><span>Latency:</span> <span class="val">${log.latency_ms} ms</span></div>` : ''}
            ${log.tokens_sent ? `<div><span>Tokens (sent / recv):</span> <span class="val">${log.tokens_sent} / ${log.tokens_received}</span></div>` : ''}
        </div>`;
    contentDiv.appendChild(generalSection);

    // Lifecycle Timeline
    const events = log.events || [];
    if (events.length > 0) {
        const timelineSection = document.createElement("div");
        timelineSection.className = "detail-section";
        timelineSection.innerHTML = `<h4>Request Lifecycle</h4>`;

        const timeline = document.createElement("div");
        timeline.className = "log-timeline";

        events.forEach(evt => {
            const meta = STAGE_META[evt.stage] || { label: evt.stage, icon: "•" };
            const time = new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 });

            let statusBadge = "";
            if (evt.status_code != null) {
                const cls = evt.status_code < 400 ? "ok" : "err";
                statusBadge = `<span class="timeline-status ${cls}">${evt.status_code}</span>`;
            } else if (isPending && evt.stage === events[events.length - 1].stage) {
                statusBadge = `<span class="timeline-status pending">pending</span>`;
            }

            const stageEl = document.createElement("div");
            stageEl.className = `timeline-stage ${evt.stage}`;
            stageEl.innerHTML = `
                <div class="timeline-dot stage-${evt.stage}">${meta.icon}</div>
                <div class="timeline-body">
                    <div class="timeline-header">
                        <span class="timeline-stage-name">${meta.label}${statusBadge}</span>
                        <span class="timeline-time">${time}</span>
                    </div>
                </div>
            `;

            // Attach collapsible body if there's content
            if (evt.body && evt.body.trim()) {
                const bodyWrapper = stageEl.querySelector(".timeline-body");
                const bodySection = document.createElement("div");
                bodySection.className = "detail-section";
                bodySection.style.marginTop = "6px";
                bodySection.appendChild(collapsiblePre(formatJSON(evt.body)));
                bodyWrapper.appendChild(bodySection);
            }

            timeline.appendChild(stageEl);
        });

        // If pending, add a placeholder for remaining stages
        const knownStages = ["router_received", "provider_request", "provider_response", "client_response"];
        const doneStages = new Set(events.map(e => e.stage));
        knownStages.filter(s => !doneStages.has(s)).forEach(s => {
            const meta = STAGE_META[s] || { label: s, icon: "•" };
            const stageEl = document.createElement("div");
            stageEl.className = `timeline-stage ${s}`;
            stageEl.style.opacity = "0.35";
            stageEl.innerHTML = `
                <div class="timeline-dot stage-${s}">${meta.icon}</div>
                <div class="timeline-body">
                    <div class="timeline-header">
                        <span class="timeline-stage-name">${meta.label}</span>
                        <span class="timeline-time">waiting…</span>
                    </div>
                </div>
            `;
            timeline.appendChild(stageEl);
        });

        timelineSection.appendChild(timeline);
        contentDiv.appendChild(timelineSection);
    } else {
        // Legacy log — no events, fall back to flat view
        const reqSection = document.createElement("div");
        reqSection.className = "detail-section";
        reqSection.innerHTML = `<h4>Request Body</h4>`;
        reqSection.appendChild(collapsiblePre(formatJSON(log.request_body)));
        contentDiv.appendChild(reqSection);

        const resSection = document.createElement("div");
        resSection.className = "detail-section";
        resSection.innerHTML = `<h4>Response Body</h4>`;
        resSection.appendChild(collapsiblePre(formatJSON(log.response_body)));
        contentDiv.appendChild(resSection);
    }

    detailPane.innerHTML = `<h3>Log Inspector</h3>`;
    detailPane.appendChild(contentDiv);
}

async function clearLogs() {
    if (!confirm("Are you sure you want to clear all logs from the database?")) return;
    try {
        await fetch("/api/logs", { method: "DELETE" });
        fetchLogs();
    } catch (err) {
        console.error(err);
    }
}

// Chat Testbed Logic
async function sendChatMessage() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;
    
    appendChatMessage(message, "user");
    input.value = "";
    
    const loadingMessage = appendChatMessage("Typing...", "assistant");
    
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });
        
        const data = await res.json();
        loadingMessage.remove();
        
        if (res.ok) {
            appendChatMessage(data.response, "assistant");
        } else {
            appendChatMessage(`Error: ${data.detail || 'Failed to fetch response'}`, "assistant");
        }
    } catch (err) {
        loadingMessage.remove();
        appendChatMessage(`Error: ${err.message}`, "assistant");
    }
}

function appendChatMessage(text, sender) {
    const chatContainer = document.getElementById("chat-messages");
    const msg = document.createElement("div");
    msg.className = `chat-message ${sender}`;
    msg.innerText = text;

    // For assistant messages, check if long enough to collapse
    if (sender === "assistant" && text.length > 300) {
        msg.classList.add("collapsed");
        const expandBtn = document.createElement("button");
        expandBtn.className = "chat-expand-btn";
        expandBtn.textContent = "Show more";
        expandBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (msg.classList.contains("collapsed")) {
                msg.classList.remove("collapsed");
                expandBtn.textContent = "Show less";
            } else {
                msg.classList.add("collapsed");
                expandBtn.textContent = "Show more";
                // Scroll back to message top so it's visible after collapse
                msg.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        });
        msg.appendChild(expandBtn);
    }

    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msg;
}

function handleChatKey(event) {
    if (event.key === "Enter") {
        sendChatMessage();
    }
}

function clearChat() {
    const chatContainer = document.getElementById("chat-messages");
    chatContainer.innerHTML = `
        <div class="chat-message assistant">
            Hello! I am connected to the active proxy provider. Send a message to test the response.
        </div>
    `;
}

// Embedding Tab Logic
const EMBEDDING_ENDPOINT = "/v1/embeddings";
let embeddingSelectedFile = null;

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        embeddingSelectedFile = file;
        document.getElementById("file-upload-text").textContent = file.name;
        document.getElementById("embedding-input-text").placeholder = "File selected — click Generate to embed";
    }
}

function handleFileDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove("drag-over");
    const file = event.dataTransfer.files[0];
    if (file) {
        embeddingSelectedFile = file;
        document.getElementById("file-upload-text").textContent = file.name;
        document.getElementById("embedding-input-text").placeholder = "File selected — click Generate to embed";
    }
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

async function sendEmbeddingRequest() {
    const textInput = document.getElementById("embedding-input-text");
    const modelInput = document.getElementById("embedding-model");
    const resultDiv = document.getElementById("embedding-result");
    const statsDiv = document.getElementById("embedding-stats");
    const jsonDiv = document.getElementById("embedding-raw-json");

    let inputData;
    if (textInput.value.trim()) {
        inputData = textInput.value.trim();
    } else if (embeddingSelectedFile) {
        try {
            inputData = await readFileAsText(embeddingSelectedFile);
        } catch {
            alert("Failed to read file.");
            return;
        }
    } else {
        alert("Please enter text or upload a file.");
        return;
    }

    const model = modelInput.value.trim();

    const btn = document.querySelector(".btn-embedding");
    btn.disabled = true;
    btn.textContent = "Generating...";

    const reqBody = { input: inputData };
    if (model) reqBody.model = model;
    
    reqBody.input_type = "passage";
    reqBody.truncate = "NONE";

    try {
        const res = await fetch(EMBEDDING_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody)
        });
        const data = await res.json();
        btn.disabled = false;
        btn.textContent = "Generate Embeddings";

        if (!res.ok) {
            statsDiv.innerHTML = `<div class="embedding-error">${data.detail || "Request failed"}</div>`;
            jsonDiv.textContent = JSON.stringify(data, null, 2);
            resultDiv.style.display = "block";
            return;
        }

        // Render stats
        const vectorCount = data.data ? data.data.length : 0;
        const dimensions = (data.data && data.data[0] && data.data[0].embedding)
            ? data.data[0].embedding.length : 0;
        const promptTokens = data.usage ? data.usage.prompt_tokens : "N/A";
        const totalTokens = data.usage ? data.usage.total_tokens : "N/A";

        statsDiv.innerHTML = `
            <div class="embedding-stat-grid">
                <div class="embedding-stat"><span class="stat-label">Vectors</span><span class="stat-value">${vectorCount}</span></div>
                <div class="embedding-stat"><span class="stat-label">Dimensions</span><span class="stat-value">${dimensions}</span></div>
                <div class="embedding-stat"><span class="stat-label">Prompt Tokens</span><span class="stat-value">${promptTokens}</span></div>
                <div class="embedding-stat"><span class="stat-label">Total Tokens</span><span class="stat-value">${totalTokens}</span></div>
                <div class="embedding-stat"><span class="stat-label">Model</span><span class="stat-value">${data.model || model}</span></div>
            </div>
        `;

        jsonDiv.textContent = JSON.stringify(data, null, 2);
        resultDiv.style.display = "block";
    } catch (err) {
        btn.disabled = false;
        btn.textContent = "Generate Embeddings";
        statsDiv.innerHTML = `<div class="embedding-error">Error: ${err.message} — Is the embedding server running?</div>`;
        jsonDiv.textContent = "";
        resultDiv.style.display = "block";
    }
}

// Model Routing Management
async function fetchRouting() {
    const container = document.getElementById("routing-mappings-list");
    container.innerHTML = skeletonRoutingRows(3);
    try {
        const res = await fetch("/api/routing");
        const mappings = await res.json();
        renderRouting(mappings);
    } catch (err) {
        console.error("Error fetching routing mappings:", err);
        container.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--text-secondary);padding:20px;">Failed to load routing mappings.</td></tr>`;
    }
}

function renderRouting(mappings) {
    const container = document.getElementById("routing-mappings-list");
    container.innerHTML = "";

    if (mappings.length === 0) {
        container.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--text-secondary); padding: 20px;">No custom routing mappings configured.</td></tr>`;
        return;
    }

    mappings.forEach(m => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td class="val-cell">${m.model_id}</td>
            <td class="val-cell">${m.provider_name}</td>
            <td class="action-cell">
                <button class="btn btn-danger btn-sm" onclick="deleteRoutingMapping('${m.model_id}')">Delete</button>
            </td>
        `;
        container.appendChild(row);
    });
}

function openRoutingModal() {
    const modal = document.getElementById("routing-modal");
    const providerSelect = document.getElementById("routing-provider-id");

    // Populate provider dropdown
    fetch("/api/providers")
        .then(res => res.json())
        .then(providers => {
            providerSelect.innerHTML = '<option value="" disabled selected>Select a provider</option>';
            providers.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.innerText = p.name;
                providerSelect.appendChild(opt);
            });
        });

    modal.classList.add("open");
}

function closeRoutingModal() {
    document.getElementById("routing-modal").classList.remove("open");
}

async function handleRoutingSubmit(event) {
    event.preventDefault();

    const mapping = {
        model_id: document.getElementById("routing-model-id").value,
        provider_id: parseInt(document.getElementById("routing-provider-id").value)
    };

    try {
        const res = await fetch("/api/routing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(mapping)
        });
        if (res.ok) {
            closeRoutingModal();
            fetchRouting();
            document.getElementById("routing-form").reset();
        }
    } catch (err) {
        console.error(err);
    }
}

async function deleteRoutingMapping(modelId) {
    if (!confirm(`Are you sure you want to delete the routing for ${modelId}?`)) return;
    try {
        const res = await fetch(`/api/routing/${modelId}`, { method: "DELETE" });
        if (res.ok) {
            fetchRouting();
        }
    } catch (err) {
        console.error(err);
    }
}

// Metrics Charting Logic
let metricsCharts = {};

const CHART_COLORS = [
    '#4f46e5', // Indigo
    '#7c3aed', // Purple
    '#2563eb', // Blue
    '#db2777', // Pink
    '#ea580c', // Orange
    '#16a34a', // Green
    '#0891b2', // Cyan
    '#eab308', // Yellow
    '#dc2626', // Red
    '#4ade80', // Light Green
    '#f472b6', // Light Pink
    '#fb923c'  // Light Orange
];

function getProviderColor(name, opacity = 1) {
    // Deterministic color selection based on name hash
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const index = Math.abs(hash) % CHART_COLORS.length;
    const color = CHART_COLORS[index];
    
    if (opacity < 1) {
        // Simple hex to rgba conversion
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }
    return color;
}

async function updateMetricsCharts() {
    // Show shimmer in each chart container while loading
    ["chart-latency-history", "chart-requests", "chart-tokens", "chart-latency"].forEach(id => {
        const canvas = document.getElementById(id);
        if (canvas) {
            const parent = canvas.parentElement;
            if (parent) {
                parent.innerHTML = skeletonMetrics();
                // Restore canvas so Chart.js can redraw after data arrives
                const newCanvas = document.createElement("canvas");
                newCanvas.id = id;
                parent.appendChild(newCanvas);
            }
        }
    });

    try {
        const [summaryRes, historyRes] = await Promise.all([
            fetch("/api/metrics"),
            fetch("/api/metrics/history")
        ]);
        
        const summaryData = await summaryRes.json();
        const historyData = await historyRes.json();
        
        renderMetricsCharts(summaryData, historyData);
    } catch (err) {
        console.error("Error updating metrics charts:", err);
    }
}

function renderMetricsCharts(summaryData, historyData) {
    const labels = summaryData.map(m => m.provider_name);
    const requestCounts = summaryData.map(m => m.request_count);
    const tokensSent = summaryData.map(m => m.total_tokens_sent);
    const tokensReceived = summaryData.map(m => m.total_tokens_received);
    const avgLatencies = summaryData.map(m => m.avg_latency);

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#a0a0a0', font: { family: 'Inter' } }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255,255,255,0.1)' },
                ticks: { color: '#a0a0a0', font: { family: 'Inter' } }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#a0a0a0', font: { family: 'Inter' } }
            }
        }
    };

    // Latency History (Line Chart)
    if (metricsCharts.history) metricsCharts.history.destroy();
    
    // Group history by provider
    const providers = [...new Set(historyData.map(h => h.provider_name))];
    const datasets = providers.map((p) => {
        const providerData = historyData.filter(h => h.provider_name === p);
        const color = getProviderColor(p);
        return {
            label: p,
            data: providerData.map(h => ({ x: new Date(h.timestamp).getTime(), y: h.latency_ms })),
            borderColor: color,
            backgroundColor: getProviderColor(p, 0.1),
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2,
            fill: false
        };
    });

    metricsCharts.history = new Chart(document.getElementById("chart-latency-history"), {
        type: 'line',
        data: { datasets },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                x: {
                    type: 'linear',
                    grid: { display: false },
                    ticks: {
                        color: '#a0a0a0',
                        maxRotation: 0,
                        callback: function(val) {
                            const date = new Date(val);
                            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                        }
                    }
                }
            }
        }
    });

    // Request Volume Chart
    if (metricsCharts.requests) metricsCharts.requests.destroy();
    metricsCharts.requests = new Chart(document.getElementById("chart-requests"), {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: requestCounts,
                backgroundColor: labels.map(name => getProviderColor(name)),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#a0a0a0', font: { family: 'Inter' } } }
            }
        }
    });

    // Token Usage Chart (Stacked Bar)
    if (metricsCharts.tokens) metricsCharts.tokens.destroy();
    metricsCharts.tokens = new Chart(document.getElementById("chart-tokens"), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Sent',
                    data: tokensSent,
                    backgroundColor: '#4f46e5',
                    borderRadius: 4
                },
                {
                    label: 'Received',
                    data: tokensReceived,
                    backgroundColor: '#7c3aed',
                    borderRadius: 4
                }
            ]
        },
        options: {
            ...commonOptions,
            scales: {
                x: { ...commonOptions.scales.x, stacked: true },
                y: { ...commonOptions.scales.y, stacked: true }
            }
        }
    });

    // Average Latency Chart
    if (metricsCharts.latency) metricsCharts.latency.destroy();
    metricsCharts.latency = new Chart(document.getElementById("chart-latency"), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Latency (ms)',
                data: avgLatencies,
                backgroundColor: labels.map(name => getProviderColor(name)),
                borderRadius: 4
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                legend: { display: false } // Hide legend for single-dataset bar chart
            }
        }
    });
}
