// App logic for LLM Proxy Router Dashboard

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    fetchProviders();
    fetchSettings();
    fetchLogs();
});

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
    try {
        const res = await fetch("/api/providers");
        const providers = await res.json();
        renderProviders(providers);
        
        // Update sidebar and playground details
        const active = providers.find(p => p.is_active === 1);
        const activeText = active ? `${active.name} (${active.model_name})` : "None Selected";
        document.getElementById("sidebar-active-provider").innerText = activeText;
        
        const chatStatusText = document.getElementById("chat-active-provider");
        if (chatStatusText) {
            chatStatusText.innerText = active 
                ? `Connected to Active Provider: ${active.name} [${active.model_name}]` 
                : "No Active Provider Selected. Configure one in Providers tab.";
        }
    } catch (err) {
        console.error("Error fetching providers:", err);
    }
}

function renderProviders(providers) {
    const container = document.getElementById("providers-list");
    container.innerHTML = "";
    
    if (providers.length === 0) {
        container.innerHTML = `<div class="glass-panel" style="grid-column: 1/-1; text-align: center; color: var(--text-secondary);">No providers configured. Click "Add Provider" to get started.</div>`;
        return;
    }
    
    providers.forEach(p => {
        const card = document.createElement("div");
        card.className = `provider-card glass-panel ${p.is_active ? 'active-card' : ''}`;
        card.innerHTML = `
            <div class="card-header">
                <div>
                    <h3 class="provider-title">${p.name}</h3>
                    <span class="badge badge-api">${p.api_type}</span>
                </div>
                ${p.is_active ? '<span class="badge badge-active">Active</span>' : ''}
            </div>
            <div class="card-details">
                <div><span>Endpoint:</span> <span class="val">${p.endpoint_url}</span></div>
                <div><span>Routing ID:</span> <span class="val">${p.model_name}</span></div>
                <div><span>API Key:</span> <span class="val">••••••••</span></div>
            </div>
            <div class="card-actions">
                ${!p.is_active ? `<button class="btn btn-primary" onclick="setActiveProvider(${p.id})">Set Active</button>` : ''}
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
        document.getElementById("provider-is-active").checked = provider.is_active === 1;
    } else if (provider) {
        title.innerText = "Add Provider";
        document.getElementById("provider-id").value = "";
        document.getElementById("provider-name").value = provider.name;
        document.getElementById("provider-api-type").value = provider.api_type;
        document.getElementById("provider-endpoint-url").value = provider.endpoint_url;
        document.getElementById("provider-api-key").value = provider.api_key;
        document.getElementById("provider-model-name").value = provider.model_name;
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
    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rate_limit_tps: tps })
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

async function fetchLogs() {
    try {
        const res = await fetch("/api/logs");
        allLogs = await res.json();
        renderLogsList();
    } catch (err) {
        console.error(err);
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
        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.onclick = () => showLogDetail(index, entry);
        entry.innerHTML = `
            <div class="log-meta">
                <span>${date} ${time}</span>
                <span>${log.provider_name || 'Unknown'}</span>
            </div>
            <div class="log-path">${log.request_method} ${log.request_path}</div>
            <div style="margin-top: 4px; display: flex; justify-content: space-between; align-items: center;">
                <span class="log-status ${log.response_status < 400 ? 'status-success' : 'status-error'}">
                    ${log.response_status}
                </span>
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
        try {
            return JSON.stringify(JSON.parse(val), null, 2);
        } catch {
            return val;
        }
    };
    
    detailPane.innerHTML = `
        <h3>Log Inspector</h3>
        <div class="log-detail-content" style="margin-top: 16px;">
            <div class="detail-section">
                <h4>General Details</h4>
                <div class="card-details">
                    <div><span>Timestamp:</span> <span class="val">${new Date(log.timestamp).toLocaleString()}</span></div>
                    <div><span>Provider:</span> <span class="val">${log.provider_name}</span></div>
                    <div><span>Method / Path:</span> <span class="val">${log.request_method} ${log.request_path}</span></div>
                    <div><span>Status Code:</span> <span class="val">${log.response_status}</span></div>
                </div>
            </div>
            <div class="detail-section">
                <h4>Request Body</h4>
                <pre>${formatJSON(log.request_body)}</pre>
            </div>
            <div class="detail-section">
                <h4>Response Body</h4>
                <pre>${formatJSON(log.response_body)}</pre>
            </div>
        </div>
    `;
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

// Model Routing Management
async function fetchRouting() {
    try {
        const res = await fetch("/api/routing");
        const mappings = await res.json();
        renderRouting(mappings);
    } catch (err) {
        console.error("Error fetching routing mappings:", err);
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
