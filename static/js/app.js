/* ── SmartContainer Risk Engine — Frontend Logic ─────────────────────────── */

const API = '';
let currentPage = 1;
let currentLevel = 'ALL';
let currentSearch = '';
let map = null;
let mapMarkers = [];

// ── Navigation ──────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
        e.preventDefault();
        navigateTo(item.dataset.page);
    });
});

function navigateTo(page) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (navItem) navItem.classList.add('active');

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');

    if (page === 'dashboard' && !map) setTimeout(initMap, 100);
    if (page === 'predictions') loadPredictions();
    if (page === 'features') loadFeatureAnalysis();
    if (page === 'metrics') loadModelMetrics();
}

// ── Time ────────────────────────────────────────────────────────────────────
function updateTime() {
    const now = new Date();
    const el = document.getElementById('topbar-time');
    if (el) el.textContent = now.toLocaleString('en-US', {
        month:'short', day:'numeric', year:'numeric',
        hour:'2-digit', minute:'2-digit', second:'2-digit'
    });
}
setInterval(updateTime, 1000);
updateTime();

// ── Init ────────────────────────────────────────────────────────────────────
async function init() {
    await Promise.all([loadStats(), loadCharts(), loadMapData()]);
    loadConfig();
}

// ── KPI Strip ───────────────────────────────────────────────────────────────
async function loadStats() {
    const res = await fetch(`${API}/api/stats`);
    const d = await res.json();

    document.getElementById('kpi-strip').innerHTML = `
        <div class="kpi-card">
            <div class="kpi-icon blue">📦</div>
            <div class="kpi-info">
                <span class="kpi-value">${d.total.toLocaleString()}</span>
                <span class="kpi-label">Total Containers</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon red">⚠</div>
            <div class="kpi-info">
                <span class="kpi-value text-red">${d.critical.toLocaleString()}</span>
                <span class="kpi-label">Critical</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon yellow">◎</div>
            <div class="kpi-info">
                <span class="kpi-value text-yellow">${d.low_risk.toLocaleString()}</span>
                <span class="kpi-label">Low Risk</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon green">✔</div>
            <div class="kpi-info">
                <span class="kpi-value text-green">${d.clear.toLocaleString()}</span>
                <span class="kpi-label">Cleared</span>
            </div>
        </div>
    `;
}

// ── Charts ──────────────────────────────────────────────────────────────────
let donutChart = null, histChart = null;

async function loadCharts() {
    const res = await fetch(`${API}/api/charts`);
    const d = await res.json();

    // Donut
    const donutCtx = document.getElementById('donutChart').getContext('2d');
    if (donutChart) donutChart.destroy();
    donutChart = new Chart(donutCtx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'Low Risk', 'Clear'],
            datasets: [{
                data: [d.distribution.Critical, d.distribution['Low Risk'], d.distribution.Clear],
                backgroundColor: ['#f85149', '#d29922', '#3fb950'],
                borderWidth: 0, hoverOffset: 8
            }]
        },
        options: {
            cutout: '60%',
            plugins: {
                legend: { position: 'bottom', labels: { color: '#8b949e', padding: 10, font: { size: 10 } } }
            },
            responsive: true, maintainAspectRatio: false
        }
    });

    // Histogram
    const histCtx = document.getElementById('histChart').getContext('2d');
    const bands = ['0-10','10-20','20-40','40-60','60-80','80-100'];
    const colors = ['#3fb950','#2ea043','#d29922','#e3b341','#f85149','#da3633'];
    if (histChart) histChart.destroy();
    histChart = new Chart(histCtx, {
        type: 'bar',
        data: {
            labels: bands,
            datasets: [{
                data: bands.map(b => d.histogram[b] || 0),
                backgroundColor: colors,
                borderRadius: 4, borderSkipped: false
            }]
        },
        options: {
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => ctx.raw.toLocaleString() + ' containers' } }
            },
            scales: {
                x: { ticks: { color: '#8b949e', font: { size: 9 } }, grid: { display: false } },
                y: { ticks: { color: '#8b949e', font: { size: 9 }, callback: v => v >= 1000 ? (v/1000)+'K' : v }, grid: { color: '#21262d' } }
            },
            responsive: true, maintainAspectRatio: false
        }
    });

    // Critical Origins bars
    renderBarChart('critical-origins', d.critical_origins, 'red');

    // Feature Importance
    const featData = {};
    d.feature_importance.forEach(f => featData[f.name] = f.value);
    renderBarChart('feat-importance', featData, 'blue');

    // High-Risk Ports
    renderBarChart('high-risk-ports', d.high_risk_ports, 'gradient');

    // Model Performance
    document.getElementById('model-perf').innerHTML = `
        <div class="metric-row"><span class="metric-label">Overall Accuracy</span><span class="metric-value text-blue">99.82%</span></div>
        <div class="metric-row"><span class="metric-label">F1 — Critical</span><span class="metric-value text-red">96.68%</span></div>
        <div class="metric-row"><span class="metric-label">F1 — Low Risk</span><span class="metric-value text-yellow">99.62%</span></div>
        <div class="metric-row"><span class="metric-label">Test Split</span><span class="metric-value">80 / 20 %</span></div>
    `;
}

function renderBarChart(containerId, data, colorClass) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const entries = Object.entries(data);
    if (!entries.length) { el.innerHTML = '<p class="text-muted">No data</p>'; return; }
    const max = Math.max(...entries.map(e => e[1]));
    el.innerHTML = entries.map(([label, val]) => {
        const pct = max > 0 ? (val / max * 100) : 0;
        const display = typeof val === 'number' && val % 1 !== 0 ? val.toFixed(1) : val;
        return `<div class="bar-row">
            <span class="bar-label">${label}</span>
            <div class="bar-track"><div class="bar-fill ${colorClass}" style="width:${pct}%"></div></div>
            <span class="bar-val">${display}</span>
        </div>`;
    }).join('');
}

// ── Map ─────────────────────────────────────────────────────────────────────
async function loadMapData() {
    const res = await fetch(`${API}/api/map_data`);
    window._mapData = await res.json();
    if (map) renderMapMarkers();
}

function initMap() {
    if (map) return;
    map = L.map('map', {
        center: [25, 40],
        zoom: 2,
        minZoom: 2,
        maxZoom: 6,
        attributionControl: false,
        zoomControl: true
    });

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    if (window._mapData) renderMapMarkers();
}

function renderMapMarkers() {
    mapMarkers.forEach(m => map.removeLayer(m));
    mapMarkers = [];

    window._mapData.forEach(d => {
        const radius = Math.max(8, Math.min(35, Math.sqrt(d.total) * 1.8));
        let color = '#3fb950';
        if (d.critical > 0) color = '#f85149';
        else if (d.low_risk > 0) color = '#d29922';

        const circle = L.circleMarker([d.lat, d.lng], {
            radius: radius,
            fillColor: color,
            fillOpacity: 0.5,
            color: color,
            weight: 1,
            opacity: 0.7
        }).addTo(map);

        // Show popup on hover
        circle.bindPopup(`
            <div class="popup-content">
                <div class="popup-title">${d.country}</div>
                <div class="popup-row"><span class="popup-label">Total</span><span class="popup-val">${d.total.toLocaleString()}</span></div>
                <div class="popup-row"><span class="popup-label">Critical</span><span class="popup-val" style="color:#f85149">${d.critical}</span></div>
                <div class="popup-row"><span class="popup-label">Low Risk</span><span class="popup-val" style="color:#d29922">${d.low_risk}</span></div>
                <div class="popup-row"><span class="popup-label">Clear</span><span class="popup-val" style="color:#3fb950">${d.clear}</span></div>
            </div>
        `, { closeButton: true, className: 'dark-popup' });

        circle.on('mouseover', function() {
            this.setStyle({ fillOpacity: 0.8, weight: 2, opacity: 1 });
            this.openPopup();
        });
        circle.on('mouseout', function() {
            this.setStyle({ fillOpacity: 0.5, weight: 1, opacity: 0.7 });
        });

        // Pulse animation for critical
        if (d.critical > 2) {
            const pulse = L.circleMarker([d.lat, d.lng], {
                radius: radius + 8,
                fillColor: '#f85149',
                fillOpacity: 0,
                color: '#f85149',
                weight: 2,
                opacity: 0.4,
                className: 'pulse-marker'
            }).addTo(map);
            mapMarkers.push(pulse);
        }

        mapMarkers.push(circle);
    });
}

// ── Predictions ─────────────────────────────────────────────────────────────
async function loadPredictions() {
    const res = await fetch(`${API}/api/predictions?level=${currentLevel}&search=${encodeURIComponent(currentSearch)}&page=${currentPage}`);
    const d = await res.json();

    const tbody = document.getElementById('pred-tbody');
    tbody.innerHTML = d.rows.map(r => {
        const scoreColor = r.risk_score >= 50 ? 'var(--red)' : r.risk_score >= 20 ? 'var(--yellow)' : 'var(--text)';
        const barColor = r.risk_score >= 50 ? '#f85149' : r.risk_score >= 20 ? '#d29922' : '#3fb950';
        const levelClass = r.risk_level === 'Critical' ? 'level-critical' : r.risk_level === 'Low Risk' ? 'level-low' : 'level-clear';
        const dwellClass = r.dwell > 120 ? 'dwell-high' : '';

        return `<tr>
            <td>${r.container_id}</td>
            <td><div class="score-cell">
                <span class="score-num" style="color:${scoreColor}">${r.risk_score}</span>
                <div class="score-bar"><div class="score-bar-fill" style="width:${r.risk_score}%;background:${barColor}"></div></div>
            </div></td>
            <td><span class="level-badge ${levelClass}">${r.risk_level.toUpperCase()}</span></td>
            <td>${r.origin}</td>
            <td>${r.destination}</td>
            <td class="${dwellClass}">${r.dwell}</td>
            <td>${r.value}</td>
            <td class="explanation-cell">${r.explanation}</td>
        </tr>`;
    }).join('');

    // Pagination
    const start = (d.page - 1) * d.per_page + 1;
    const end = Math.min(d.page * d.per_page, d.total_rows);
    let pagHTML = `<span class="info">Showing ${start}–${end} of ${d.total_rows.toLocaleString()} rows</span>`;
    pagHTML += `<button class="page-btn" onclick="goPage(${d.page-1})" ${d.page<=1?'disabled':''}>←</button>`;

    const maxBtns = 5;
    let startP = Math.max(1, d.page - 2);
    let endP = Math.min(d.total_pages, startP + maxBtns - 1);
    if (endP - startP < maxBtns - 1) startP = Math.max(1, endP - maxBtns + 1);

    if (startP > 1) pagHTML += `<button class="page-btn" onclick="goPage(1)">1</button><span style="color:var(--text2)">…</span>`;
    for (let i = startP; i <= endP; i++) {
        pagHTML += `<button class="page-btn ${i===d.page?'active':''}" onclick="goPage(${i})">${i}</button>`;
    }
    if (endP < d.total_pages) pagHTML += `<span style="color:var(--text2)">…</span><button class="page-btn" onclick="goPage(${d.total_pages})">${d.total_pages}</button>`;
    pagHTML += `<button class="page-btn" onclick="goPage(${d.page+1})" ${d.page>=d.total_pages?'disabled':''}>→</button>`;

    document.getElementById('pagination').innerHTML = pagHTML;
}

function goPage(p) {
    currentPage = p;
    loadPredictions();
}

// Search + Filter
document.getElementById('pred-search').addEventListener('input', debounce(function() {
    currentSearch = this.value;
    currentPage = 1;
    loadPredictions();
}, 300));

document.querySelectorAll('.level-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.level-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        currentLevel = this.dataset.level;
        currentPage = 1;
        loadPredictions();
    });
});

function debounce(fn, ms) {
    let t; return function(...a) { clearTimeout(t); t = setTimeout(() => fn.apply(this, a), ms); };
}

function exportCSV() {
    window.location.href = `${API}/api/export`;
}

// ── Feature Analysis ────────────────────────────────────────────────────────
async function loadFeatureAnalysis() {
    const res = await fetch(`${API}/api/feature_analysis`);
    const d = await res.json();

    // All features
    const el = document.getElementById('all-features');
    const feats = d.feature_importance;
    const maxVal = feats.length ? Math.max(...feats.map(f => f.value)) : 1;
    el.innerHTML = feats.map((f, i) => {
        const pct = maxVal > 0 ? (f.value / maxVal * 100) : 0;
        return `<div class="bar-row">
            <span class="bar-label bar-label-wide" style="min-width:160px">${i+1}. ${f.name}</span>
            <div class="bar-track"><div class="bar-fill blue" style="width:${pct}%"></div></div>
            <span class="bar-val">${f.value}%</span>
        </div>`;
    }).join('');

    // HS Codes
    const hsEl = document.getElementById('top-hs-codes');
    const hsEntries = Object.entries(d.top_hs_codes);
    const hsMax = hsEntries.length ? Math.max(...hsEntries.map(e => e[1])) : 1;
    hsEl.innerHTML = hsEntries.map(([code, cnt]) => {
        const pct = hsMax > 0 ? (cnt / hsMax * 100) : 0;
        return `<div class="bar-row">
            <span class="bar-label">${code}</span>
            <div class="bar-track"><div class="bar-fill red" style="width:${pct}%"></div></div>
            <span class="bar-val">${cnt}</span>
        </div>`;
    }).join('') || '<p class="text-muted">No data</p>';

    // Trade Regime
    const regEl = document.getElementById('risk-regime');
    const regEntries = Object.entries(d.risk_by_regime);
    const regMax = regEntries.length ? Math.max(...regEntries.map(e => e[1])) : 1;
    regEl.innerHTML = regEntries.map(([regime, score]) => {
        const pct = regMax > 0 ? (score / regMax * 100) : 0;
        const color = score >= 15 ? 'red' : score >= 10 ? 'gradient' : 'blue';
        return `<div class="bar-row">
            <span class="bar-label" style="min-width:70px">${regime}</span>
            <div class="bar-track"><div class="bar-fill ${color}" style="width:${pct}%"></div></div>
            <span class="bar-val">${score}</span>
        </div>`;
    }).join('') || '<p class="text-muted">No data</p>';
}

// ── Model Metrics ───────────────────────────────────────────────────────────
async function loadModelMetrics() {
    const res = await fetch(`${API}/api/model_metrics`);
    const d = await res.json();

    document.getElementById('class-perf').innerHTML = `
        <div class="metric-row"><span class="metric-label">Overall Accuracy</span><span class="metric-value text-blue">${d.accuracy}%</span></div>
        <div class="metric-row"><span class="metric-label">F1 — Critical</span><span class="metric-value text-red">${d.f1_critical}%</span></div>
        <div class="metric-row"><span class="metric-label">F1 — Low Risk</span><span class="metric-value text-yellow">${d.f1_low_risk}%</span></div>
        <div class="metric-row"><span class="metric-label">Records Analyzed</span><span class="metric-value">${d.records.toLocaleString()}</span></div>
        <div class="metric-row"><span class="metric-label">Test Split</span><span class="metric-value">${d.test_split}</span></div>
    `;

    document.getElementById('algo-stack').innerHTML = `
        <div class="metric-row"><span class="metric-label">Primary Model</span><span class="metric-value">${d.model_type}</span></div>
        <div class="metric-row"><span class="metric-label">RF Estimators</span><span class="metric-value">${d.n_estimators_rf}</span></div>
        <div class="metric-row"><span class="metric-label">Anomaly Detector</span><span class="metric-value">${d.anomaly}</span></div>
        <div class="metric-row"><span class="metric-label">IF Estimators</span><span class="metric-value">${d.n_estimators_if}</span></div>
        <div class="metric-row"><span class="metric-label">Contamination</span><span class="metric-value">${(d.contamination*100)}%</span></div>
        <div class="metric-row"><span class="metric-label">Features</span><span class="metric-value">${d.features}</span></div>
    `;

    const bands = Object.entries(d.score_bands);
    const bandColors = {'0-10':'#3fb950','10-20':'#2ea043','20-40':'#d29922','40-60':'#e3b341','60-80':'#f85149','80-100':'#da3633'};
    const maxBand = Math.max(...bands.map(b => b[1]));
    document.getElementById('score-bands').innerHTML = bands.map(([band, count]) => {
        const pct = maxBand > 0 ? (count / maxBand * 100) : 0;
        const color = bandColors[band] || '#58a6ff';
        return `<div class="bar-row">
            <span class="bar-label" style="min-width:50px"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color};margin-right:5px;"></span>${band}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="bar-val">${count.toLocaleString()} <span style="color:var(--text2);font-size:10px">(${d.score_band_pcts[band]}%)</span></span>
        </div>`;
    }).join('');
}

// ── Config ──────────────────────────────────────────────────────────────────
async function loadConfig() {
    const res = await fetch(`${API}/api/stats`);
    const d = await res.json();
    const el = document.getElementById('cfg-records');
    if (el) el.textContent = d.total.toLocaleString() + ' records';
}

// ── Import CSV ──────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const csvInput = document.getElementById('csv-file');

if (uploadZone) {
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
}

if (csvInput) {
    csvInput.addEventListener('change', () => {
        if (csvInput.files.length) handleFile(csvInput.files[0]);
    });
}

async function handleFile(file) {
    const validExts = ['.csv', '.xlsx', '.xls'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!validExts.includes(ext)) {
        alert('Please upload a CSV or Excel (.xlsx/.xls) file');
        return;
    }
    const maxSize = 200 * 1024 * 1024; // 200 MB
    if (file.size > maxSize) {
        const sizeMB = (file.size / 1024 / 1024).toFixed(1);
        alert(`File too large (${sizeMB} MB). Maximum upload size is 200 MB.`);
        return;
    }
    const resultsDiv = document.getElementById('upload-results');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Processing... This may take a moment for large files.</div>';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
        const contentType = res.headers.get('content-type') || '';
        if (!res.ok) {
            if (contentType.includes('application/json')) {
                const errData = await res.json();
                resultsDiv.innerHTML = `<p class="text-red" style="text-align:center;padding:20px;">Error: ${errData.error || 'Upload failed'}</p>`;
            } else {
                const text = await res.text();
                const msg = res.status === 413 ? 'File too large. Maximum upload size is 200 MB.'
                    : `Server error (${res.status}): ${text.substring(0, 200)}`;
                resultsDiv.innerHTML = `<p class="text-red" style="text-align:center;padding:20px;">Error: ${msg}</p>`;
            }
            return;
        }
        const d = await res.json();
        if (d.error) {
            resultsDiv.innerHTML = `<p class="text-red" style="text-align:center;padding:20px;">Error: ${d.error}</p>`;
            return;
        }
        resultsDiv.innerHTML = `
            <p style="text-align:center;color:var(--green);font-weight:600;margin-bottom:14px;">✔ ${d.message}</p>
            <div class="upload-result-cards">
                <div class="upload-stat"><div class="upload-stat-val">${d.total.toLocaleString()}</div><div class="upload-stat-label">Total</div></div>
                <div class="upload-stat"><div class="upload-stat-val text-red">${d.critical.toLocaleString()}</div><div class="upload-stat-label">Critical</div></div>
                <div class="upload-stat"><div class="upload-stat-val text-yellow">${d.low_risk.toLocaleString()}</div><div class="upload-stat-label">Low Risk</div></div>
                <div class="upload-stat"><div class="upload-stat-val text-green">${d.clear.toLocaleString()}</div><div class="upload-stat-label">Clear</div></div>
            </div>
            <div style="text-align:center;margin-top:16px;">
                <button class="btn-browse" onclick="exportCSV()">⬇ Download Predictions</button>
                <button class="btn-browse" style="background:var(--green);margin-left:8px;" onclick="navigateTo('dashboard');init();">View Dashboard</button>
            </div>
        `;
        // Refresh stats
        loadStats();
    } catch (err) {
        resultsDiv.innerHTML = `<p class="text-red" style="text-align:center;padding:20px;">Upload failed: ${err.message}</p>`;
    }
}

// ── Pulse animation via CSS ─────────────────────────────────────────────────
const style = document.createElement('style');
style.textContent = `
    .pulse-marker { animation: pulse-ring 2s ease-out infinite; }
    @keyframes pulse-ring {
        0% { opacity: 0.6; stroke-width: 2; }
        100% { opacity: 0; stroke-width: 0; r: 30; }
    }
`;
document.head.appendChild(style);

// ── Chatbot ─────────────────────────────────────────────────────────────────
let chatOpen = false;
let geminiKeySet = false;

function toggleChat() {
    const panel = document.getElementById('chat-panel');
    chatOpen = !chatOpen;
    panel.classList.toggle('open', chatOpen);
    if (chatOpen) {
        document.getElementById('chat-input').focus();
        document.getElementById('chat-badge').style.display = 'none';
    }
}

function toggleKeyInput() {
    const bar = document.getElementById('chat-key-bar');
    bar.style.display = bar.style.display === 'none' ? 'flex' : 'none';
    if (bar.style.display === 'flex') document.getElementById('gemini-key-input').focus();
}

function saveGeminiKey() {
    const input = document.getElementById('gemini-key-input');
    const key = input.value.trim();
    if (!key) return;
    fetch(`${API}/api/chat/set_key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
    })
    .then(r => r.json())
    .then(data => {
        if (data.message) {
            geminiKeySet = true;
            appendChatMsg('bot', '✅ Gemini API key set! Ask me anything about your data.');
            document.getElementById('chat-key-bar').style.display = 'none';
            input.value = '';
        } else {
            appendChatMsg('bot', '❌ ' + (data.error || 'Failed to set key.'));
        }
    })
    .catch(() => appendChatMsg('bot', '❌ Failed to connect to server.'));
}

function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    appendChatMsg('user', msg);
    showTyping();

    fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
    })
    .then(res => res.json())
    .then(data => {
        removeTyping();
        appendChatMsg('bot', formatMarkdown(data.reply));
    })
    .catch(() => {
        removeTyping();
        appendChatMsg('bot', 'Sorry, something went wrong. Please try again.');
    });
}

function appendChatMsg(role, html) {
    const container = document.getElementById('chat-messages');
    const avatar = role === 'bot' ? '🤖' : '👤';
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = `<div class="chat-msg-avatar">${avatar}</div><div class="chat-msg-bubble">${html}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-msg bot';
    div.id = 'chat-typing';
    div.innerHTML = `<div class="chat-msg-avatar">🤖</div><div class="chat-msg-bubble"><div class="chat-typing"><span></span><span></span><span></span></div></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById('chat-typing');
    if (el) el.remove();
}

function formatMarkdown(text) {
    // Sanitize HTML tags from AI output
    text = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return text
        // Headers: ### or ## or #
        .replace(/^### (.+)$/gm, '<strong style="font-size:13px;color:var(--blue)">$1</strong>')
        .replace(/^## (.+)$/gm, '<strong style="font-size:14px;color:var(--blue)">$1</strong>')
        .replace(/^# (.+)$/gm, '<strong style="font-size:15px;color:var(--blue)">$1</strong>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Numbered lists: 1. item
        .replace(/^(\d+)\. (.+)$/gm, '<div style="margin-left:8px"><strong>$1.</strong> $2</div>')
        // Bullet points
        .replace(/^[•\-\*] (.+)$/gm, '<div style="margin-left:8px">&bull; $1</div>')
        // Horizontal rule ---
        .replace(/^---$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:6px 0">')
        // Line breaks
        .replace(/\n/g, '<br>');
}

// Enter key sends message
document.getElementById('chat-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
});

// ── Start ───────────────────────────────────────────────────────────────────
init();
setTimeout(initMap, 200);
