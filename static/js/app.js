/* ── SmartContainer Risk Engine — Frontend Logic ─────────────────────────── */

const API = '';
let currentPage = 1;
let currentLevel = 'ALL';
let currentSearch = '';
let map = null;

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

// ── Choropleth Map ──────────────────────────────────────────────────────────
const GEOJSON_URL = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson';
let geoLayer = null;

async function loadMapData() {
    const res = await fetch(`${API}/api/map_data`);
    window._mapData = await res.json();
    // Build lookup by country code
    window._mapLookup = {};
    let totalContainers = 0, countryCount = 0;
    window._mapData.forEach(d => {
        window._mapLookup[d.country] = d;
        totalContainers += d.total;
        countryCount++;
    });
    const summaryEl = document.getElementById('map-summary');
    if (summaryEl) summaryEl.textContent = `${countryCount} countries • ${totalContainers.toLocaleString()} total containers`;
    if (map) renderChoropleth();
}

function riskColor(score) {
    if (score <= 0) return '#161b22';
    if (score <= 10) return '#1a4d2e';
    if (score <= 20) return '#3fb950';
    if (score <= 35) return '#a3d977';
    if (score <= 50) return '#d4e157';
    if (score <= 65) return '#ffca28';
    if (score <= 80) return '#f85149';
    return '#da3633';
}

function initMap() {
    if (map) return;
    map = L.map('map', {
        center: [25, 20],
        zoom: 2,
        minZoom: 2,
        maxZoom: 6,
        attributionControl: false,
        zoomControl: true,
        worldCopyJump: true
    });

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Load GeoJSON and render
    fetch(GEOJSON_URL)
        .then(r => r.json())
        .then(geojson => {
            window._geoData = geojson;
            if (window._mapData) renderChoropleth();
        });
}

// ISO-3 to ISO-2 mapping for matching API data
const iso3to2 = {
    AFG:'AF',ALB:'AL',DZA:'DZ',AGO:'AO',ARG:'AR',ARM:'AM',AUS:'AU',AUT:'AT',AZE:'AZ',
    BHS:'BS',BHR:'BH',BGD:'BD',BRB:'BB',BLR:'BY',BEL:'BE',BLZ:'BZ',BEN:'BJ',BTN:'BT',
    BOL:'BO',BIH:'BA',BWA:'BW',BRA:'BR',BRN:'BN',BGR:'BG',BFA:'BF',BDI:'BI',KHM:'KH',
    CMR:'CM',CAN:'CA',CPV:'CV',CAF:'CF',TCD:'TD',CHL:'CL',CHN:'CN',COL:'CO',COM:'KM',
    COG:'CG',COD:'CD',CRI:'CR',CIV:'CI',HRV:'HR',CUB:'CU',CYP:'CY',CZE:'CZ',DNK:'DK',
    DJI:'DJ',DOM:'DO',ECU:'EC',EGY:'EG',SLV:'SV',GNQ:'GQ',ERI:'ER',EST:'EE',ETH:'ET',
    FJI:'FJ',FIN:'FI',FRA:'FR',GAB:'GA',GMB:'GM',GEO:'GE',DEU:'DE',GHA:'GH',GRC:'GR',
    GTM:'GT',GIN:'GN',GNB:'GW',GUY:'GY',HTI:'HT',HND:'HN',HUN:'HU',ISL:'IS',IND:'IN',
    IDN:'ID',IRN:'IR',IRQ:'IQ',IRL:'IE',ISR:'IL',ITA:'IT',JAM:'JM',JPN:'JP',JOR:'JO',
    KAZ:'KZ',KEN:'KE',KWT:'KW',KGZ:'KG',LAO:'LA',LVA:'LV',LBN:'LB',LSO:'LS',LBR:'LR',
    LBY:'LY',LIE:'LI',LTU:'LT',LUX:'LU',MKD:'MK',MDG:'MG',MWI:'MW',MYS:'MY',MLI:'ML',
    MLT:'MT',MRT:'MR',MUS:'MU',MEX:'MX',MDA:'MD',MNG:'MN',MNE:'ME',MAR:'MA',MOZ:'MZ',
    MMR:'MM',NAM:'NA',NPL:'NP',NLD:'NL',NZL:'NZ',NIC:'NI',NER:'NE',NGA:'NG',PRK:'KP',
    NOR:'NO',OMN:'OM',PAK:'PK',PAN:'PA',PNG:'PG',PRY:'PY',PER:'PE',PHL:'PH',POL:'PL',
    PRT:'PT',QAT:'QA',ROU:'RO',RUS:'RU',RWA:'RW',SAU:'SA',SEN:'SN',SRB:'RS',SLE:'SL',
    SGP:'SG',SVK:'SK',SVN:'SI',SOM:'SO',ZAF:'ZA',KOR:'KR',SSD:'SS',ESP:'ES',LKA:'LK',
    SDN:'SD',SUR:'SR',SWZ:'SZ',SWE:'SE',CHE:'CH',SYR:'SY',TWN:'TW',TJK:'TJ',TZA:'TZ',
    THA:'TH',TLS:'TL',TGO:'TG',TTO:'TT',TUN:'TN',TUR:'TR',TKM:'TM',UGA:'UG',UKR:'UA',
    ARE:'AE',GBR:'GB',USA:'US',URY:'UY',UZB:'UZ',VEN:'VE',VNM:'VN',YEM:'YE',ZMB:'ZM',
    ZWE:'ZW',PSE:'PS',XKX:'XK',SXM:'SX',CUW:'CW',SSD:'SS',COK:'CK'
};

function renderChoropleth() {
    if (!window._geoData || !window._mapLookup) return;
    if (geoLayer) map.removeLayer(geoLayer);

    const infoBox = document.getElementById('map-info');

    geoLayer = L.geoJSON(window._geoData, {
        style: feature => {
            const iso3 = feature.properties.ISO_A3;
            const iso2 = iso3to2[iso3] || '';
            const d = window._mapLookup[iso2];
            const score = d ? d.avg_risk_score : -1;
            return {
                fillColor: riskColor(score),
                weight: 1,
                opacity: 0.7,
                color: '#30363d',
                fillOpacity: score >= 0 ? 0.85 : 0.15
            };
        },
        onEachFeature: (feature, layer) => {
            const iso3 = feature.properties.ISO_A3;
            const iso2 = iso3to2[iso3] || '';
            const d = window._mapLookup[iso2];

            layer.on({
                mouseover: e => {
                    e.target.setStyle({ weight: 2, color: '#e6edf3', fillOpacity: 0.95 });
                    e.target.bringToFront();
                    if (d && infoBox) {
                        infoBox.style.display = 'block';
                        infoBox.innerHTML = `
                            <div class="info-country">${iso2}</div>
                            <div class="info-row">Total_Containers=<b>${d.total.toLocaleString()}</b></div>
                            <div class="info-row">Critical_Count=<b>${d.critical}</b></div>
                            <div class="info-row">Avg_Risk_Score=<b>${d.avg_risk_score}</b></div>
                        `;
                    }
                },
                mouseout: e => {
                    geoLayer.resetStyle(e.target);
                    if (infoBox) infoBox.style.display = 'none';
                },
                click: e => {
                    map.fitBounds(e.target.getBounds());
                }
            });
        }
    }).addTo(map);
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
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }
    const resultsDiv = document.getElementById('upload-results');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Processing...</div>';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
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

// ── Start ───────────────────────────────────────────────────────────────────
init();
setTimeout(initMap, 200);
