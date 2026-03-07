"""
SmartContainer Risk Engine - Flask API Backend
Serves predictions, stats, and data for the HTML/CSS/JS dashboard
"""
from flask import Flask, jsonify, request, send_from_directory, send_file
import pandas as pd
import numpy as np
import pickle, os, io, re
import google.generativeai as genai
from predict import RiskPredictor
from utils import load_data

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 95 * 1024 * 1024  # 95 MB (Render proxy limit ~100MB)


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum upload size is 95 MB.'}), 413


# ── Cache ────────────────────────────────────────────────────────────────────
_cache = {}

def get_predictions():
    """Load or return cached predictions"""
    if 'preds' not in _cache:
        df = load_data('data/historical_data.csv')
        predictor = RiskPredictor()
        preds = predictor.predict(df)
        # Merge original columns
        for col in ['Origin_Country','Destination_Port','Dwell_Time_Hours',
                     'Declared_Value','HS_Code','Declared_Weight','Measured_Weight']:
            if col in df.columns:
                preds[col] = df[col].values
        if 'Trade_Regime (Import / Export / Transit)' in df.columns:
            preds['Trade_Regime'] = df['Trade_Regime (Import / Export / Transit)'].values
        _cache['preds'] = preds
        _cache['df'] = df
    return _cache['preds']

def get_feature_importances():
    """Get RF feature importances"""
    if 'feat_imp' not in _cache:
        rf = pickle.load(open('models/random_forest_model.pkl','rb'))
        fe = pickle.load(open('models/feature_engineer.pkl','rb'))
        names = fe.numeric_features
        imp = rf.feature_importances_
        pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
        short = {
            'Weight_Diff_%':'Weight Diff %','Dwell_Time_Hours':'Dwell Hours',
            'Declared_Value':'Declared Value','HS_Code_Risk_Score':'HS Code Risk',
            'Origin_Risk_Score':'Origin Risk','Port_Risk_Score':'Port Risk',
            'Value_Weight_Ratio':'Value/Weight','Anomaly_Score':'Anomaly Score',
            'Trade_Risk_Score':'Trade Risk','High_Dwell':'High Dwell',
            'Weight_Ratio':'Weight Ratio','Declared_Weight':'Declared Wt',
            'Measured_Weight':'Measured Wt','Value_Per_Kg':'Value/Kg',
            'Dwell_Risk_Score':'Dwell Risk'
        }
        _cache['feat_imp'] = [{'name': short.get(n,n), 'value': round(float(v)*100,2)} for n,v in pairs]
    return _cache['feat_imp']

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/stats')
def api_stats():
    preds = get_predictions()
    total = len(preds)
    critical = int((preds['Risk_Level']=='Critical').sum())
    low_risk = int((preds['Risk_Level']=='Low Risk').sum())
    clear = int((preds['Risk_Level']=='Clear').sum())
    avg_score = round(float(preds['Risk_Score'].mean()), 2)
    return jsonify({
        'total': total, 'critical': critical, 'low_risk': low_risk,
        'clear': clear, 'avg_score': avg_score
    })

@app.route('/api/predictions')
def api_predictions():
    preds = get_predictions()
    level = request.args.get('level','ALL')
    search = request.args.get('search','').strip()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    sort_by = request.args.get('sort_by', 'Risk_Score')
    sort_dir = request.args.get('sort_dir', 'desc')

    df = preds.copy()
    if level != 'ALL':
        df = df[df['Risk_Level'] == level]
    if search:
        mask = df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        df = df[mask]

    asc = sort_dir == 'asc'
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=asc)

    total_rows = len(df)
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    page_df = df.iloc[start:start+per_page]

    rows = []
    for _, r in page_df.iterrows():
        val = r.get('Declared_Value', 0)
        if val >= 1_000_000:
            val_fmt = f"${val/1_000_000:.1f}M"
        elif val >= 1_000:
            val_fmt = f"${val/1_000:.0f}K"
        else:
            val_fmt = f"${val:.0f}"
        rows.append({
            'container_id': str(r['Container_ID']),
            'risk_score': round(float(r['Risk_Score']), 1),
            'risk_level': r['Risk_Level'],
            'origin': r.get('Origin_Country',''),
            'destination': r.get('Destination_Port',''),
            'dwell': round(float(r.get('Dwell_Time_Hours',0)), 1),
            'value': val_fmt,
            'explanation': r.get('Explanation','')
        })
    return jsonify({
        'rows': rows, 'total_rows': total_rows,
        'page': page, 'total_pages': total_pages, 'per_page': per_page
    })

@app.route('/api/charts')
def api_charts():
    preds = get_predictions()
    # Risk distribution
    dist = {'Critical': int((preds['Risk_Level']=='Critical').sum()),
            'Low Risk': int((preds['Risk_Level']=='Low Risk').sum()),
            'Clear': int((preds['Risk_Level']=='Clear').sum())}

    # Score histogram bands
    bins = [0,10,20,40,60,80,100]
    labels = ['0-10','10-20','20-40','40-60','60-80','80-100']
    preds['band'] = pd.cut(preds['Risk_Score'], bins=bins, labels=labels, include_lowest=True)
    hist = preds['band'].value_counts().reindex(labels).fillna(0).astype(int).to_dict()

    # Critical origins top 6
    crit = preds[preds['Risk_Level']=='Critical']
    origins = crit['Origin_Country'].value_counts().head(6).to_dict() if 'Origin_Country' in preds.columns else {}

    # High-risk ports top 8
    if 'Destination_Port' in preds.columns:
        ports = preds.groupby('Destination_Port')['Risk_Score'].mean().nlargest(8).round(1).to_dict()
    else:
        ports = {}

    # Feature importances
    feat_imp = get_feature_importances()

    return jsonify({
        'distribution': dist, 'histogram': hist,
        'critical_origins': origins, 'high_risk_ports': ports,
        'feature_importance': feat_imp[:8]
    })

@app.route('/api/map_data')
def api_map_data():
    preds = get_predictions()
    if 'Origin_Country' not in preds.columns:
        return jsonify([])

    # Country coordinates
    coords = {
        'CN':(35.86,104.19),'US':(37.09,-95.71),'DE':(51.17,10.45),'JP':(36.20,138.25),
        'KR':(35.91,127.77),'IN':(20.59,78.96),'BR':(-14.24,-51.93),'RU':(61.52,105.32),
        'GB':(55.38,-3.44),'FR':(46.60,1.89),'IT':(41.87,12.57),'CA':(56.13,-106.35),
        'AU':(-25.27,133.78),'MX':(23.63,-102.55),'ID':(-0.79,113.92),'TR':(38.96,35.24),
        'SA':(23.89,45.08),'TH':(15.87,100.99),'VN':(14.06,108.28),'MY':(4.21,101.98),
        'PH':(12.88,121.77),'PK':(30.38,69.35),'NG':(9.08,8.68),'EG':(26.82,30.80),
        'AR':(-38.42,-63.62),'ZA':(-30.56,22.94),'PL':(51.92,19.15),'NL':(52.13,5.29),
        'BE':(50.50,4.47),'SE':(60.13,18.64),'CH':(46.82,8.23),'AT':(47.52,14.55),
        'SG':(1.35,103.82),'HK':(22.40,114.11),'TW':(23.70,120.96),'AE':(23.42,53.85),
        'IL':(31.05,34.85),'GR':(39.07,21.82),'CZ':(49.82,15.47),'RO':(45.94,24.97),
        'HU':(47.16,19.50),'PT':(39.40,-8.22),'ES':(40.46,-3.75),'CL':(-35.68,-71.54),
        'CO':(4.57,-74.30),'PE':(-9.19,-75.02),'UA':(48.38,31.17),'BD':(23.68,90.36),
        'LK':(7.87,80.77),'MM':(21.91,95.96),'KH':(12.57,104.99),'NP':(28.39,84.12),
        'MA':(31.79,-7.09),'TN':(33.89,9.54),'GH':(7.95,-1.02),'KE':(-0.02,37.91),
        'ET':(9.15,40.49),'UG':(1.37,32.29),'SD':(12.86,30.22),'SN':(14.50,-14.45),
        'DK':(56.26,9.50),'NO':(60.47,8.47),'FI':(61.92,25.75),'IE':(53.14,-7.69),
        'NZ':(-40.90,174.89),'QA':(25.35,51.18),'BH':(26.07,50.56),'OM':(21.47,55.98),
        'JO':(30.59,36.24),'LB':(33.85,35.86),'IR':(32.43,53.69),'IQ':(33.22,43.68),
        'RS':(44.02,21.01),'BG':(42.73,25.49),'HR':(45.10,15.20),'SI':(46.15,14.99),
        'SK':(48.67,19.70),'LT':(55.17,23.88),'LV':(56.88,24.60),'EE':(58.60,25.01),
        'AL':(41.15,20.17),'BA':(43.92,17.68),'MN':(46.86,103.85),'LA':(19.86,102.50),
        'AM':(40.07,45.04),'GE':(42.32,43.36),'UZ':(41.38,64.59),'MT':(35.94,14.38),
        'LU':(49.82,6.13),'LI':(47.17,9.56),'MO':(22.20,113.54),'CR':(9.75,-83.75),
        'DO':(18.74,-70.16),'EC':(-1.83,-78.18),'GT':(15.78,-90.23),'HN':(15.20,-86.24),
        'HT':(18.97,-72.29),'NI':(12.87,-85.21),'SV':(13.79,-88.90),'PY':(-23.44,-58.44),
        'TT':(10.69,-61.22),'VE':(6.42,-66.59),'BN':(4.54,114.73),'PR':(18.22,-66.59),
        'WS':(-13.76,-172.10),'ZZ':(0,0),'BV':(-54.42,3.41),'BW':(-22.33,24.68),
        'CD':(-4.04,21.76),'GG':(49.45,-2.54),'GN':(9.95,-9.70),'GW':(11.80,-15.18),
        'LR':(6.43,-9.43),'MD':(47.41,28.37),'MG':(-18.77,46.87),'ML':(17.57,-4.00),
        'MP':(15.10,145.67),'PG':(-6.31,143.96),'SZ':(-26.52,31.47),'TG':(8.62,1.21),
        'YE':(15.55,48.52)
    }

    # Aggregate by origin + risk level
    grouped = preds.groupby(['Origin_Country','Risk_Level']).size().reset_index(name='count')
    result = []
    for country in preds['Origin_Country'].unique():
        if country not in coords or country == 'ZZ':
            continue
        lat, lng = coords[country]
        c_data = grouped[grouped['Origin_Country']==country]
        total = int(c_data['count'].sum())
        critical = int(c_data[c_data['Risk_Level']=='Critical']['count'].sum()) if len(c_data[c_data['Risk_Level']=='Critical']) else 0
        low_risk = int(c_data[c_data['Risk_Level']=='Low Risk']['count'].sum()) if len(c_data[c_data['Risk_Level']=='Low Risk']) else 0
        clear = total - critical - low_risk
        result.append({
            'country': country, 'lat': lat, 'lng': lng,
            'total': total, 'critical': critical,
            'low_risk': low_risk, 'clear': clear
        })
    return jsonify(sorted(result, key=lambda x: x['critical'], reverse=True))

@app.route('/api/feature_analysis')
def api_feature_analysis():
    preds = get_predictions()
    feat_imp = get_feature_importances()

    # Top critical HS codes
    crit = preds[preds['Risk_Level']=='Critical']
    hs_codes = crit['HS_Code'].value_counts().head(5).to_dict() if 'HS_Code' in preds.columns else {}

    # Risk by trade regime
    if 'Trade_Regime' in preds.columns:
        regime = preds.groupby('Trade_Regime')['Risk_Score'].mean().round(1).to_dict()
    else:
        regime = {}

    return jsonify({
        'feature_importance': feat_imp,
        'top_hs_codes': {str(k):int(v) for k,v in hs_codes.items()},
        'risk_by_regime': regime
    })

@app.route('/api/model_metrics')
def api_model_metrics():
    preds = get_predictions()
    bins = [0,10,20,40,60,80,100]
    labels = ['0-10','10-20','20-40','40-60','60-80','80-100']
    preds_copy = preds.copy()
    preds_copy['band'] = pd.cut(preds_copy['Risk_Score'], bins=bins, labels=labels, include_lowest=True)
    bands = preds_copy['band'].value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    total = len(preds)
    band_pcts = {k: round(v/total*100, 1) for k,v in bands.items()}

    return jsonify({
        'accuracy': 99.82, 'f1_critical': 96.68, 'f1_low_risk': 99.62,
        'records': total, 'test_split': '80/20',
        'model_type': 'Random Forest', 'anomaly': 'Isolation Forest',
        'contamination': 0.03, 'n_estimators_rf': 300, 'n_estimators_if': 200,
        'features': 15, 'classes': 3,
        'score_bands': bands, 'score_band_pcts': band_pcts
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files accepted'}), 400

    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    required = ['Container_ID','Declared_Weight','Measured_Weight','Dwell_Time_Hours','Declared_Value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({'error': f'Missing columns: {", ".join(missing)}'}), 400

    # Fill optional columns with defaults so the pipeline doesn't break
    defaults = {
        'Declaration_Date (YYYY-MM-DD)': '2025-01-01',
        'Declaration_Time': '12:00:00',
        'Trade_Regime (Import / Export / Transit)': 'Import',
        'Origin_Country': 'ZZ',
        'Destination_Port': 'PORT_40',
        'Destination_Country': 'ZZ',
        'HS_Code': '000000',
        'Importer_ID': 'UNKNOWN',
        'Exporter_ID': 'UNKNOWN',
        'Shipping_Line': 'LINE_MODE_10',
        'Clearance_Status': 'Clear',
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    try:
        predictor = RiskPredictor()
        preds = predictor.predict(df)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    for col in ['Origin_Country','Destination_Port','Dwell_Time_Hours',
                 'Declared_Value','HS_Code','Declared_Weight','Measured_Weight']:
        if col in df.columns:
            preds[col] = df[col].values
    if 'Trade_Regime (Import / Export / Transit)' in df.columns:
        preds['Trade_Regime'] = df['Trade_Regime (Import / Export / Transit)'].values

    _cache['preds'] = preds
    _cache['df'] = df
    if 'feat_imp' in _cache:
        del _cache['feat_imp']

    total = len(preds)
    critical = int((preds['Risk_Level']=='Critical').sum())
    low_risk = int((preds['Risk_Level']=='Low Risk').sum())
    clear = int((preds['Risk_Level']=='Clear').sum())

    return jsonify({
        'message': f'Processed {total} containers from {file.filename}',
        'total': total, 'critical': critical, 'low_risk': low_risk, 'clear': clear
    })

@app.route('/api/export')
def api_export():
    preds = get_predictions()
    buf = io.BytesIO()
    preds.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='risk_predictions.csv')

# ── Chatbot ──────────────────────────────────────────────────────────────────

_gemini_key = os.environ.get('GEMINI_API_KEY', '')
_chat_sessions = {}


@app.route('/api/chat/set_key', methods=['POST'])
def set_gemini_key():
    global _gemini_key
    data = request.get_json()
    key = data.get('api_key', '').strip() if data else ''
    if not key:
        return jsonify({'error': 'API key is required'}), 400
    _gemini_key = key
    _chat_sessions.clear()
    return jsonify({'message': 'API key set successfully'})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    global _gemini_key
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'reply': 'Please send a message.'}), 400
    msg = data['message'].strip()
    if not msg:
        return jsonify({'reply': 'Please type a question about your container risk data.'})

    # Allow setting key inline
    if msg.lower().startswith('api key:') or msg.lower().startswith('key:'):
        key = msg.split(':', 1)[1].strip()
        if key:
            _gemini_key = key
            _chat_sessions.clear()
            return jsonify({'reply': '✅ Gemini API key set! You can now ask questions about your data.'})

    if not _gemini_key:
        return jsonify({'reply': ('⚠️ **Gemini API key not set.**\n\n'
                                  'Click the 🔑 button in the chat header to enter your key, '
                                  'or type: **key: YOUR_API_KEY**')})

    try:
        reply = gemini_chat(msg)
        return jsonify({'reply': reply})
    except Exception as e:
        err = str(e)
        if 'API_KEY_INVALID' in err or 'invalid' in err.lower():
            return jsonify({'reply': '❌ Invalid API key. Please set a valid Gemini API key using the 🔑 button.'})
        if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
            return jsonify({'reply': '⏳ **Rate limit reached.** Your Gemini API free tier quota is exhausted. Please wait a minute and try again, or enable billing on your Google AI account for higher limits.'})
        return jsonify({'reply': f'⚠️ Error: {err}'})


def build_data_context():
    """Build a data summary string from the current predictions for Gemini context."""
    preds = get_predictions()
    total = len(preds)
    critical = int((preds['Risk_Level'] == 'Critical').sum())
    low_risk = int((preds['Risk_Level'] == 'Low Risk').sum())
    clear = int((preds['Risk_Level'] == 'Clear').sum())
    avg_score = round(float(preds['Risk_Score'].mean()), 2)

    ctx = f"""CURRENT DATASET STATISTICS:
- Total Containers: {total:,}
- Critical: {critical:,} ({round(critical/total*100,2)}%)
- Low Risk: {low_risk:,} ({round(low_risk/total*100,2)}%)
- Clear: {clear:,} ({round(clear/total*100,2)}%)
- Average Risk Score: {avg_score}/100
- Columns: {', '.join(preds.columns.tolist())}
"""
    # Top 5 riskiest containers
    top5 = preds.nlargest(5, 'Risk_Score')
    ctx += "\nTOP 5 HIGHEST RISK CONTAINERS:\n"
    for _, r in top5.iterrows():
        origin = r.get('Origin_Country', '?')
        dest = r.get('Destination_Port', '?')
        ctx += (f"  Container {r['Container_ID']}: Score={r['Risk_Score']:.1f}, "
                f"Level={r['Risk_Level']}, Origin={origin}, Port={dest}, "
                f"Explanation={r.get('Explanation','')}\n")

    # Country breakdown
    if 'Origin_Country' in preds.columns:
        crit_origins = preds[preds['Risk_Level'] == 'Critical']['Origin_Country'].value_counts().head(10)
        ctx += "\nTOP CRITICAL ORIGINS:\n"
        for country, count in crit_origins.items():
            total_c = len(preds[preds['Origin_Country'] == country])
            ctx += f"  {country}: {count} critical out of {total_c} total\n"

    # Port breakdown
    if 'Destination_Port' in preds.columns:
        port_risk = preds.groupby('Destination_Port')['Risk_Score'].mean().nlargest(8).round(1)
        ctx += "\nHIGHEST RISK PORTS (avg score):\n"
        for port, score in port_risk.items():
            ctx += f"  {port}: avg score {score}\n"

    # Feature importances
    try:
        feat_imp = get_feature_importances()
        ctx += "\nFEATURE IMPORTANCES (Random Forest):\n"
        for f in feat_imp[:10]:
            ctx += f"  {f['name']}: {f['value']}%\n"
    except Exception:
        pass

    # Weight & dwell stats
    if 'Declared_Weight' in preds.columns and 'Measured_Weight' in preds.columns:
        wt_diff = ((preds['Measured_Weight'] - preds['Declared_Weight']).abs() / preds['Declared_Weight'].replace(0, 1) * 100)
        ctx += f"\nWEIGHT DISCREPANCY: avg={wt_diff.mean():.1f}%, max={wt_diff.max():.1f}%, >20%: {int((wt_diff > 20).sum())} containers\n"

    if 'Dwell_Time_Hours' in preds.columns:
        ctx += (f"DWELL TIME: avg={preds['Dwell_Time_Hours'].mean():.1f}h, "
                f"max={preds['Dwell_Time_Hours'].max():.1f}h, "
                f">72h: {int((preds['Dwell_Time_Hours'] > 72).sum())}, "
                f">120h: {int((preds['Dwell_Time_Hours'] > 120).sum())}\n")

    # Sample of 10 critical containers for detailed queries
    crit_sample = preds[preds['Risk_Level'] == 'Critical'].head(10)
    if len(crit_sample) > 0:
        ctx += "\nSAMPLE CRITICAL CONTAINERS:\n"
        for _, r in crit_sample.iterrows():
            ctx += f"  ID={r['Container_ID']}, Score={r['Risk_Score']:.1f}"
            for col in ['Origin_Country', 'Destination_Port', 'Dwell_Time_Hours', 'Declared_Value', 'Explanation']:
                if col in r and pd.notna(r.get(col)):
                    ctx += f", {col}={r[col]}"
            ctx += "\n"

    return ctx


def build_container_context(msg):
    """If user asks about a specific container, include its full details."""
    preds = get_predictions()
    cid_match = re.search(r'\b(\d{5,})\b', msg)
    if not cid_match:
        return ""
    cid = cid_match.group(1)
    row = preds[preds['Container_ID'].astype(str) == cid]
    if len(row) == 0:
        return f"\nContainer {cid} was NOT found in the dataset.\n"
    r = row.iloc[0]
    ctx = f"\nSPECIFIC CONTAINER LOOKUP — Container {cid}:\n"
    for col in r.index:
        ctx += f"  {col}: {r[col]}\n"
    return ctx


def gemini_chat(user_msg):
    """Send message to Gemini with full data context."""
    genai.configure(api_key=_gemini_key)

    data_ctx = build_data_context()
    container_ctx = build_container_context(user_msg)

    system_prompt = f"""You are the SmartContainer Risk Engine AI Assistant — an expert customs risk analysis chatbot.
You answer questions ONLY based on the actual container risk data provided below. Do not make up data.

MODEL INFORMATION:
- Primary: Random Forest (300 estimators, max_depth=30)
- Anomaly: Isolation Forest (contamination=3%, 200 estimators)
- Risk Score formula: 70% RF probability + 30% Anomaly score (0-100 scale)
- Accuracy: 99.82% test, 99.98% validation
- F1 Critical: 96.68%, F1 Low Risk: 99.62%
- 15 engineered features, 3 classes (Critical, Low Risk, Clear)
- Training: SMOTE-balanced sampling

RISK THRESHOLDS:
- Critical: Score ≥ 50 (immediate inspection)
- Low Risk: Score 20-49 (enhanced monitoring)
- Clear: Score < 20 (routine processing)
- Weight discrepancy > 20% = suspicious
- Dwell time > 72h = high, > 120h = very high
- High-risk origins: CN(25), RO(22), VN(20), ID(18), JP(15)

{data_ctx}
{container_ctx}

INSTRUCTIONS:
- Answer based on the actual data above. Be specific with numbers.
- Use markdown formatting: **bold** for emphasis, bullet points with •
- If asked about a specific container, provide all its details from the data.
- If data doesn't contain the answer, say so honestly.
- Keep answers concise but informative. Use emojis for risk levels (🔴 Critical, 🟡 Low Risk, 🟢 Clear).
- If the user asks about something unrelated to container risk/customs, politely redirect them.
"""

    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=system_prompt
    )

    # Use per-session chat history for multi-turn
    session_id = 'default'
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = model.start_chat(history=[])

    chat = _chat_sessions[session_id]
    response = chat.send_message(user_msg)
    return response.text


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
