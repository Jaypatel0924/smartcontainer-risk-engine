"""
SmartContainer Risk Engine - Flask API Backend
Serves predictions, stats, and data for the HTML/CSS/JS dashboard
"""
from flask import Flask, jsonify, request, send_from_directory, send_file
import pandas as pd
import numpy as np
import pickle, os, io, re
from predict import RiskPredictor
from utils import load_data

app = Flask(__name__, static_folder='static', template_folder='templates')

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

    df = pd.read_csv(io.BytesIO(file.read()))
    required = ['Container_ID','Declared_Weight','Measured_Weight','Dwell_Time_Hours','Declared_Value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({'error': f'Missing columns: {", ".join(missing)}'}), 400

    predictor = RiskPredictor()
    preds = predictor.predict(df)
    for col in ['Origin_Country','Destination_Port','Dwell_Time_Hours',
                 'Declared_Value','HS_Code','Declared_Weight','Measured_Weight']:
        if col in df.columns:
            preds[col] = df[col].values
    if 'Trade_Regime (Import / Export / Transit)' in df.columns:
        preds['Trade_Regime'] = df['Trade_Regime (Import / Export / Transit)'].values

    _cache['preds'] = preds
    _cache['df'] = df

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

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'reply': 'Please send a message.'}), 400
    msg = data['message'].strip()
    if not msg:
        return jsonify({'reply': 'Please type a question about your container risk data.'})
    reply = chatbot_reply(msg)
    return jsonify({'reply': reply})


def chatbot_reply(msg):
    """Rule-based chatbot that answers questions about the container risk data."""
    low = msg.lower()
    preds = get_predictions()
    total = len(preds)
    critical = int((preds['Risk_Level'] == 'Critical').sum())
    low_risk = int((preds['Risk_Level'] == 'Low Risk').sum())
    clear = int((preds['Risk_Level'] == 'Clear').sum())
    avg_score = round(float(preds['Risk_Score'].mean()), 2)

    # ── Greeting ──
    if re.search(r'\b(hi|hello|hey|greetings|good morning|good afternoon)\b', low):
        return (f"Hello! I'm the SmartContainer Risk Assistant. I can answer questions about your "
                f"{total:,} containers — risk levels, countries, ports, features, model info, and more. "
                f"Try asking: \"How many critical containers?\" or \"Tell me about container 12345\".")

    # ── Help ──
    if re.search(r'\b(help|what can you|what do you|how to use|commands)\b', low):
        return ("Here's what I can help with:\n"
                "• **Statistics** — \"How many critical containers?\"\n"
                "• **Container lookup** — \"Tell me about container 12345\"\n"
                "• **Country analysis** — \"Which countries are high risk?\" or \"Show me CN stats\"\n"
                "• **Port analysis** — \"What are the riskiest ports?\"\n"
                "• **Feature importance** — \"What features matter most?\"\n"
                "• **Model info** — \"What model do you use?\" or \"What is the accuracy?\"\n"
                "• **Thresholds** — \"What makes a container critical?\"\n"
                "• **Data summary** — \"Give me a summary\"")

    # ── Container lookup ──
    cid_match = re.search(r'\b(\d{5,})\b', msg)
    if cid_match or re.search(r'\bcontainer\b', low):
        if cid_match:
            cid = cid_match.group(1)
            row = preds[preds['Container_ID'].astype(str) == cid]
            if len(row) > 0:
                r = row.iloc[0]
                level_emoji = {'Critical': '🔴', 'Low Risk': '🟡', 'Clear': '🟢'}.get(r['Risk_Level'], '')
                reply = (f"**Container {cid}** {level_emoji}\n"
                         f"• Risk Score: **{r['Risk_Score']:.1f}** / 100\n"
                         f"• Risk Level: **{r['Risk_Level']}**\n")
                if 'Origin_Country' in r and pd.notna(r.get('Origin_Country')):
                    reply += f"• Origin: {r['Origin_Country']}\n"
                if 'Destination_Port' in r and pd.notna(r.get('Destination_Port')):
                    reply += f"• Destination: {r['Destination_Port']}\n"
                if 'Dwell_Time_Hours' in r and pd.notna(r.get('Dwell_Time_Hours')):
                    reply += f"• Dwell Time: {r['Dwell_Time_Hours']:.1f} hours\n"
                if 'Declared_Value' in r and pd.notna(r.get('Declared_Value')):
                    val = r['Declared_Value']
                    reply += f"• Declared Value: ${val:,.0f}\n"
                if 'Explanation' in r and pd.notna(r.get('Explanation')):
                    reply += f"• Reason: {r['Explanation']}"
                return reply
            else:
                return f"Container **{cid}** not found in the current dataset of {total:,} containers."
        else:
            return ("Please provide a container ID number. Example: \"Tell me about container 47386228\"")

    # ── Summary / overview ──
    if re.search(r'\b(summary|overview|status|dashboard|report|how many total)\b', low):
        crit_pct = round(critical / total * 100, 2)
        return (f"📊 **Dataset Summary**\n"
                f"• Total Containers: **{total:,}**\n"
                f"• 🔴 Critical: **{critical:,}** ({crit_pct}%)\n"
                f"• 🟡 Low Risk: **{low_risk:,}** ({round(low_risk/total*100,2)}%)\n"
                f"• 🟢 Clear: **{clear:,}** ({round(clear/total*100,2)}%)\n"
                f"• Average Risk Score: **{avg_score}**")

    # ── Critical count ──
    if re.search(r'\b(critical|dangerous|high risk|highest risk|red)\b', low) and re.search(r'\b(how many|count|number|total)\b', low):
        return f"There are **{critical:,}** Critical containers out of {total:,} total ({round(critical/total*100,2)}%)."

    # ── Low risk count ──
    if re.search(r'\b(low risk|medium|yellow)\b', low) and re.search(r'\b(how many|count|number|total)\b', low):
        return f"There are **{low_risk:,}** Low Risk containers out of {total:,} total ({round(low_risk/total*100,2)}%)."

    # ── Clear count ──
    if re.search(r'\b(clear|safe|green|clean)\b', low) and re.search(r'\b(how many|count|number|total)\b', low):
        return f"There are **{clear:,}** Clear containers out of {total:,} total ({round(clear/total*100,2)}%)."

    # ── Average score ──
    if re.search(r'\b(average|avg|mean)\b', low) and re.search(r'\b(score|risk)\b', low):
        return f"The average risk score across all {total:,} containers is **{avg_score}** / 100."

    # ── Top critical containers ──
    if re.search(r'\b(top|worst|most dangerous|highest|riskiest)\b', low) and re.search(r'\b(container|risk)\b', low):
        top = preds.nlargest(5, 'Risk_Score')
        lines = ["**Top 5 Highest Risk Containers:**"]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            origin = r.get('Origin_Country', '?')
            lines.append(f"{i}. Container **{r['Container_ID']}** — Score: {r['Risk_Score']:.1f}, "
                         f"Level: {r['Risk_Level']}, Origin: {origin}")
        return '\n'.join(lines)

    # ── Country analysis ──
    if re.search(r'\b(countr|origin|nation|region)\b', low):
        if 'Origin_Country' in preds.columns:
            # Check for a specific country code
            cc_match = re.search(r'\b([A-Z]{2})\b', msg)
            if cc_match:
                cc = cc_match.group(1)
                c_data = preds[preds['Origin_Country'] == cc]
                if len(c_data) > 0:
                    c_crit = int((c_data['Risk_Level'] == 'Critical').sum())
                    c_low = int((c_data['Risk_Level'] == 'Low Risk').sum())
                    c_clear = int((c_data['Risk_Level'] == 'Clear').sum())
                    c_avg = round(float(c_data['Risk_Score'].mean()), 1)
                    return (f"🌍 **Country: {cc}**\n"
                            f"• Total Containers: {len(c_data):,}\n"
                            f"• 🔴 Critical: {c_crit}\n"
                            f"• 🟡 Low Risk: {c_low}\n"
                            f"• 🟢 Clear: {c_clear}\n"
                            f"• Avg Risk Score: {c_avg}")
                else:
                    return f"No containers found from country **{cc}**."

            # Top critical origins
            crit_df = preds[preds['Risk_Level'] == 'Critical']
            top_origins = crit_df['Origin_Country'].value_counts().head(8)
            lines = ["**Top Countries by Critical Containers:**"]
            for country, count in top_origins.items():
                lines.append(f"• **{country}**: {count} critical containers")
            return '\n'.join(lines)
        return "Origin country data is not available in the current dataset."

    # ── Port analysis ──
    if re.search(r'\b(port|destination|dock)\b', low):
        if 'Destination_Port' in preds.columns:
            port_risk = preds.groupby('Destination_Port')['Risk_Score'].agg(['mean','count']).nlargest(8, 'mean')
            lines = ["**Top 8 High-Risk Ports:**"]
            for port, row in port_risk.iterrows():
                lines.append(f"• **{port}**: Avg Score {row['mean']:.1f}, {int(row['count'])} containers")
            return '\n'.join(lines)
        return "Destination port data is not available."

    # ── Feature importance ──
    if re.search(r'\b(feature|importance|important|factor|what matters|why|cause)\b', low):
        feat_imp = get_feature_importances()
        lines = ["**Top Risk Features (RF Importance):**"]
        for f in feat_imp[:8]:
            lines.append(f"• **{f['name']}**: {f['value']}%")
        lines.append("\nWeight discrepancy and dwell time are the strongest risk signals.")
        return '\n'.join(lines)

    # ── Model info ──
    if re.search(r'\b(model|algorithm|accuracy|f1|precision|recall|machine learning|ml|how.*work)\b', low):
        return ("🤖 **Model Information**\n"
                "• **Primary**: Random Forest (300 estimators, max_depth=30)\n"
                "• **Anomaly**: Isolation Forest (contamination=3%, 200 estimators)\n"
                "• **Risk Score**: 70% RF probability + 30% Anomaly score\n"
                "• **Accuracy**: 99.82% (test), 99.98% (validation)\n"
                "• **F1 Critical**: 96.68%\n"
                "• **F1 Low Risk**: 99.62%\n"
                "• **Features**: 15 engineered features\n"
                "• **Classes**: Critical, Low Risk, Clear\n"
                "• **Training**: SMOTE-balanced sampling")

    # ── Thresholds ──
    if re.search(r'\b(threshold|criteria|rules|what makes|when is|classify)\b', low):
        return ("**Risk Classification Thresholds:**\n"
                "• 🔴 **Critical**: Score ≥ 50 — immediate inspection needed\n"
                "• 🟡 **Low Risk**: Score 20–49 — enhanced monitoring\n"
                "• 🟢 **Clear**: Score < 20 — routine processing\n\n"
                "**Key Flags:**\n"
                "• Weight discrepancy > 20% → suspicious\n"
                "• Dwell time > 72h → high dwell flag\n"
                "• Dwell time > 120h → very high dwell\n"
                "• High-risk origins: CN, RO, VN, ID score higher")

    # ── Weight discrepancy ──
    if re.search(r'\b(weight|discrepancy|mismatch)\b', low):
        if 'Declared_Weight' in preds.columns and 'Measured_Weight' in preds.columns:
            preds_c = preds.copy()
            preds_c['wt_diff'] = ((preds_c['Measured_Weight'] - preds_c['Declared_Weight']).abs() / preds_c['Declared_Weight'] * 100)
            high_disc = preds_c[preds_c['wt_diff'] > 20]
            return (f"**Weight Discrepancy Analysis:**\n"
                    f"• Containers with >20% discrepancy: **{len(high_disc):,}**\n"
                    f"• Average discrepancy: {preds_c['wt_diff'].mean():.1f}%\n"
                    f"• Max discrepancy: {preds_c['wt_diff'].max():.1f}%\n"
                    f"• Weight discrepancy is the #1 risk feature in the model.")
        return "Weight data is not available in the current dataset."

    # ── Dwell time ──
    if re.search(r'\b(dwell|time|hours|wait|delay)\b', low):
        if 'Dwell_Time_Hours' in preds.columns:
            avg_dwell = preds['Dwell_Time_Hours'].mean()
            high_dwell = int((preds['Dwell_Time_Hours'] > 72).sum())
            very_high = int((preds['Dwell_Time_Hours'] > 120).sum())
            return (f"**Dwell Time Analysis:**\n"
                    f"• Average dwell: {avg_dwell:.1f} hours\n"
                    f"• High dwell (>72h): **{high_dwell:,}** containers\n"
                    f"• Very high dwell (>120h): **{very_high:,}** containers\n"
                    f"• Max dwell: {preds['Dwell_Time_Hours'].max():.1f} hours")
        return "Dwell time data is not available."

    # ── Export ──
    if re.search(r'\b(export|download|csv|save)\b', low):
        return ("You can export all predictions as CSV:\n"
                "• Click the **⬇ Export CSV** button on the Predictions page\n"
                "• Or use the API directly: `/api/export`")

    # ── Import ──
    if re.search(r'\b(import|upload|new data|add data)\b', low):
        return ("To import new data:\n"
                "• Go to the **Import CSV** page using the sidebar\n"
                "• Drag & drop or browse for a CSV file\n"
                "• Required columns: Container_ID, Declared_Weight, Measured_Weight, "
                "Dwell_Time_Hours, Declared_Value, HS_Code, Origin_Country, Destination_Port, Trade_Regime")

    # ── Fallback ──
    return (f"I'm not sure how to answer that. I can help with:\n"
            f"• Container lookups (give me an ID number)\n"
            f"• Risk statistics and summaries\n"
            f"• Country and port analysis\n"
            f"• Feature importance and model info\n"
            f"• Thresholds and classification rules\n"
            f"• Weight and dwell time analysis\n\n"
            f"Try: \"Give me a summary\" or \"Tell me about container 47386228\"")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
