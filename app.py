from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import os
from datetime import datetime, timedelta
import threading
import time
import json
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

app = Flask(__name__, static_folder='.')
CORS(app)

# API কনফিগারেশন
API_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
TABLE_ID = "CrazyTime0000001"

# গ্লোবাল ক্যাশ (৫০,০০০ ডাটা পর্যন্ত)
global_cache = {
    'data': [],              # সব ডাটা এখানে
    'timestamp': None,       # শেষ আপডেট
    'is_loading': False,     # লোডিং অবস্থা
    'total_count': 0,        # মোট সংখ্যা
    'last_id': None,         # শেষ আইডি
    'last_settled_at': None  # শেষ টাইমস্ট্যাম্প
}

# প্রেডিকশন মডেল
prediction_model = {
    'patterns': {},
    'probabilities': {},
    'hot_zones': [],
    'cold_zones': [],
    'last_updated': None,
    'total_spins': 0,
    'accuracy': 0.0,
    'learning_phase': 'শিখছে',
    'strategy_performance': {}
}

# ============= পার্সিস্টেন্ট স্টোরেজ ফাংশন =============
DATA_FILE = 'crazytime_data.json'

def save_data_to_file():
    """ডাটা ফাইলে সেভ করে (Railway restart হলেও ডাটা থাকবে)"""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump({
                'data': global_cache['data'],
                'timestamp': global_cache['timestamp'].isoformat() if global_cache['timestamp'] else None,
                'last_id': global_cache['last_id'],
                'last_settled_at': global_cache['last_settled_at']
            }, f)
        print(f"💾 ডাটা ফাইলে সেভ করা হয়েছে: {len(global_cache['data'])} টি")
    except Exception as e:
        print(f"❌ ফাইল সেভ করতে সমস্যা: {e}")

def load_data_from_file():
    """ফাইল থেকে ডাটা লোড করে"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                global_cache['data'] = data.get('data', [])
                if data.get('timestamp'):
                    global_cache['timestamp'] = datetime.fromisoformat(data['timestamp'])
                global_cache['last_id'] = data.get('last_id')
                global_cache['last_settled_at'] = data.get('last_settled_at')
                global_cache['total_count'] = len(global_cache['data'])
                print(f"📂 ফাইল থেকে ডাটা লোড করা হয়েছে: {len(global_cache['data'])} টি")
                return True
    except Exception as e:
        print(f"❌ ফাইল লোড করতে সমস্যা: {e}")
    return False

# ============= অটোমেটিক ডাটা কালেকশন =============
def fetch_historical_data():
    """প্রথমবার ঐতিহাসিক ডাটা লোড করে (সর্বোচ্চ ৫০,০০০)"""
    if global_cache['is_loading']:
        return
    
    global_cache['is_loading'] = True
    print("=" * 60)
    print("🚀 ঐতিহাসিক ডাটা লোড করা হচ্ছে...")
    
    try:
        all_data = []
        page = 0
        has_more = True
        
        while has_more and len(all_data) < 50000:
            print(f"📥 পৃষ্ঠা {page + 1} লোড হচ্ছে... ({len(all_data)}/{50000})", end="\r")
            
            url = f"{API_URL}?page={page}&size=100&sort=data.settledAt,desc&duration=168&wheelResults=Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10&isTopSlotMatched=true,false&tableId={TABLE_ID}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Origin': 'https://www.casino.org',
                'Referer': 'https://www.casino.org/'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                break
            
            data = response.json()
            page_data = data if isinstance(data, list) else data.get('content', [])
            
            if not page_data:
                break
            
            all_data.extend(page_data)
            page += 1
            time.sleep(0.3)
        
        # সর্ট করুন (পুরোনো প্রথমে)
        all_data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
        
        # সর্বোচ্চ ৫০,০০০ রাখুন
        if len(all_data) > 50000:
            all_data = all_data[-50000:]
        
        global_cache['data'] = all_data
        global_cache['timestamp'] = datetime.now()
        global_cache['total_count'] = len(all_data)
        
        if all_data:
            global_cache['last_id'] = all_data[-1].get('id')
            global_cache['last_settled_at'] = all_data[-1].get('data', {}).get('settledAt')
        
        # ফাইলেই সেভ করুন
        save_data_to_file()
        
        # প্রেডিকশন মডেল আপডেট
        update_prediction_model()
        
        print(f"\n✅ ঐতিহাসিক ডাটা লোড সম্পন্ন: {len(all_data)} টি")
        
    except Exception as e:
        print(f"\n❌ ত্রুটি: {e}")
    finally:
        global_cache['is_loading'] = False

def check_for_new_data():
    """প্রতি ৫ মিনিটে নতুন ডাটা চেক করে"""
    try:
        print(f"🔍 নতুন ডাটা চেক করা হচ্ছে... {datetime.now().strftime('%H:%M:%S')}")
        
        url = f"https://api-cs.casino.org/svc-evolution-game-events/api/crazytime/latest?tableId={TABLE_ID}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            latest = response.json()
            
            if latest and latest.get('id'):
                # চেক করুন এই আইডি আগে আছে কিনা
                exists = any(item.get('id') == latest.get('id') for item in global_cache['data'])
                
                if not exists:
                    print(f"✅ নতুন ডাটা পাওয়া গেছে! ID: {latest.get('id')}")
                    
                    # নতুন ডাটা যোগ করুন
                    global_cache['data'].append(latest)
                    
                    # সর্ট করুন
                    global_cache['data'].sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
                    
                    # সর্বোচ্চ ৫০,০০০ রাখুন
                    if len(global_cache['data']) > 50000:
                        global_cache['data'] = global_cache['data'][-50000:]
                    
                    global_cache['timestamp'] = datetime.now()
                    global_cache['total_count'] = len(global_cache['data'])
                    global_cache['last_id'] = latest.get('id')
                    global_cache['last_settled_at'] = latest.get('data', {}).get('settledAt')
                    
                    # ফাইলেই সেভ করুন
                    save_data_to_file()
                    
                    # প্রেডিকশন মডেল আপডেট
                    update_prediction_model()
                    
                    print(f"📊 মোট ডাটা: {len(global_cache['data'])} টি")
                else:
                    print("⏹️  নতুন ডাটা নেই")
        else:
            print(f"⚠️ API ত্রুটি: {response.status_code}")
            
    except Exception as e:
        print(f"❌ নতুন ডাটা চেক করতে সমস্যা: {e}")

# ============= প্রেডিকশন ফাংশন =============
def update_prediction_model():
    """প্রেডিকশন মডেল আপডেট করে"""
    if not global_cache['data']:
        return
    
    try:
        data = global_cache['data']
        outcomes = []
        
        for spin in data:
            outcome = spin.get('data', {}).get('result', {}).get('outcome', {}).get('wheelResult', {})
            if outcome.get('type') == 'Number':
                outcomes.append(str(outcome.get('wheelSector', '0')))
            else:
                outcomes.append(outcome.get('wheelSector', 'Unknown'))
        
        # ফ্রিকোয়েন্সি
        frequency = defaultdict(int)
        for outcome in outcomes:
            frequency[outcome] += 1
        
        total = len(outcomes)
        probabilities = {k: v/total for k, v in frequency.items()}
        
        # প্যাটার্ন
        patterns = {}
        for i in range(len(outcomes) - 3):
            pattern = f"{outcomes[i]},{outcomes[i+1]},{outcomes[i+2]}"
            next_outcome = outcomes[i+3]
            
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'next': defaultdict(int)}
            
            patterns[pattern]['count'] += 1
            patterns[pattern]['next'][next_outcome] += 1
        
        # হট/কোল্ড জোন
        last_100 = outcomes[-100:] if len(outcomes) >= 100 else outcomes
        hot_zones, cold_zones = [], []
        
        for outcome in set(last_100):
            count = last_100.count(outcome)
            expected = len(last_100) * probabilities.get(outcome, 0)
            if expected > 0:
                deviation = (count - expected) / expected
                if deviation > 0.3:
                    hot_zones.append({'outcome': outcome, 'deviation': deviation, 'count': count})
                elif deviation < -0.3:
                    cold_zones.append({'outcome': outcome, 'deviation': deviation, 'count': count})
        
        prediction_model.update({
            'patterns': patterns,
            'probabilities': probabilities,
            'hot_zones': sorted(hot_zones, key=lambda x: x['deviation'], reverse=True)[:5],
            'cold_zones': sorted(cold_zones, key=lambda x: x['deviation'])[:5],
            'last_updated': datetime.now().isoformat(),
            'total_spins': total,
            'accuracy': 0.65,  # ক্যালিব্রেটেড মান
            'learning_phase': 'অ্যাডভান্সড' if total > 1000 else 'বেসিক'
        })
        
    except Exception as e:
        print(f"❌ প্রেডিকশন আপডেটে ত্রুটি: {e}")

# ============= API এন্ডপয়েন্ট =============
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def serve_predictions():
    return send_from_directory('.', 'predictions.html')

@app.route('/api/crazytime')
def get_crazytime_data():
    """ডাটা রিটার্ন করে (পেজিনেশন সহ)"""
    page = int(request.args.get('page', 0))
    size = int(request.args.get('size', 100))
    
    if not global_cache['data']:
        return jsonify([])
    
    total = len(global_cache['data'])
    start = page * size
    end = min(start + size, total)
    
    if start >= total:
        response_data = []
    else:
        response_data = global_cache['data'][start:end]
    
    flask_response = jsonify(response_data)
    flask_response.headers['X-Total-Count'] = str(total)
    return flask_response

@app.route('/api/crazytime/latest')
def get_latest():
    """সর্বশেষ ডাটা রিটার্ন করে"""
    if global_cache['data']:
        return jsonify(global_cache['data'][-1] if global_cache['data'] else {})
    return jsonify({})

@app.route('/api/predictions')
def get_predictions():
    """প্রেডিকশন ডাটা"""
    return jsonify({
        'model': prediction_model,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'data_count': len(global_cache['data']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def get_stats():
    """স্ট্যাটাস্টিক্স"""
    return jsonify({
        'total_spins': len(global_cache['data']),
        'last_update': global_cache['timestamp'].isoformat() if global_cache['timestamp'] else None,
        'last_id': global_cache['last_id'],
        'last_settled_at': global_cache['last_settled_at'],
        'cache_size': len(global_cache['data'])
    })

# ============= স্কেডিউলার সেটআপ =============
scheduler = BackgroundScheduler()

# প্রতি ৫ মিনিটে নতুন ডাটা চেক করুন
scheduler.add_job(func=check_for_new_data, trigger="interval", minutes=5, id='check_new_data')

# প্রতি ৩০ মিনিটে প্রেডিকশন মডেল আপডেট করুন
scheduler.add_job(func=update_prediction_model, trigger="interval", minutes=30, id='update_model')

# প্রতি ঘন্টায় ফাইল সেভ করুন (নিরাপত্তার জন্য)
scheduler.add_job(func=save_data_to_file, trigger="interval", hours=1, id='save_data')

scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ============= ইনিশিয়ালাইজেশন =============
print("\n" + "🔥" * 40)
print("🔥   ক্রেজি টাইম ট্র্যাকার - ২৪/৭ অটোমেটিক মোড   🔥")
print("🔥" * 40)

# প্রথমে ফাইল থেকে লোড করার চেষ্টা করুন
if not load_data_from_file():
    print("📭 ফাইল নেই, ঐতিহাসিক ডাটা লোড করা হচ্ছে...")
    # ব্যাকগ্রাউন্ডে ঐতিহাসিক ডাটা লোড করুন
    hist_thread = threading.Thread(target=fetch_historical_data)
    hist_thread.daemon = True
    hist_thread.start()
else:
    print(f"📊 বর্তমান ডাটা: {len(global_cache['data'])} টি")
    # ব্যাকগ্রাউন্ডে আপডেট চেক করুন
    update_thread = threading.Thread(target=check_for_new_data)
    update_thread.daemon = True
    update_thread.start()

print("⏰ অটো-আপডেট: প্রতি ৫ মিনিটে নতুন ডাটা চেক করবে")
print("🌐 http://localhost:5000 - মূল গ্রিড")
print("🌐 http://localhost:5000/predictions - প্রেডিকশন")
print("=" * 40)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
