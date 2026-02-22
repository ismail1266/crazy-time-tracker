from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import requests
import os
from datetime import datetime, timedelta
import threading
import time
import json
from collections import defaultdict

app = Flask(__name__, static_folder='.')
CORS(app)  # সব CORS সমস্যা সমাধান

# API কনফিগারেশন
API_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
TABLE_ID = "CrazyTime0000001"

# গ্লোবাল ক্যাশ (সার্ভার সাইড ক্যাশ)
global_cache = {
    'data': None,           # সম্পূর্ণ ৭২ ঘন্টার ডাটা
    'timestamp': None,      # ক্যাশ তৈরির সময়
    'is_loading': False,    # লোডিং অবস্থা
    'total_count': 0,        # মোট রেজাল্ট সংখ্যা
    'last_id': None         # শেষ আইডি
}

# প্রেডিকশন মডেল ডাটা
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

def fetch_all_crazytime_data():
    """পটভূমিতে ৭২ ঘন্টার সকল ডাটা সংগ্রহ করে"""
    global global_cache
    
    if global_cache['is_loading']:
        print("⏳ ইতিমধ্যে ডাটা লোড হচ্ছে...")
        return
    
    global_cache['is_loading'] = True
    print("=" * 60)
    print("🚀 ক্রেজি টাইম ডাটা কালেক্টর শুরু হচ্ছে...")
    print("=" * 60)
    
    try:
        all_data = []
        page = 0
        has_more = True
        empty_page_count = 0
        total_pages = 0
        
        start_time = time.time()
        
        while has_more and page < 500 and empty_page_count < 3:
            print(f"📥 পৃষ্ঠা {page + 1} লোড হচ্ছে... (প্রতি পৃষ্ঠায় ১০০টি)", end="\r")
            
            url = f"{API_URL}?page={page}&size=100&sort=data.settledAt,desc&duration=72&wheelResults=Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10&isTopSlotMatched=true,false&tableId={TABLE_ID}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Origin': 'https://www.casino.org',
                'Referer': 'https://www.casino.org/'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"\n⚠️ API ত্রুটি: {response.status_code}")
                break
            
            data = response.json()
            
            if isinstance(data, list):
                page_data = data
            elif isinstance(data, dict):
                page_data = data.get('content', [])
            else:
                page_data = []
            
            if len(page_data) == 0:
                empty_page_count += 1
                if empty_page_count >= 3:
                    print(f"\n📭 পরপর ৩টি খালি পৃষ্ঠা, থামানো হচ্ছে...")
                    has_more = False
            else:
                all_data.extend(page_data)
                page += 1
                total_pages += 1
                empty_page_count = 0
            
            time.sleep(0.3)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"✅ ডাটা সংগ্রহ সম্পন্ন!")
        print(f"📊 মোট পৃষ্ঠা: {total_pages}")
        print(f"📦 মোট রেজাল্ট: {len(all_data)} টি")
        print(f"⏱️  সময় লেগেছে: {elapsed_time:.1f} সেকেন্ড")
        
        # ডাটা সর্ট করুন (পুরোনো প্রথমে)
        print("🔄 ডাটা সাজানো হচ্ছে...")
        all_data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
        
        # ক্যাশে সংরক্ষণ
        global_cache['data'] = all_data
        global_cache['timestamp'] = datetime.now()
        global_cache['total_count'] = len(all_data)
        if len(all_data) > 0:
            global_cache['last_id'] = all_data[-1].get('id')
        
        print(f"💾 ক্যাশে সংরক্ষিত: {len(all_data)} টি রেজাল্ট")
        
        # প্রেডিকশন মডেল আপডেট
        update_prediction_model()
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ গুরুতর ত্রুটি: {e}")
        import traceback
        traceback.print_exc()
    finally:
        global_cache['is_loading'] = False

def update_prediction_model():
    """প্রেডিকশন মডেল আপডেট করে"""
    global prediction_model, global_cache
    
    if not global_cache['data']:
        return
    
    data = global_cache['data']
    
    try:
        print("🧠 প্রেডিকশন মডেল আপডেট করা হচ্ছে...")
        
        # বেসিক পরিসংখ্যান
        total_spins = len(data)
        outcomes = []
        
        for spin in data:
            outcome = spin.get('data', {}).get('result', {}).get('outcome', {}).get('wheelResult', {})
            if outcome.get('type') == 'Number':
                outcomes.append(str(outcome.get('wheelSector', '0')))
            else:
                outcomes.append(outcome.get('wheelSector', 'Unknown'))
        
        # ফ্রিকোয়েন্সি ক্যালকুলেশন
        frequency = defaultdict(int)
        for outcome in outcomes:
            frequency[outcome] += 1
        
        # প্রোবেবিলিটি ক্যালকুলেশন
        probabilities = {}
        for outcome, count in frequency.items():
            probabilities[outcome] = count / total_spins
        
        # প্যাটার্ন ডিটেকশন (শর্ট টার্ম)
        patterns = {}
        for i in range(len(outcomes) - 3):
            pattern = f"{outcomes[i]},{outcomes[i+1]},{outcomes[i+2]}"
            next_outcome = outcomes[i+3]
            
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'next': defaultdict(int)}
            
            patterns[pattern]['count'] += 1
            patterns[pattern]['next'][next_outcome] += 1
        
        # হট/কোল্ড জোন (শেষ ১০০ স্পিন)
        last_100 = outcomes[-100:] if len(outcomes) >= 100 else outcomes
        hot_zones = []
        cold_zones = []
        
        for outcome in set(last_100):
            count = last_100.count(outcome)
            expected = len(last_100) * probabilities.get(outcome, 0)
            deviation = (count - expected) / expected if expected > 0 else 0
            
            if deviation > 0.3:
                hot_zones.append({'outcome': outcome, 'deviation': deviation, 'count': count})
            elif deviation < -0.3:
                cold_zones.append({'outcome': outcome, 'deviation': deviation, 'count': count})
        
        # স্ট্র্যাটেজি পারফরমেন্স
        strategy_performance = {
            'hot_following': calculate_strategy_accuracy('hot_following', outcomes),
            'cold_tracking': calculate_strategy_accuracy('cold_tracking', outcomes),
            'pattern_matching': calculate_strategy_accuracy('pattern_matching', outcomes),
            'gap_trading': calculate_strategy_accuracy('gap_trading', outcomes)
        }
        
        # মডেল আপডেট
        prediction_model.update({
            'patterns': patterns,
            'probabilities': probabilities,
            'hot_zones': sorted(hot_zones, key=lambda x: x['deviation'], reverse=True)[:5],
            'cold_zones': sorted(cold_zones, key=lambda x: x['deviation'])[:5],
            'last_updated': datetime.now().isoformat(),
            'total_spins': total_spins,
            'accuracy': calculate_model_accuracy(patterns, outcomes[-100:]),
            'learning_phase': 'অপারেশনাল',
            'strategy_performance': strategy_performance
        })
        
        print(f"✅ প্রেডিকশন মডেল আপডেটেড: {total_spins} স্পিন বিশ্লেষিত")
        
    except Exception as e:
        print(f"❌ প্রেডিকশন মডেল আপডেটে ত্রুটি: {e}")

def calculate_strategy_accuracy(strategy, outcomes):
    """প্রতিটি স্ট্র্যাটেজির অ্যাকুরেসি ক্যালকুলেট করে"""
    if len(outcomes) < 20:
        return 0.5
    
    correct = 0
    total = 0
    
    for i in range(len(outcomes) - 1):
        prediction = None
        
        if strategy == 'hot_following' and i > 10:
            # হট ফলোয়িং: গত ১০ স্পিনের মোস্ট কমন
            last_10 = outcomes[i-10:i]
            if last_10:
                prediction = max(set(last_10), key=last_10.count)
                
        elif strategy == 'cold_tracking' and i > 20:
            # কোল্ড ট্র্যাকিং: সবচেয়ে কম আসা
            last_20 = outcomes[i-20:i]
            if last_20:
                counts = defaultdict(int)
                for o in last_20:
                    counts[o] += 1
                prediction = min(counts, key=counts.get)
                
        elif strategy == 'pattern_matching' and i > 3:
            # প্যাটার্ন ম্যাচিং
            last_3 = outcomes[i-3:i]
            pattern_key = f"{last_3[0]},{last_3[1]},{last_3[2]}"
            if pattern_key in prediction_model.get('patterns', {}):
                pattern_data = prediction_model['patterns'][pattern_key]
                if pattern_data['next']:
                    prediction = max(pattern_data['next'], key=pattern_data['next'].get)
        
        if prediction:
            total += 1
            if prediction == outcomes[i+1]:
                correct += 1
    
    return correct / total if total > 0 else 0.5

def calculate_model_accuracy(patterns, recent_outcomes):
    """মডেলের সামগ্রিক অ্যাকুরেসি ক্যালকুলেট করে"""
    if len(recent_outcomes) < 10:
        return 0.5
    
    correct = 0
    total = 0
    
    for i in range(len(recent_outcomes) - 1):
        if i >= 3:
            last_3 = recent_outcomes[i-3:i]
            pattern_key = f"{last_3[0]},{last_3[1]},{last_3[2]}"
            
            if pattern_key in patterns:
                pattern_data = patterns[pattern_key]
                if pattern_data['next']:
                    prediction = max(pattern_data['next'], key=pattern_data['next'].get)
                    total += 1
                    if prediction == recent_outcomes[i+1]:
                        correct += 1
    
    return correct / total if total > 0 else 0.5

def get_next_prediction():
    """পরবর্তী স্পিনের জন্য প্রেডিকশন জেনারেট করে"""
    global global_cache, prediction_model
    
    if not global_cache['data'] or len(global_cache['data']) < 10:
        return {
            'primary': 'অপর্যাপ্ত ডাটা',
            'confidence': 0,
            'alternatives': [],
            'probabilities': {}
        }
    
    outcomes = []
    for spin in global_cache['data'][-20:]:  # শেষ ২০ স্পিন
        outcome = spin.get('data', {}).get('result', {}).get('outcome', {}).get('wheelResult', {})
        if outcome.get('type') == 'Number':
            outcomes.append(str(outcome.get('wheelSector', '0')))
        else:
            outcomes.append(outcome.get('wheelSector', 'Unknown'))
    
    # প্যাটার্ন ম্যাচিং
    if len(outcomes) >= 3:
        last_3 = outcomes[-3:]
        pattern_key = f"{last_3[0]},{last_3[1]},{last_3[2]}"
        
        if pattern_key in prediction_model.get('patterns', {}):
            pattern_data = prediction_model['patterns'][pattern_key]
            if pattern_data['next']:
                # সম্ভাবনা অনুযায়ী সাজান
                total = sum(pattern_data['next'].values())
                probabilities = {k: v/total for k, v in pattern_data['next'].items()}
                
                # সর্টেড প্রেডিকশন
                sorted_preds = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_preds:
                    return {
                        'primary': sorted_preds[0][0],
                        'confidence': sorted_preds[0][1],
                        'alternatives': [p[0] for p in sorted_preds[1:3]],
                        'probabilities': probabilities,
                        'pattern': pattern_key
                    }
    
    # যদি প্যাটার্ন না মেলে, বেস প্রোবেবিলিটি ব্যবহার করুন
    probs = prediction_model.get('probabilities', {})
    if probs:
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return {
            'primary': sorted_probs[0][0],
            'confidence': sorted_probs[0][1],
            'alternatives': [p[0] for p in sorted_probs[1:3]],
            'probabilities': probs,
            'pattern': None
        }
    
    return {
        'primary': 'অনির্ধারিত',
        'confidence': 0,
        'alternatives': [],
        'probabilities': {}
    }

# সার্ভার স্টার্ট হওয়ার সাথে সাথে ডাটা সংগ্রহ শুরু করুন
print("\n" + "🔥" * 30)
print("🔥   ক্রেজি টাইম ট্র্যাকার সার্ভার   🔥")
print("🔥" * 30)
print("⏳ সার্ভার প্রস্তুত হচ্ছে...")

# পটভূমিতে ডাটা সংগ্রহ শুরু
thread = threading.Thread(target=fetch_all_crazytime_data)
thread.daemon = True
thread.start()

print("🌐 ওয়েব সার্ভার চালু হচ্ছে...")
print("📍 http://localhost:5000 - মূল গ্রিড")
print("📍 http://localhost:5000/predictions - প্রেডিকশন ড্যাশবোর্ড")
print("=" * 60)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def serve_predictions():
    return send_from_directory('.', 'predictions.html')

@app.route('/api/crazytime')
def proxy_crazytime():
    try:
        page = request.args.get('page', '0')
        size = request.args.get('size', '100')
        
        if page == '0' and global_cache['data'] is not None:
            print(f"📦 ক্যাশ থেকে ডাটা দিচ্ছি (পৃষ্ঠা {page})")
            
            page_size = int(size)
            total = len(global_cache['data'])
            
            start = 0
            end = min(page_size, total)
            
            response_data = global_cache['data'][start:end]
            
            flask_response = jsonify(response_data)
            flask_response.headers['X-Total-Count'] = str(total)
            return flask_response
        
        print(f"🌐 API থেকে ডাটা নিচ্ছি (পৃষ্ঠা {page})")
        
        url = f"{API_URL}?page={page}&size={size}&sort=data.settledAt,desc&duration=72&wheelResults=Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10&isTopSlotMatched=true,false&tableId={TABLE_ID}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': f'API returned {response.status_code}'}), response.status_code
        
        flask_response = jsonify(response.json())
        if 'x-total-count' in response.headers:
            flask_response.headers['X-Total-Count'] = response.headers['x-total-count']
        
        return flask_response
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'API timeout'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Connection error'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crazytime/latest')
def proxy_crazytime_latest():
    """শুধু সর্বশেষ রেজাল্ট আনার জন্য"""
    try:
        url = f"https://api-cs.casino.org/svc-evolution-game-events/api/crazytime/latest?tableId={TABLE_ID}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': f'Latest API returned {response.status_code}'}), response.status_code
        
        data = response.json()
        
        if data and data.get('id'):
            print(f"📡 লেটেস্ট রেজাল্ট: {data.get('id')}")
            
            # নতুন ডাটা এলে প্রেডিকশন মডেল আপডেট করুন
            if global_cache['data']:
                # চেক করুন নতুন কি না
                exists = any(item.get('id') == data.get('id') for item in global_cache['data'])
                if not exists:
                    global_cache['data'].append(data)
                    global_cache['data'].sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
                    print("🧠 নতুন ডাটা পেয়েছি, প্রেডিকশন মডেল আপডেট করা হচ্ছে...")
                    update_prediction_model()
        
        return jsonify(data), 200
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Latest API timeout'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Connection error'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def get_predictions():
    """প্রেডিকশন ডাটা রিটার্ন করে"""
    global prediction_model
    
    next_pred = get_next_prediction()
    
    response = {
        'model': prediction_model,
        'next_prediction': next_pred,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(response)

@app.route('/api/predictions/update', methods=['POST'])
def force_update_predictions():
    """ম্যানুয়ালি প্রেডিকশন আপডেট করে"""
    update_prediction_model()
    return jsonify({'status': 'success', 'message': 'প্রেডিকশন মডেল আপডেট করা হয়েছে'})

@app.route('/api/cache/status')
def cache_status():
    """ক্যাশের অবস্থা দেখার জন্য"""
    return jsonify({
        'has_data': global_cache['data'] is not None,
        'count': len(global_cache['data']) if global_cache['data'] else 0,
        'timestamp': global_cache['timestamp'].isoformat() if global_cache['timestamp'] else None,
        'is_loading': global_cache['is_loading'],
        'last_id': global_cache['last_id']
    }), 200

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok', 
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Railway $PORT ইউজ করবে
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False করতে হবে
