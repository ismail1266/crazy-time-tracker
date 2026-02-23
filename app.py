# filename: app.py
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import os
from datetime import datetime, timedelta
import threading
import time
import json
import pickle
from collections import defaultdict
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import logging

# লগিং সেটআপ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
CORS(app)

# ==================== কনফিগারেশন ====================
API_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
TABLE_ID = "CrazyTime0000001"
DATA_FILE = "crazytime_data.pkl"
MODEL_FILE = "ai_model.pkl"
UPDATE_INTERVAL = 5  # ৫ সেকেন্ড

# ==================== গ্লোবাল স্ট্রাকচার ====================
class CrazyTimeTracker:
    def __init__(self):
        self.data = []              # সব ডাটা
        self.last_id = None         # শেষ আইডি
        self.total_count = 0        # মোট সংখ্যা
        self.is_loading = False     # লোডিং স্ট্যাটাস
        self.last_update = None     # শেষ আপডেট সময়
        self.lock = threading.RLock() # থ্রেড সেফটি (RLock ব্যবহার করুন)
        
# ==================== AI মডেল ক্লাস ====================
class AIPredictionModel:
    def __init__(self):
        self.markov_chain = {}
        self.dealer_patterns = {}
        self.time_patterns = {}
        self.multiplier_patterns = {}
        self.cycles = {}
        self.hot_zones = []
        self.cold_zones = []
        self.probabilities = {}
        self.strategy_performance = {}
        self.accuracy = 0.5
        self.total_spins = 0
        self.training_history = []
        self.ensemble_weights = {
            'markov': 0.30,
            'dealer': 0.25,
            'multiplier': 0.20,
            'time': 0.15,
            'hot': 0.10
        }
        
    def save(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"✅ মডেল সেভ করা হয়েছে: {filename}")
        except Exception as e:
            logger.error(f"❌ মডেল সেভ করতে সমস্যা: {e}")
    
    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                logger.info(f"✅ মডেল লোড করা হয়েছে: {filename}")
                return model
        except:
            logger.info("🆕 নতুন মডেল তৈরি হচ্ছে")
            return AIPredictionModel()

# ==================== গ্লোবাল ভেরিয়েবল ====================
tracker = CrazyTimeTracker()
ai_model = AIPredictionModel.load(MODEL_FILE)
scheduler = BackgroundScheduler()

# ==================== ডাটা পার্সিস্টেন্স ====================
def save_data():
    """ডাটা ফাইলে সেভ করে"""
    try:
        with tracker.lock:
            data_to_save = {
                'data': tracker.data,
                'last_id': tracker.last_id,
                'total_count': tracker.total_count,
                'last_update': tracker.last_update.isoformat() if tracker.last_update else None
            }
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info(f"✅ ডাটা সেভ করা হয়েছে: {tracker.total_count} টি")
    except Exception as e:
        logger.error(f"❌ ডাটা সেভ করতে সমস্যা: {e}")

def load_data():
    """ডাটা ফাইল থেকে লোড করে"""
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            with tracker.lock:
                tracker.data = data.get('data', [])
                tracker.last_id = data.get('last_id')
                tracker.total_count = data.get('total_count', 0)
                last_update = data.get('last_update')
                if last_update:
                    tracker.last_update = datetime.fromisoformat(last_update)
            logger.info(f"📦 {tracker.total_count} টি ডাটা লোড করা হয়েছে")
            return True
    except FileNotFoundError:
        logger.info("🆕 কোন ডাটা ফাইল নেই, নতুন করে সংগ্রহ হবে")
        return False
    except Exception as e:
        logger.error(f"❌ ডাটা লোড করতে সমস্যা: {e}")
        return False

# ==================== ৭২ ঘন্টার ডাটা কালেকশন ====================
def fetch_72h_data():
    """সার্ভার স্টার্টে ৭২ ঘন্টার ডাটা সংগ্রহ করে"""
    if tracker.is_loading:
        logger.info("⏳ ইতিমধ্যে ডাটা লোড হচ্ছে...")
        return
    
    tracker.is_loading = True
    logger.info("=" * 60)
    logger.info("🚀 ৭২ ঘন্টার ডাটা সংগ্রহ শুরু...")
    
    try:
        all_data = []
        page = 0
        empty_pages = 0
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/'
        }
        
        while empty_pages < 3 and page < 500:
            url = f"{API_URL}?page={page}&size=100&sort=data.settledAt,desc&duration=72&wheelResults=Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10&isTopSlotMatched=true,false&tableId={TABLE_ID}"
            
            try:
                logger.info(f"📥 পৃষ্ঠা {page + 1} লোড হচ্ছে...")
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code != 200:
                    logger.warning(f"⚠️ API ত্রুটি: {response.status_code}")
                    break
                
                data = response.json()
                page_data = data if isinstance(data, list) else data.get('content', [])
                
                if len(page_data) == 0:
                    empty_pages += 1
                    logger.info(f"📭 পৃষ্ঠা {page + 1} খালি")
                else:
                    all_data.extend(page_data)
                    page += 1
                    empty_pages = 0
                    logger.info(f"✅ পৃষ্ঠা {page} থেকে {len(page_data)} টি ডাটা (মোট: {len(all_data)})")
                
                time.sleep(0.3)  # রেট লিমিট এড়াতে
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ পৃষ্ঠা {page + 1} টাইমআউট")
                break
            except Exception as e:
                logger.error(f"⚠️ পৃষ্ঠা {page + 1} এ ত্রুটি: {e}")
                break
        
        # ডাটা সর্ট (পুরোনো → নতুন)
        logger.info("🔄 ডাটা সাজানো হচ্ছে...")
        all_data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
        
        with tracker.lock:
            tracker.data = all_data
            tracker.total_count = len(all_data)
            if all_data:
                tracker.last_id = all_data[-1].get('id')
            tracker.last_update = datetime.now()
        
        # ডাটা সেভ
        save_data()
        
        logger.info(f"✅ মোট {len(all_data)} টি ডাটা সংগ্রহ সম্পন্ন")
        logger.info("=" * 60)
        
        # AI মডেল ট্রেন
        if len(all_data) >= 50:
            train_ai_model()
        
    except Exception as e:
        logger.error(f"❌ গুরুতর ত্রুটি: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.is_loading = False

# ==================== নতুন ডাটা চেক (প্রতি ৫ সেকেন্ড) ====================
def check_new_data():
    """প্রতি ৫ সেকেন্ডে নতুন ডাটা চেক করে"""
    try:
        url = f"https://api-cs.casino.org/svc-evolution-game-events/api/crazytime/latest?tableId={TABLE_ID}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            new_data = response.json()
            
            if new_data and new_data.get('id'):
                with tracker.lock:
                    # ডুপ্লিকেট চেক
                    exists = any(item.get('id') == new_data.get('id') for item in tracker.data)
                    
                    if not exists and new_data.get('data', {}).get('settledAt'):
                        logger.info(f"🆕 নতুন ডাটা: {new_data.get('id')}")
                        
                        tracker.data.append(new_data)
                        tracker.data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
                        tracker.total_count = len(tracker.data)
                        tracker.last_id = new_data.get('id')
                        tracker.last_update = datetime.now()
                        
                        # ডাটা সেভ
                        save_data()
                        
                        # AI মডেল আপডেট
                        if tracker.total_count % 10 == 0:
                            # প্রতি ১০টি নতুন ডাটায় পূর্ণ ট্রেনিং
                            threading.Thread(target=train_ai_model).start()
                        else:
                            # দ্রুত আপডেট
                            quick_update_ai(new_data)
    
    except Exception as e:
        # সাইলেন্ট ফেইল - শুধু ডিবাগে দেখাবে
        logger.debug(f"নতুন ডাটা চেকে ত্রুটি: {e}")

# ==================== AI মডেল ট্রেনিং ====================
def train_ai_model():
    """সম্পূর্ণ AI মডেল ট্রেনিং"""
    global ai_model
    
    with tracker.lock:
        if len(tracker.data) < 50:
            logger.info("⚠️ ট্রেনিংয়ের জন্য পর্যাপ্ত ডাটা নেই")
            return
        
        data_copy = tracker.data.copy()
    
    logger.info("🧠 AI মডেল ট্রেনিং শুরু...")
    
    try:
        outcomes = []
        dealers = []
        multipliers = []
        times = []
        
        for spin in data_copy:
            result_data = spin.get('data', {}).get('result', {}).get('outcome', {})
            wheel = result_data.get('wheelResult', {})
            
            # আউটকাম
            if wheel.get('type') == 'Number':
                outcome = str(wheel.get('wheelSector', '0'))
            else:
                outcome = wheel.get('wheelSector', 'Unknown')
            outcomes.append(outcome)
            
            # ডিলার
            dealers.append(spin.get('data', {}).get('dealer', {}).get('name', 'Unknown'))
            
            # মাল্টিপ্লায়ার
            multipliers.append(result_data.get('maxMultiplier', 1))
            
            # সময়
            settled = spin.get('data', {}).get('settledAt')
            if settled:
                try:
                    hour = datetime.fromisoformat(settled.replace('Z', '+00:00')).hour
                    if 6 <= hour < 12: times.append('morning')
                    elif 12 <= hour < 18: times.append('afternoon')
                    elif 18 <= hour < 24: times.append('evening')
                    else: times.append('night')
                except:
                    times.append('unknown')
            else:
                times.append('unknown')
        
        new_model = AIPredictionModel()
        
        # ১. মাল্টি-লেভেল মার্কোভ চেইন
        for level in [1, 2, 3]:
            markov = {}
            for i in range(len(outcomes) - level):
                state = '→'.join(outcomes[i:i+level])
                next_out = outcomes[i+level]
                
                if state not in markov:
                    markov[state] = {}
                if next_out not in markov[state]:
                    markov[state][next_out] = 0
                markov[state][next_out] += 1
            
            # প্রোবেবিলিটি ক্যালকুলেশন
            for state in markov:
                total = sum(markov[state].values())
                if total > 0:
                    for next_out in markov[state]:
                        markov[state][next_out] /= total
            
            new_model.markov_chain[f'level_{level}'] = markov
        
        # ২. ডিলার প্যাটার্ন
        dealer_stats = {}
        for i, dealer in enumerate(dealers):
            if dealer not in dealer_stats:
                dealer_stats[dealer] = {'total': 0, 'outcomes': {}}
            dealer_stats[dealer]['total'] += 1
            dealer_stats[dealer]['outcomes'][outcomes[i]] = \
                dealer_stats[dealer]['outcomes'].get(outcomes[i], 0) + 1
        
        for dealer in dealer_stats:
            total = dealer_stats[dealer]['total']
            if total > 0:
                for out in dealer_stats[dealer]['outcomes']:
                    dealer_stats[dealer]['outcomes'][out] /= total
        
        new_model.dealer_patterns = dealer_stats
        
        # ৩. হট/কোল্ড জোন
        if len(outcomes) >= 50:
            recent = outcomes[-50:]
            all_outcomes = set(outcomes)
            
            # বেস প্রোবেবিলিটি
            base_probs = {}
            for o in all_outcomes:
                base_probs[o] = outcomes.count(o) / len(outcomes)
            
            hot = []
            cold = []
            
            for outcome in all_outcomes:
                expected = base_probs.get(outcome, 0) * 50
                actual = recent.count(outcome)
                
                if expected > 0:
                    deviation = (actual - expected) / expected
                    if deviation > 0.2:
                        hot.append({'outcome': outcome, 'deviation': deviation, 'actual': actual})
                    elif deviation < -0.2:
                        cold.append({'outcome': outcome, 'deviation': deviation, 'actual': actual})
            
            hot.sort(key=lambda x: x['deviation'], reverse=True)
            cold.sort(key=lambda x: x['deviation'])
            
            new_model.hot_zones = hot[:5]
            new_model.cold_zones = cold[:5]
        
        # ৪. বেস প্রোবেবিলিটি
        for outcome in set(outcomes):
            new_model.probabilities[outcome] = outcomes.count(outcome) / len(outcomes)
        
        # ৫. অ্যাকুরেসি ক্যালকুলেশন
        if len(outcomes) > 100:
            correct = 0
            total = 0
            
            for i in range(50, len(outcomes) - 10, 10):
                test_outcomes = outcomes[:i]
                if test_outcomes:
                    pred = max(set(test_outcomes[-20:]), key=test_outcomes[-20:].count)
                    if pred == outcomes[i]:
                        correct += 1
                    total += 1
            
            new_model.accuracy = correct / total if total > 0 else 0.5
        
        new_model.total_spins = len(outcomes)
        
        # মডেল সেভ
        global ai_model
        ai_model = new_model
        ai_model.save(MODEL_FILE)
        
        logger.info(f"✅ AI মডেল আপডেট: {len(outcomes)} স্পিন, অ্যাকুরেসি: {ai_model.accuracy:.2%}")
        
    except Exception as e:
        logger.error(f"❌ AI ট্রেনিংয়ে ত্রুটি: {e}")
        import traceback
        traceback.print_exc()

def quick_update_ai(new_data):
    """দ্রুত আপডেট (শুধু নতুন ডাটা দিয়ে)"""
    global ai_model
    
    try:
        # এক্সট্রাক্ট ডাটা
        result_data = new_data.get('data', {}).get('result', {}).get('outcome', {})
        wheel = result_data.get('wheelResult', {})
        
        if wheel.get('type') == 'Number':
            outcome = str(wheel.get('wheelSector', '0'))
        else:
            outcome = wheel.get('wheelSector', 'Unknown')
        
        # হট/কোল্ড আপডেট
        with tracker.lock:
            if ai_model.hot_zones:
                for zone in ai_model.hot_zones:
                    if zone.get('outcome') == outcome:
                        zone['actual'] = zone.get('actual', 0) + 1
            
            ai_model.total_spins += 1
        
    except Exception as e:
        logger.debug(f"দ্রুত আপডেটে ত্রুটি: {e}")

# ==================== প্রেডিকশন জেনারেটর ====================
def get_ensemble_prediction():
    """AI মডেল থেকে এনসেম্বল প্রেডিকশন"""
    with tracker.lock:
        if len(tracker.data) < 20:
            return {
                'primary': 'অপর্যাপ্ত ডাটা',
                'confidence': 0,
                'alternatives': [],
                'probabilities': {},
                'factors': {}
            }
        
        # সর্বশেষ ডাটা
        last_20 = tracker.data[-20:]
    
    outcomes = []
    for spin in last_20:
        result_data = spin.get('data', {}).get('result', {}).get('outcome', {})
        wheel = result_data.get('wheelResult', {})
        
        if wheel.get('type') == 'Number':
            outcomes.append(str(wheel.get('wheelSector', '0')))
        else:
            outcomes.append(wheel.get('wheelSector', 'Unknown'))
    
    current_outcome = outcomes[-1] if outcomes else None
    
    # এনসেম্বল প্রেডিকশন
    predictions = {}
    
    # ১. মার্কোভ
    if ai_model.markov_chain and current_outcome:
        level1 = ai_model.markov_chain.get('level_1', {}).get(current_outcome, {})
        if level1:
            for out, prob in level1.items():
                predictions[out] = predictions.get(out, 0) + prob * ai_model.ensemble_weights['markov']
    
    # ২. হট জোন
    for zone in ai_model.hot_zones:
        if zone.get('deviation', 0) > 0.5:
            predictions[zone['outcome']] = predictions.get(zone['outcome'], 0) + 0.1 * ai_model.ensemble_weights['hot']
    
    # বেস প্রোবেবিলিটি যোগ করুন (যদি কোনো প্রেডিকশন না থাকে)
    if not predictions and ai_model.probabilities:
        predictions = ai_model.probabilities.copy()
    
    # নরমালাইজ
    total = sum(predictions.values())
    if total > 0:
        for out in predictions:
            predictions[out] /= total
    
    # টপ প্রেডিকশন
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_preds:
        primary = sorted_preds[0][0]
        confidence = sorted_preds[0][1]
        
        # অ্যাকুরেসি ফ্যাক্টর
        confidence = confidence * (0.7 + 0.3 * ai_model.accuracy)
        
        return {
            'primary': primary,
            'confidence': min(confidence, 0.95),
            'alternatives': [p[0] for p in sorted_preds[1:4]],
            'probabilities': predictions,
            'factors': {
                'markov': len(level1) if 'level1' in locals() else 0,
                'hot': len(ai_model.hot_zones)
            },
            'accuracy': ai_model.accuracy
        }
    
    return {
        'primary': 'অনির্ধারিত',
        'confidence': 0,
        'alternatives': [],
        'probabilities': {},
        'factors': {},
        'accuracy': ai_model.accuracy
    }

# ==================== API এন্ডপয়েন্ট ====================
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def serve_predictions():
    return send_from_directory('.', 'predictions.html')

@app.route('/api/crazytime')
def get_crazytime_data():
    """ডাটা রিটার্ন করে"""
    try:
        page = int(request.args.get('page', 0))
        size = int(request.args.get('size', 100))
        
        with tracker.lock:
            start = page * size
            end = start + size
            data = tracker.data[start:end] if tracker.data else []
            total = tracker.total_count
            
            response = jsonify(data)
            response.headers['X-Total-Count'] = str(total)
            return response
    except Exception as e:
        logger.error(f"API ত্রুটি: {e}")
        return jsonify([]), 200

@app.route('/api/crazytime/latest')
def get_latest():
    """সর্বশেষ ডাটা রিটার্ন করে"""
    with tracker.lock:
        if tracker.data and len(tracker.data) > 0:
            return jsonify(tracker.data[-1])
        return jsonify({})

@app.route('/api/predictions')
def get_predictions():
    """AI প্রেডিকশন রিটার্ন করে"""
    prediction = get_ensemble_prediction()
    
    return jsonify({
        'prediction': prediction,
        'model': {
            'total_spins': ai_model.total_spins,
            'accuracy': ai_model.accuracy,
            'hot_zones': ai_model.hot_zones,
            'cold_zones': ai_model.cold_zones
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    """হেলথ চেক"""
    with tracker.lock:
        return jsonify({
            'status': 'ok',
            'data_count': tracker.total_count,
            'model_accuracy': ai_model.accuracy,
            'last_update': tracker.last_update.isoformat() if tracker.last_update else None,
            'uptime': '24/7',
            'scheduler_running': len(scheduler.get_jobs()) > 0
        })

@app.route('/api/stats')
def stats():
    """স্ট্যাটিসটিক্স"""
    with tracker.lock:
        return jsonify({
            'total_spins': tracker.total_count,
            'model_accuracy': ai_model.accuracy,
            'hot_zones': ai_model.hot_zones,
            'cold_zones': ai_model.cold_zones,
            'last_update': tracker.last_update.isoformat() if tracker.last_update else None
        })

@app.route('/api/force-update', methods=['POST'])
def force_update():
    """ফোর্স আপডেট"""
    try:
        threading.Thread(target=check_new_data).start()
        return jsonify({'status': 'success', 'message': 'আপডেট শুরু হয়েছে'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ==================== সার্ভার স্টার্টআপ ====================
def start_background_tasks():
    """ব্যাকগ্রাউন্ড টাস্ক শুরু করে"""
    
    # ডাটা লোড
    load_data()
    
    # ৭২ ঘন্টার ডাটা কালেকশন শুরু (যদি না থাকে)
    if tracker.total_count < 100:
        logger.info("🔄 ৭২ ঘন্টার ডাটা সংগ্রহ শুরু হচ্ছে...")
        thread = threading.Thread(target=fetch_72h_data)
        thread.daemon = True
        thread.start()
    else:
        logger.info(f"📦 {tracker.total_count} টি ডাটা ইতিমধ্যে আছে")
        
        # AI মডেল ট্রেন (যদি প্রয়োজন হয়)
        if ai_model.total_spins < 100 and tracker.total_count >= 100:
            logger.info("🔄 AI মডেল ট্রেনিং শুরু হচ্ছে...")
            thread = threading.Thread(target=train_ai_model)
            thread.daemon = True
            thread.start()
    
    # প্রতি ৫ সেকেন্ডে নতুন ডাটা চেক করার জন্য scheduler
    scheduler.add_job(
        func=check_new_data,
        trigger=IntervalTrigger(seconds=5),
        id='check_new_data',
        replace_existing=True
    )
    
    # প্রতি ১০ মিনিটে ডাটা সেভ
    scheduler.add_job(
        func=save_data,
        trigger=IntervalTrigger(seconds=600),
        id='save_data',
        replace_existing=True
    )
    
    # প্রতি ঘন্টায় AI মডেল ট্রেন
    scheduler.add_job(
        func=train_ai_model,
        trigger=IntervalTrigger(seconds=3600),
        id='train_ai_model',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("✅ ব্যাকগ্রাউন্ড টাস্ক শুরু হয়েছে")
    logger.info("⏱️ প্রতি ৫ সেকেন্ডে নতুন ডাটা চেক")

if __name__ == '__main__':
    print("🔥" * 40)
    print("🔥   ২৪/৭ ক্রেজি টাইম AI ট্র্যাকার   🔥")
    print("🔥" * 40)
    
    # ব্যাকগ্রাউন্ড টাস্ক শুরু
    start_background_tasks()
    
    # শাটডাউনে ক্লিনআপ
    atexit.register(lambda: scheduler.shutdown())
    
    print(f"📊 বর্তমান ডাটা: {tracker.total_count} টি")
    print(f"🤖 মডেল অ্যাকুরেসি: {ai_model.accuracy:.2%}")
    print(f"⏱️  প্রতি ৫ সেকেন্ডে নতুন ডাটা চেক")
    print(f"🌐 পোর্ট: {os.environ.get('PORT', 5000)}")
    print("=" * 40)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
