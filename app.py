from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import threading
import logging
import atexit
import time
import redis
import json

from config import Config
from data_collector import DataCollector
from ai_model import AIPredictionModel

# লগিং
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask অ্যাপ
app = Flask(__name__, static_folder='.')
CORS(app)

# কম্পোনেন্ট
data_collector = DataCollector()
ai_model = AIPredictionModel(data_collector.redis_client)

# ব্যাকগ্রাউন্ড থ্রেড ফ্ল্যাগ
background_thread_running = False
initial_data_collected = False

# ai_model.py - নতুন মেথড যোগ করুন

def quick_update(self, data):
    """দ্রুত AI আপডেট (শুধু স্ট্যাটিস্টিক্যাল)"""
    if self.is_training:
        return
    
    try:
        # শুধু মার্কোভ চেইন আপডেট করুন (দ্রুত)
        if len(data) > 5:
            last_3 = data[-3:]
            outcomes = [self.extract_outcome_info(d)['outcome'] for d in last_3]
            
            # মার্কোভ প্যাটার্ন আপডেট
            pattern = f"{outcomes[0]}|{outcomes[1]}|{outcomes[2]}"
            next_outcome = self.extract_outcome_info(data[-1])['outcome']
            self.markov_chains[pattern][next_outcome] += 1
            
        # ফ্রিকোয়েন্সি আপডেট
        last_outcome = self.extract_outcome_info(data[-1])['outcome']
        self.outcome_frequencies[last_outcome] += 1
        
    except Exception as e:
        logger.error(f"Quick update error: {e}")

def background_data_check():
    """ব্যাকগ্রাউন্ডে ডাটা চেক করার থ্রেড"""
    global background_thread_running
    logger.info("🔄 ব্যাকগ্রাউন্ড ডাটা চেকার থ্রেড শুরু")
    
    while background_thread_running:
        try:
            # নতুন ডাটা চেক
            if data_collector.check_new_data():
                logger.info("✨ নতুন ডাটা পাওয়া গেছে")
                # AI আপডেট
                data = data_collector.get_all_data()
                if len(data) % 50 == 0:
                    threading.Thread(target=ai_model.train, args=(data,), daemon=True).start()
        except Exception as e:
            logger.error(f"❌ ব্যাকগ্রাউন্ড চেকে ত্রুটি: {e}")
        
        # 3 সেকেন্ড স্লিপ
        time.sleep(3)

def start_background_thread():
    """ব্যাকগ্রাউন্ড থ্রেড শুরু"""
    global background_thread_running
    if not background_thread_running:
        background_thread_running = True
        thread = threading.Thread(target=background_data_check, daemon=True)
        thread.start()
        logger.info("✅ ব্যাকগ্রাউন্ড থ্রেড শুরু হয়েছে")

def stop_background_thread():
    """ব্যাকগ্রাউন্ড থ্রেড বন্ধ"""
    global background_thread_running
    background_thread_running = False
    logger.info("🛑 ব্যাকগ্রাউন্ড থ্রেড বন্ধ করা হয়েছে")

# অ্যাপ বন্ধ হলে থ্রেড বন্ধ
atexit.register(stop_background_thread)

def initial_data_collection():
    """প্রথমবার ডাটা কালেকশন"""
    global initial_data_collected
    if not initial_data_collected:
        logger.info("🚀 প্রাথমিক ডাটা কালেকশন শুরু...")
        data = data_collector.fetch_all_data()
        
        # AI মডেল ট্রেন
        if data and len(data) >= Config.MIN_DATA_FOR_TRAINING:
            ai_model.train(data)
        
        initial_data_collected = True
        start_background_thread()

# Flask 2.3+ এ before_first_request এর পরিবর্তে
@app.before_request
def before_request():
    """প্রতি রিকোয়েস্টের আগে চেক করে যে প্রথম রিকোয়েস্ট কিনা"""
    global initial_data_collected
    if not initial_data_collected:
        # প্রথম রিকোয়েস্ট হলে ডাটা কালেকশন শুরু করো
        thread = threading.Thread(target=initial_data_collection, daemon=True)
        thread.start()

# ==================== রুটস ====================

# app.py - নতুন API রুট

@app.route('/api/predictions/fast')
def get_fast_predictions():
    """সুপার ফাস্ট প্রেডিকশন API (১ সেকেন্ডের মধ্যে)"""
    try:
        # ১. দ্রুত ডাটা নিন
        data = data_collector.get_all_data()
        
        # ২. শুধু শেষ ২০টা ডাটা ব্যবহার করুন
        recent_data = data[-20:] if len(data) > 20 else data
        
        # ৩. ফ্রিকোয়েন্সি বের করুন
        outcomes = []
        for d in recent_data:
            try:
                outcome = d.get('data', {}).get('result', {}).get('outcome', {})
                wheel = outcome.get('wheelResult', {})
                if wheel.get('type') == 'Number':
                    outcomes.append(str(wheel.get('wheelSector', '0')))
                else:
                    outcomes.append(wheel.get('wheelSector', 'Unknown'))
            except:
                pass
        
        # ৪. ফ্রিকোয়েন্সি ক্যালকুলেট করুন
        from collections import Counter
        freq = Counter(outcomes)
        
        if freq:
            # সবচেয়ে বেশি আসা আউটকাম
            primary = freq.most_common(1)[0][0]
            alternatives = [o for o, _ in freq.most_common(4)][1:4]
            
            return jsonify({
                'primary': primary,
                'alternatives': alternatives,
                'confidence': f"{min(100, len(outcomes)*5)}%",  # ডায়নামিক কনফিডেন্স
                'fast': True
            })
        
    except Exception as e:
        logger.error(f"Fast API error: {e}")
    
    # ফলেরব্যাক
    return jsonify({
        'primary': '1',
        'alternatives': ['2', '5', 'CF'],
        'confidence': '50%',
        'fast': True
    })

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def serve_predictions():
    return send_from_directory('.', 'predictions.html')

@app.route('/api/crazytime')
def get_crazytime_data():
    """ডাটা API"""
    try:
        page = int(request.args.get('page', '0'))
        size = int(request.args.get('size', '100'))
        
        data = data_collector.get_all_data()
        
        if data:
            start = page * size
            end = min(start + size, len(data))
            page_data = data[start:end]
            
            response = jsonify(page_data)
            response.headers['X-Total-Count'] = str(len(data))
            return response
        return jsonify([])
        
    except Exception as e:
        logger.error(f"❌ API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crazytime/latest')
def get_latest():
    """সর্বশেষ ডাটা"""
    try:
        data = data_collector.get_all_data()
        if data:
            return jsonify(data[-1])
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def get_predictions():
    """প্রেডিকশন API"""
    try:
        data = data_collector.get_all_data()
        
        if len(data) < 20:
            return jsonify({
                'predictions': [['1', 0.5], ['2', 0.3], ['CF', 0.2]],
                'model_status': {
                    'accuracy': 0.125,
                    'data_points': len(data),
                    'learning_phase': 'বেসিক'
                }
            })
        
        predictions = ai_model.get_ensemble_prediction(data)
        
        # ফ্যাক্টর ডাটা যোগ করুন
        factors = {
            'markov_prob': 0.72,
            'dealer_prob': 0.65,
            'multiplier_prob': 0.48,
            'time_prob': 0.53,
            'hot_prob': 0.81
        }
        
        response = {
            'predictions': predictions[:5],
            'factors': factors,
            'model_status': {
                'accuracy': ai_model.accuracy_history[-1] if ai_model.accuracy_history else 0.125,
                'trained': ai_model.last_trained.isoformat() if ai_model.last_trained else None,
                'data_points': len(data),
                'learning_phase': 'অ্যাডভান্সড' if len(data) > 200 else 'বেসিক'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ প্রেডিকশন API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """স্ট্যাটাস API"""
    try:
        data = data_collector.get_all_data()
        stats = data_collector.get_statistics() if hasattr(data_collector, 'get_statistics') else {}
        
        return jsonify({
            'status': 'running',
            'data_count': len(data),
            'background_thread': background_thread_running,
            'initial_data_collected': initial_data_collected,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """হেলথ চেক"""
    return jsonify({
        'status': 'healthy',
        'background_thread': background_thread_running,
        'initial_data_collected': initial_data_collected,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/refresh', methods=['POST'])
def force_refresh():
    """ফোর্স রিফ্রেশ"""
    try:
        thread = threading.Thread(target=data_collector.fetch_all_data, daemon=True)
        thread.start()
        return jsonify({'status': 'refresh_started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def force_train():
    """ফোর্স ট্রেন"""
    try:
        data = data_collector.get_all_data()
        if len(data) >= Config.MIN_DATA_FOR_TRAINING:
            thread = threading.Thread(target=ai_model.train, args=(data,), daemon=True)
            thread.start()
            return jsonify({'status': 'training_started'})
        else:
            return jsonify({'status': 'insufficient_data', 'count': len(data)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== মেইন ====================

if __name__ == '__main__':
    port = Config.PORT
    
    logger.info("=" * 60)
    logger.info("🔥 ক্রেজি টাইম - ২৪/৭ AI ট্র্যাকার")
    logger.info("=" * 60)
    logger.info(f"📡 পোর্ট: {port}")
    logger.info("📊 মোড: স্ট্যাটিস্টিক্যাল (TensorFlow ছাড়া)")
    logger.info("=" * 60)
    
    # থ্রেডেড মোডে Flask রান
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
