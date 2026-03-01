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

def start_initial_collection():
    """আলাদা থ্রেডে ডাটা কালেকশন শুরু"""
    time.sleep(5)  # healthcheck এর জন্য ৫ সেকেন্ড wait
    with app.app_context():
        logger.info("🚀 ডাটা কালেকশন শুরু হচ্ছে...")
        data = data_collector.fetch_all_data()
        if data and len(data) >= Config.MIN_DATA_FOR_TRAINING:
            ai_model.train(data)

@app.before_request
def before_request():
    """প্রথম রিকোয়েস্টে ডাটা কালেকশন শুরু"""
    global initial_data_collected
    if not initial_data_collected:
        initial_data_collected = True
        # আলাদা থ্রেডে ডাটা কালেকশন শুরু করুন
        thread = threading.Thread(target=start_initial_collection, daemon=True)
        thread.start()

# ==================== রুটস ====================

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
                'predictions': [['1', 0.2], ['2', 0.15], ['5', 0.15], ['10', 0.1],
                               ['CF', 0.15], ['CH', 0.1], ['CB', 0.08], ['PC', 0.07]],
                'model_status': {
                    'accuracy': 0.125,
                    'data_points': len(data),
                    'learning_phase': 'বেসিক',
                    'bonus_stats': {}
                }
            })
        
        predictions = ai_model.get_ensemble_prediction(data)
        
        response = {
            'predictions': predictions[:8],
            'model_status': {
                'accuracy': ai_model.accuracy_history[-1] if ai_model.accuracy_history else 0.125,
                'trained': ai_model.last_trained.isoformat() if ai_model.last_trained else None,
                'data_points': len(data),
                'learning_phase': 'অ্যাডভান্সড' if len(data) > 200 else 'বেসিক',
                'bonus_stats': ai_model.get_model_stats().get('bonus_counts', {})
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ প্রেডিকশন API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """স্ট্যাটাস API - বোনাস ইনফো সহ"""
    try:
        data = data_collector.get_all_data()
        stats = data_collector.get_statistics() if hasattr(data_collector, 'get_statistics') else {}
        
        # বোনাস কাউন্ট বের করা
        bonus_counts = {'CF': 0, 'CH': 0, 'CB': 0, 'PC': 0}
        for item in data[-500:]:  # শেষ ৫০০ টা
            try:
                outcome = item.get('data', {}).get('result', {}).get('outcome', {})
                wheel = outcome.get('wheelResult', {})
                if wheel.get('type') != 'Number':
                    raw = wheel.get('wheelSector', '')
                    if raw == 'CoinFlip': bonus_counts['CF'] += 1
                    elif raw == 'CashHunt': bonus_counts['CH'] += 1
                    elif raw == 'CrazyBonus': bonus_counts['CB'] += 1
                    elif raw == 'Pachinko': bonus_counts['PC'] += 1
            except:
                pass
        
        return jsonify({
            'status': 'running',
            'data_count': len(data),
            'background_thread': background_thread_running,
            'initial_data_collected': initial_data_collected,
            'statistics': stats,
            'bonus_counts': bonus_counts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """হেলথ চেক - সবসময় 200 return"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

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
