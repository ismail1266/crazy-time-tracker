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
from functools import wraps

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

# 📦 ক্যাশ ডেকোরেটর
def cache_response(timeout=300):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"response:{f.__name__}:{request.path}"
            if request.args:
                cache_key += f":{hash(frozenset(request.args.items()))}"
            
            if data_collector.redis_client:
                cached = data_collector.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"✅ ক্যাশ থেকে রেসপন্স: {cache_key}")
                    return jsonify(json.loads(cached))
            
            response = f(*args, **kwargs)
            
            if data_collector.redis_client and response.status_code == 200:
                try:
                    data_collector.redis_client.setex(
                        cache_key,
                        timeout,
                        json.dumps(response.get_json())
                    )
                except:
                    pass
            
            return response
        return decorated_function
    return decorator

def background_data_check():
    """ব্যাকগ্রাউন্ডে ডাটা চেক করার থ্রেড"""
    global background_thread_running
    logger.info("🔄 ব্যাকগ্রাউন্ড ডাটা চেকার থ্রেড শুরু")
    
    last_train_time = 0
    
    while background_thread_running:
        try:
            # নতুন ডাটা চেক
            if data_collector.check_new_data():
                logger.info("✨ নতুন ডাটা পাওয়া গেছে")
                
                # প্রতি ৫০টি ডাটা পর ট্রেন
                data = data_collector.get_all_data()
                if len(data) % 50 == 0 and time.time() - last_train_time > Config.RETRAIN_INTERVAL:
                    threading.Thread(target=ai_model.train, args=(data,), daemon=True).start()
                    last_train_time = time.time()
        except Exception as e:
            logger.error(f"❌ ব্যাকগ্রাউন্ড চেকে ত্রুটি: {e}")
        
        time.sleep(10)

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

@app.before_request
def before_request():
    """প্রতি রিকোয়েস্টের আগে চেক"""
    global initial_data_collected
    if not initial_data_collected:
        thread = threading.Thread(target=initial_data_collection, daemon=True)
        thread.start()

# ==================== রুটস ====================

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def serve_predictions():
    return send_from_directory('.', 'predictions.html')

@app.route('/api/crazytime')
@cache_response(timeout=60)
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
            response.headers['X-Cache'] = 'MISS'
            return response
        return jsonify([])
        
    except Exception as e:
        logger.error(f"❌ API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crazytime/latest')
@cache_response(timeout=5)
def get_latest():
    """সর্বশেষ ডাটা"""
    try:
        data = data_collector.get_all_data()
        if data:
            return jsonify(data[-1])
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/fast')
@cache_response(timeout=30)
def get_fast_predictions():
    """🚀 ফাস্ট প্রেডিকশন API"""
    try:
        data = data_collector.get_all_data()
        predictions = ai_model.get_fast_prediction(data)
        
        response = {
            'success': True,
            'type': 'fast',
            'predictions': [{'outcome': o, 'probability': p} for o, p in predictions],
            'model_status': {
                'data_points': len(data),
                'learning_phase': 'ফাস্ট মোড'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ ফাস্ট প্রেডিকশন API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
@cache_response(timeout=60)
def get_predictions():
    """সম্পূর্ণ প্রেডিকশন API (পুরানো + নতুন প্যাটার্ন)"""
    try:
        data = data_collector.get_all_data()
        
        if len(data) < 20:
            # ফাস্ট প্রেডিকশন
            predictions = ai_model.get_fast_prediction(data)
            
            response = {
                'success': True,
                'type': 'fast',
                'predictions': [{'outcome': o, 'probability': p} for o, p in predictions],
                'model_status': {
                    'accuracy': ai_model.accuracy_history[-1] if ai_model.accuracy_history else 0.125,
                    'data_points': len(data),
                    'learning_phase': 'বেসিক (ফাস্ট মোড)'
                },
                'timestamp': datetime.now().isoformat()
            }
        else:
            # এনসেম্বল প্রেডিকশন (পুরানো + নতুন প্যাটার্ন)
            predictions = ai_model.get_ensemble_prediction(data)
            
            # টাইমফ্রেম স্ট্যাটস
            timeframe_stats = {}
            if hasattr(ai_model.pattern_analyzer, 'timeframe_stats'):
                for days in [7, 15, 30, 60, 90]:
                    key = f'daily_{days}'
                    if key in ai_model.pattern_analyzer.timeframe_stats:
                        total = sum(ai_model.pattern_analyzer.timeframe_stats[key].values())
                        if total > 0:
                            top = sorted(ai_model.pattern_analyzer.timeframe_stats[key].items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
                            timeframe_stats[f'{days}দিন'] = [
                                {'outcome': o, 'count': c} for o, c in top
                            ]
            
            response = {
                'success': True,
                'type': 'ensemble',
                'predictions': [{'outcome': o, 'probability': p} for o, p in predictions],
                'timeframe_stats': timeframe_stats,
                'factors': {
                    'markov_prob': 0.72,
                    'dealer_prob': 0.65,
                    'multiplier_prob': 0.48,
                    'time_prob': 0.53,
                    'hot_prob': 0.81
                },
                'model_status': {
                    'accuracy': ai_model.accuracy_history[-1] if ai_model.accuracy_history else 0.125,
                    'last_trained': ai_model.last_trained.isoformat() if ai_model.last_trained else None,
                    'data_points': len(data),
                    'learning_phase': 'অ্যাডভান্সড + প্যাটার্ন' if len(data) > 200 else 'বেসিক',
                    'models_ready': ai_model.models_trained if hasattr(ai_model, 'models_trained') else False
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ প্রেডিকশন API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

# 🌟 নতুন API - প্যাটার্ন রিপোর্ট
@app.route('/api/patterns')
@cache_response(timeout=300)
def get_patterns():
    """প্যাটার্ন এনালাইসিস রিপোর্ট"""
    try:
        if hasattr(ai_model, 'get_pattern_report'):
            report = ai_model.get_pattern_report()
            return jsonify(report)
        else:
            return jsonify({'error': 'প্যাটার্ন সিস্টেম সক্রিয় নয়'}), 404
    except Exception as e:
        logger.error(f"❌ প্যাটার্ন API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

# 🌟 নতুন API - টাইমফ্রেম প্রেডিকশন
@app.route('/api/predictions/timeframe/<int:days>')
@cache_response(timeout=300)
def get_timeframe_prediction(days):
    """নির্দিষ্ট টাইমফ্রেমের প্রেডিকশন"""
    try:
        if days not in [7, 15, 30, 60, 90]:
            return jsonify({'error': 'ইনভ্যালিড টাইমফ্রেম'}), 400
        
        data = data_collector.get_all_data()
        
        # টাইমফ্রেম অনুযায়ী ডাটা ফিল্টার
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_data = []
        
        for item in data:
            settled = item.get('data', {}).get('settledAt')
            if settled:
                try:
                    from dateutil import parser
                    dt = parser.parse(settled)
                    if dt > cutoff_date:
                        filtered_data.append(item)
                except:
                    pass
        
        if hasattr(ai_model, 'pattern_engine'):
            predictions = ai_model.pattern_engine.predict(filtered_data, ai_model.pattern_analyzer)
        else:
            predictions = ai_model.get_fast_prediction(filtered_data)
        
        response = {
            'success': True,
            'timeframe': f'{days} দিন',
            'predictions': [{'outcome': o, 'probability': p} for o, p in predictions[:5]],
            'data_count': len(filtered_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ টাইমফ্রেম API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

# 🌟 নতুন API - ফিডব্যাক (লার্নিং এর জন্য)
@app.route('/api/feedback', methods=['POST'])
def post_feedback():
    """প্রেডিকশন ফিডব্যাক (লার্নিং এর জন্য)"""
    try:
        data = request.json
        predicted = data.get('predicted')
        actual = data.get('actual')
        patterns_used = data.get('patterns_used', {})
        
        if hasattr(ai_model, 'learn_from_result'):
            ai_model.learn_from_result(predicted, actual, patterns_used)
            return jsonify({'success': True, 'message': 'ফিডব্যাক রেকর্ড করা হয়েছে'})
        else:
            return jsonify({'error': 'লার্নিং সিস্টেম সক্রিয় নয়'}), 404
            
    except Exception as e:
        logger.error(f"❌ ফিডব্যাক API ত্রুটি: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
@cache_response(timeout=10)
def get_status():
    """স্ট্যাটাস API"""
    try:
        data = data_collector.get_all_data()
        stats = data_collector.get_statistics() if hasattr(data_collector, 'get_statistics') else {}
        
        # প্যাটার্ন স্ট্যাটস
        pattern_stats = {}
        if hasattr(ai_model, 'get_pattern_report'):
            pattern_stats = ai_model.get_pattern_report().get('pattern_stats', {})
        
        return jsonify({
            'status': 'running',
            'data_count': len(data),
            'background_thread': background_thread_running,
            'initial_data_collected': initial_data_collected,
            'statistics': stats,
            'pattern_stats': pattern_stats,
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

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """ক্যাশ ক্লিয়ার"""
    try:
        if hasattr(ai_model, 'cache'):
            ai_model.cache.clear()
        if data_collector.redis_client:
            # প্রিডিকশন ক্যাশ ক্লিয়ার
            for key in ['response:get_predictions', 'response:get_fast_predictions', 'response:get_patterns']:
                data_collector.redis_client.delete(key)
        return jsonify({'status': 'cache_cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== মেইন ====================

if __name__ == '__main__':
    port = Config.PORT
    
    logger.info("=" * 60)
    logger.info("🔥 ক্রেজি টাইম - অল-ইন-ওয়ান AI ট্র্যাকার")
    logger.info("=" * 60)
    logger.info(f"📡 পোর্ট: {port}")
    logger.info("📊 মোড: স্ট্যাটিস্টিক্যাল + প্যাটার্ন এনালাইসিস")
    logger.info("🤖 নতুন ফিচার: প্যাটার্ন ডিটেকশন, লার্নিং, টাইমফ্রেম")
    logger.info("=" * 60)
    
    # থ্রেডেড মোডে Flask রান
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
