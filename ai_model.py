import numpy as np
from datetime import datetime, timedelta
import json
import logging
import threading
from collections import Counter, defaultdict
from dateutil import parser
import pickle
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import Config

# সেটআপ লগিং
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow চেক
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow উপলব্ধ")
except ImportError:
    TF_AVAILABLE = False
    logger.info("📊 TensorFlow ছাড়া স্ট্যাটিস্টিক্যাল মোডে চলছে")

class AIPredictionModel:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )
        
        self.scaler_fitted = False
        self.rf_trained = False
        self.training_data = []
        self.accuracy_history = []
        self.last_trained = None
        self.is_training = False
        self.tf_available = TF_AVAILABLE
        
        # মডেল প্যারামিটার
        self.seq_length = 10
        self.n_features = 20
        
        # স্ট্যাটিস্টিক্যাল মডেল
        self.markov_chains = defaultdict(lambda: defaultdict(int))
        self.outcome_frequencies = Counter()
        self.dealer_patterns = defaultdict(lambda: defaultdict(int))
        self.time_patterns = defaultdict(lambda: defaultdict(int))
        self.multiplier_history = []
        
        # ক্যাশ ডিরেক্টরি
        os.makedirs('models', exist_ok=True)
        
        # মডেল লোড
        self.load_model()
        
        logger.info("✅ AI মডেল ইনিশিয়ালাইজ করা হয়েছে")
    
    def parse_datetime(self, time_str):
        """নিরাপদ datetime পার্সিং"""
        if not time_str:
            return None
        try:
            return parser.parse(time_str)
        except:
            try:
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            except:
                return None
    
    def extract_outcome_info(self, result):
        """রেজাল্ট থেকে আউটকাম ইনফো এক্সট্র্যাক্ট"""
        try:
            data = result.get('data', {})
            outcome = data.get('result', {}).get('outcome', {})
            wheel = outcome.get('wheelResult', {})
            
            # আউটকাম টাইপ
            if wheel.get('type') == 'Number':
                outcome_val = str(wheel.get('wheelSector', '0'))
                outcome_category = 'number'
            else:
                outcome_val = wheel.get('wheelSector', 'Unknown')
                outcome_category = 'bonus'
            
            # মাল্টিপ্লায়ার
            multiplier = float(outcome.get('maxMultiplier', 1))
            
            # ডিলার
            dealer = data.get('dealer', {}).get('name', 'Unknown')
            
            # সময়
            settled = data.get('settledAt')
            dt = self.parse_datetime(settled)
            
            return {
                'outcome': outcome_val,
                'category': outcome_category,
                'multiplier': multiplier,
                'dealer': dealer,
                'timestamp': dt,
                'hour': dt.hour if dt else 0,
                'minute': dt.minute if dt else 0,
                'weekday': dt.weekday() if dt else 0
            }
        except Exception as e:
            logger.error(f"আউটকাম এক্সট্র্যাক্ট ত্রুটি: {e}")
            return {
                'outcome': 'Unknown',
                'category': 'unknown',
                'multiplier': 1,
                'dealer': 'Unknown',
                'timestamp': None,
                'hour': 0,
                'minute': 0,
                'weekday': 0
            }
    
    def update_statistical_models(self, outcomes_info):
        """স্ট্যাটিস্টিক্যাল মডেল আপডেট"""
        if len(outcomes_info) < 10:
            return
        
        # ফ্রিকোয়েন্সি আপডেট
        for info in outcomes_info:
            self.outcome_frequencies[info['outcome']] += 1
        
        # মার্কোভ চেইন আপডেট (3-gram)
        outcomes = [info['outcome'] for info in outcomes_info]
        for i in range(len(outcomes) - 3):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}"
            next_outcome = outcomes[i+3]
            self.markov_chains[pattern][next_outcome] += 1
        
        # ডিলার প্যাটার্ন আপডেট
        for i in range(len(outcomes_info) - 1):
            dealer = outcomes_info[i]['dealer']
            current = outcomes_info[i]['outcome']
            next_outcome = outcomes_info[i+1]['outcome']
            self.dealer_patterns[f"{dealer}|{current}"][next_outcome] += 1
        
        # টাইম প্যাটার্ন আপডেট
        for info in outcomes_info:
            hour_group = info['hour'] // 4  # ৪ ঘন্টার গ্রুপ
            self.time_patterns[f"hour_{hour_group}"][info['outcome']] += 1
            
            if info['multiplier'] > 10:
                self.time_patterns["high_multiplier"][info['outcome']] += 1
    
    def prepare_features(self, data):
        """ফিচার তৈরি (স্কেলেবল)"""
        features = []
        outcomes = []
        
        # সীমিত ডাটা ব্যবহার
        sample_size = min(len(data), 2000)
        data = data[-sample_size:]
        
        for i, result in enumerate(data):
            info = self.extract_outcome_info(result)
            outcomes.append(info['outcome'])
            
            # বেসিক ফিচার
            feature = [
                info['hour'] / 24.0,  # সময়
                info['weekday'] / 7.0,  # সপ্তাহের দিন
                info['multiplier'] / 50.0,  # মাল্টিপ্লায়ার (নরমালাইজড)
            ]
            
            # শেষ ৫টি আউটকামের হিস্ট্রি
            for j in range(1, 6):
                if i >= j:
                    prev_info = self.extract_outcome_info(data[i-j])
                    # আউটকাম এনকোডিং
                    outcome_code = hash(prev_info['outcome']) % 100 / 100.0
                    feature.append(outcome_code)
                    feature.append(prev_info['multiplier'] / 50.0)
                else:
                    feature.extend([0, 0])
            
            # প্যাডিং
            while len(feature) < self.n_features:
                feature.append(0)
            
            features.append(feature[:self.n_features])
        
        return np.array(features, dtype=np.float32), outcomes
    
    def train(self, data):
        """মডেল ট্রেন"""
        if self.is_training or len(data) < Config.MIN_DATA_FOR_TRAINING:
            return
        
        self.is_training = True
        logger.info(f"🚀 মডেল ট্রেনিং শুরু... ({len(data)} টি ডাটা)")
        
        try:
            # আউটকাম ইনফো এক্সট্র্যাক্ট
            outcomes_info = [self.extract_outcome_info(r) for r in data]
            
            # স্ট্যাটিস্টিক্যাল মডেল আপডেট
            self.update_statistical_models(outcomes_info)
            
            # ফিচার প্রিপারেশন
            features, outcomes = self.prepare_features(data)
            
            if len(features) < 50:
                logger.warning("⚠️ পর্যাপ্ত ডাটা নেই ট্রেনিং এর জন্য")
                return
            
            # আউটকাম লিমিট (টপ ১২)
            top_outcomes = [o[0] for o in self.outcome_frequencies.most_common(12)]
            outcomes_clean = [o if o in top_outcomes else 'Other' for o in outcomes]
            
            # এনকোডিং
            self.label_encoder.fit(outcomes_clean)
            y = self.label_encoder.transform(outcomes_clean)
            
            # ট্রেন-টেস্ট স্প্লিট
            X_train, X_test, y_train, y_test = train_test_split(
                features, y, test_size=0.2, random_state=42
            )
            
            # স্কেলিং
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.scaler_fitted = True
            
            # র্যান্ডম ফরেস্ট ট্রেন
            self.rf_model.fit(X_train_scaled, y_train)
            self.rf_trained = True
            
            # অ্যাকুরেসি চেক
            y_pred = self.rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracy_history.append(float(accuracy))
            
            # মার্কোভ বেসড অ্যাকুরেসি
            markov_accuracy = self.calculate_markov_accuracy(outcomes_info)
            
            logger.info(f"✅ র্যান্ডম ফরেস্ট অ্যাকুরেসি: {accuracy:.2%}")
            logger.info(f"✅ মার্কোভ চেইন অ্যাকুরেসি: {markov_accuracy:.2%}")
            
            self.last_trained = datetime.now()
            
            # মডেল সেভ
            self.save_model()
            
        except Exception as e:
            logger.error(f"❌ ট্রেনিং ত্রুটি: {e}")
        finally:
            self.is_training = False
    
    def calculate_markov_accuracy(self, outcomes_info):
        """মার্কোভ চেইনের অ্যাকুরেসি ক্যালকুলেশন"""
        if len(outcomes_info) < 100:
            return 0.3
        
        outcomes = [info['outcome'] for info in outcomes_info]
        correct = 0
        total = 0
        
        for i in range(len(outcomes) - 4):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}"
            actual = outcomes[i+3]
            
            if pattern in self.markov_chains:
                predicted = max(self.markov_chains[pattern].items(), key=lambda x: x[1])[0]
                if predicted == actual:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.3
    
    def get_markov_prediction(self, last_3_outcomes):
        """মার্কোভ চেইন থেকে প্রেডিকশন"""
        pattern = f"{last_3_outcomes[0]}|{last_3_outcomes[1]}|{last_3_outcomes[2]}"
        
        if pattern in self.markov_chains:
            total = sum(self.markov_chains[pattern].values())
            predictions = []
            for outcome, count in self.markov_chains[pattern].items():
                predictions.append((outcome, count / total))
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        return []
    
    def get_dealer_prediction(self, dealer, current_outcome):
        """ডিলার প্যাটার্ন থেকে প্রেডিকশন"""
        pattern = f"{dealer}|{current_outcome}"
        
        if pattern in self.dealer_patterns:
            total = sum(self.dealer_patterns[pattern].values())
            predictions = []
            for outcome, count in self.dealer_patterns[pattern].items():
                predictions.append((outcome, count / total))
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        return []
    
    def get_time_prediction(self, hour):
        """সময় ভিত্তিক প্রেডিকশন"""
        hour_group = hour // 4
        pattern = f"hour_{hour_group}"
        
        if pattern in self.time_patterns:
            total = sum(self.time_patterns[pattern].values())
            predictions = []
            for outcome, count in self.time_patterns[pattern].items():
                predictions.append((outcome, count / total))
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        return []
    
    def get_ensemble_prediction(self, data, weights=None):
        """এনসেম্বল প্রেডিকশন (সব মডেল একসাথে)"""
        if not data or len(data) < 10:
            return [('1', 0.5), ('2', 0.3), ('CoinFlip', 0.2)]
        
        # ডিফল্ট ওয়েট
        if weights is None:
            weights = {
                'frequency': 0.25,
                'markov': 0.30,
                'dealer': 0.20,
                'time': 0.15,
                'rf': 0.10
            }
        
        # লাস্ট ডাটা
        last_5 = data[-5:]
        last_info = self.extract_outcome_info(last_5[-1])
        last_3_outcomes = [self.extract_outcome_info(data[-3+i])['outcome'] for i in range(3)]
        
        # ফ্রিকোয়েন্সি বেসড
        total = sum(self.outcome_frequencies.values())
        freq_pred = {}
        if total > 0:
            for outcome, count in self.outcome_frequencies.most_common(10):
                freq_pred[outcome] = count / total
        
        # মার্কোভ প্রেডিকশন
        markov_pred = {}
        for outcome, prob in self.get_markov_prediction(last_3_outcomes):
            markov_pred[outcome] = prob
        
        # ডিলার প্রেডিকশন
        dealer_pred = {}
        if last_info['dealer'] != 'Unknown':
            for outcome, prob in self.get_dealer_prediction(last_info['dealer'], last_info['outcome']):
                dealer_pred[outcome] = prob
        
        # টাইম প্রেডিকশন
        time_pred = {}
        for outcome, prob in self.get_time_prediction(last_info['hour']):
            time_pred[outcome] = prob
        
        # আরএফ প্রেডিকশন (যদি ট্রেন করা থাকে)
        rf_pred = {}
        if self.rf_trained and self.scaler_fitted:
            try:
                features, _ = self.prepare_features(last_5)
                if len(features) > 0:
                    features_scaled = self.scaler.transform([features[-1]])
                    proba = self.rf_model.predict_proba(features_scaled)[0]
                    for i, prob in enumerate(proba):
                        outcome = self.label_encoder.inverse_transform([i])[0]
                        rf_pred[outcome] = float(prob)
            except Exception as e:
                logger.error(f"RF প্রেডিকশন ত্রুটি: {e}")
        
        # এনসেম্বল
        all_outcomes = set()
        all_outcomes.update(freq_pred.keys())
        all_outcomes.update(markov_pred.keys())
        all_outcomes.update(dealer_pred.keys())
        all_outcomes.update(time_pred.keys())
        all_outcomes.update(rf_pred.keys())
        
        ensemble = []
        for outcome in all_outcomes:
            score = (
                weights['frequency'] * freq_pred.get(outcome, 0) +
                weights['markov'] * markov_pred.get(outcome, 0) +
                weights['dealer'] * dealer_pred.get(outcome, 0) +
                weights['time'] * time_pred.get(outcome, 0) +
                weights['rf'] * rf_pred.get(outcome, 0)
            )
            ensemble.append((outcome, score))
        
        # সর্ট ও নরমালাইজ
        ensemble.sort(key=lambda x: x[1], reverse=True)
        total_score = sum(s for _, s in ensemble[:5])
        
        if total_score > 0:
            ensemble = [(o, s/total_score) for o, s in ensemble[:5]]
        else:
            ensemble = [('1', 0.5), ('2', 0.3), ('CoinFlip', 0.2)]
        
        return ensemble
    
    def save_model(self):
        """মডেল সেভ"""
        try:
            model_data = {
                'label_encoder': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
                'accuracy_history': self.accuracy_history[-100:],  # শেষ ১০০
                'last_trained': datetime.now().isoformat(),
                'outcome_frequencies': dict(self.outcome_frequencies.most_common(50)),
                'scaler_fitted': self.scaler_fitted
            }
            
            # পিকল ফাইল সেভ
            with open('models/statistical_model.pkl', 'wb') as f:
                pickle.dump({
                    'markov_chains': dict(self.markov_chains),
                    'dealer_patterns': dict(self.dealer_patterns),
                    'time_patterns': dict(self.time_patterns)
                }, f)
            
            # আরএফ মডেল সেভ
            if self.rf_trained:
                with open('models/rf_model.pkl', 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            # স্কেলার সেভ
            if self.scaler_fitted:
                with open('models/scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # রেডিসে সেভ (যদি থাকে)
            if self.redis:
                self.redis.setex('ai_model_meta', 86400, json.dumps(model_data))
            
            logger.info("💾 মডেল সফলভাবে সংরক্ষিত")
            
        except Exception as e:
            logger.error(f"❌ মডেল সেভে ত্রুটি: {e}")
    
    def load_model(self):
        """মডেল লোড"""
        try:
            # পিকল ফাইল লোড
            if os.path.exists('models/statistical_model.pkl'):
                with open('models/statistical_model.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.markov_chains = defaultdict(lambda: defaultdict(int), data.get('markov_chains', {}))
                    self.dealer_patterns = defaultdict(lambda: defaultdict(int), data.get('dealer_patterns', {}))
                    self.time_patterns = defaultdict(lambda: defaultdict(int), data.get('time_patterns', {}))
            
            # আরএফ মডেল লোড
            if os.path.exists('models/rf_model.pkl'):
                with open('models/rf_model.pkl', 'rb') as f:
                    self.rf_model = pickle.load(f)
                    self.rf_trained = True
            
            # স্কেলার লোড
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                    self.scaler_fitted = True
            
            # রেডিস থেকে মেটা ডাটা লোড
            if self.redis:
                meta = self.redis.get('ai_model_meta')
                if meta:
                    data = json.loads(meta)
                    if data.get('label_encoder'):
                        self.label_encoder.classes_ = np.array(data['label_encoder'])
                    self.accuracy_history = data.get('accuracy_history', [])
                    if data.get('last_trained'):
                        self.last_trained = datetime.fromisoformat(data['last_trained'])
                    if data.get('outcome_frequencies'):
                        self.outcome_frequencies = Counter(data['outcome_frequencies'])
            
            logger.info(f"✅ মডেল লোড করা হয়েছে (শেষ ট্রেন: {self.last_trained})")
            
        except Exception as e:
            logger.error(f"❌ মডেল লোডে ত্রুটি: {e}")
    
    def get_model_stats(self):
        """মডেল পরিসংখ্যান"""
        return {
            'total_outcomes': sum(self.outcome_frequencies.values()),
            'unique_outcomes': len(self.outcome_frequencies),
            'markov_patterns': len(self.markov_chains),
            'dealer_patterns': len(self.dealer_patterns),
            'accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.125,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'rf_trained': self.rf_trained,
            'tf_available': self.tf_available
        }