import numpy as np
from datetime import datetime, timedelta
import json
import logging
import threading
from collections import Counter, defaultdict
from dateutil import parser
import pickle
import os
import math

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
        self.n_features = 25  # বাড়ানো হয়েছে (বোনাসের জন্য)
        
        # আউটকাম লিস্ট - কনফিগ থেকে নেওয়া
        self.all_outcomes = Config.ALL_OUTCOMES
        self.bonus_games = Config.BONUS_GAMES
        self.bonus_shortcodes = Config.BONUS_SHORTCODES
        
        # মার্কোভ চেইন (মাল্টি-লেভেল)
        self.markov_chains_2gram = defaultdict(lambda: defaultdict(int))  # 2-gram
        self.markov_chains_3gram = defaultdict(lambda: defaultdict(int))  # 3-gram
        self.markov_chains_4gram = defaultdict(lambda: defaultdict(int))  # 4-gram
        
        # বোনাস-স্পেসিফিক মার্কোভ
        self.bonus_markov = {
            'CF': defaultdict(lambda: defaultdict(int)),
            'CH': defaultdict(lambda: defaultdict(int)),
            'CB': defaultdict(lambda: defaultdict(int)),
            'PC': defaultdict(lambda: defaultdict(int))
        }
        
        # বোনাস ট্রানজিশন ম্যাট্রিক্স
        self.bonus_transitions = defaultdict(lambda: defaultdict(int))
        
        # স্ট্যাটিস্টিক্যাল মডেল
        self.outcome_frequencies = Counter()
        self.dealer_patterns = defaultdict(lambda: defaultdict(int))
        self.time_patterns = defaultdict(lambda: defaultdict(int))
        self.multiplier_history = []
        
        # বোনাস স্ট্যাটস
        self.bonus_stats = {
            'CF': {'count': 0, 'avg_multiplier': 0, 'last_seen': None, 'gap_history': []},
            'CH': {'count': 0, 'avg_multiplier': 0, 'last_seen': None, 'gap_history': []},
            'CB': {'count': 0, 'avg_multiplier': 0, 'last_seen': None, 'gap_history': []},
            'PC': {'count': 0, 'avg_multiplier': 0, 'last_seen': None, 'gap_history': []}
        }
        
        # প্যাটার্ন কনফিডেন্স
        self.pattern_confidence = defaultdict(float)
        
        # ক্যাশ ডিরেক্টরি
        os.makedirs('models', exist_ok=True)
        
        # মডেল লোড
        self.load_model()
        
        logger.info("✅ AI মডেল ইনিশিয়ালাইজ করা হয়েছে (সমস্ত বোনাস সহ)")
    
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
        """রেজাল্ট থেকে আউটকাম ইনফো এক্সট্র্যাক্ট - সব বোনাস সহ"""
        try:
            data = result.get('data', {})
            outcome = data.get('result', {}).get('outcome', {})
            wheel = outcome.get('wheelResult', {})
            
            # আউটকাম টাইপ
            if wheel.get('type') == 'Number':
                outcome_val = str(wheel.get('wheelSector', '0'))
                outcome_category = 'number'
                bonus_type = None
            else:
                # বোনাস গেম
                raw_bonus = wheel.get('wheelSector', 'Unknown')
                # শর্টকোডে রূপান্তর
                outcome_val = self.bonus_shortcodes.get(raw_bonus, raw_bonus)
                outcome_category = 'bonus'
                bonus_type = raw_bonus
            
            # মাল্টিপ্লায়ার
            multiplier = float(outcome.get('maxMultiplier', 1))
            
            # ডিলার
            dealer = data.get('dealer', {}).get('name', 'Unknown')
            
            # সময়
            settled = data.get('settledAt')
            dt = self.parse_datetime(settled)
            
            return {
                'outcome': outcome_val,
                'raw_outcome': wheel.get('wheelSector', 'Unknown'),
                'category': outcome_category,
                'bonus_type': bonus_type,
                'multiplier': multiplier,
                'dealer': dealer,
                'timestamp': dt,
                'hour': dt.hour if dt else 0,
                'minute': dt.minute if dt else 0,
                'weekday': dt.weekday() if dt else 0,
                'is_bonus': outcome_category == 'bonus'
            }
        except Exception as e:
            logger.error(f"আউটকাম এক্সট্র্যাক্ট ত্রুটি: {e}")
            return {
                'outcome': 'Unknown',
                'raw_outcome': 'Unknown',
                'category': 'unknown',
                'bonus_type': None,
                'multiplier': 1,
                'dealer': 'Unknown',
                'timestamp': None,
                'hour': 0,
                'minute': 0,
                'weekday': 0,
                'is_bonus': False
            }
    
    def update_statistical_models(self, outcomes_info):
        """স্ট্যাটিস্টিক্যাল মডেল আপডেট - সব বোনাস সহ"""
        if len(outcomes_info) < 10:
            return
        
        outcomes = [info['outcome'] for info in outcomes_info]
        bonus_outcomes = [info for info in outcomes_info if info['is_bonus']]
        
        # ফ্রিকোয়েন্সি আপডেট
        for info in outcomes_info:
            self.outcome_frequencies[info['outcome']] += 1
        
        # মার্কোভ চেইন আপডেট (মাল্টি-লেভেল)
        # 2-gram
        for i in range(len(outcomes) - 2):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}"
            next_outcome = outcomes[i+2]
            self.markov_chains_2gram[pattern][next_outcome] += 1
        
        # 3-gram
        for i in range(len(outcomes) - 3):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}"
            next_outcome = outcomes[i+3]
            self.markov_chains_3gram[pattern][next_outcome] += 1
        
        # 4-gram
        for i in range(len(outcomes) - 4):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}|{outcomes[i+3]}"
            next_outcome = outcomes[i+4]
            self.markov_chains_4gram[pattern][next_outcome] += 1
        
        # বোনাস-স্পেসিফিক মার্কোভ
        for i in range(len(bonus_outcomes) - 2):
            if i+2 < len(bonus_outcomes):
                bonus_type = bonus_outcomes[i]['outcome']
                pattern = f"{bonus_outcomes[i]['outcome']}|{bonus_outcomes[i+1]['outcome']}"
                next_bonus = bonus_outcomes[i+2]['outcome']
                self.bonus_markov[bonus_type][pattern][next_bonus] += 1
        
        # বোনাস ট্রানজিশন
        for i in range(len(bonus_outcomes) - 1):
            current = bonus_outcomes[i]['outcome']
            next_bonus = bonus_outcomes[i+1]['outcome']
            self.bonus_transitions[current][next_bonus] += 1
        
        # বোনাস গ্যাপ ট্র্যাকিং
        last_bonus_index = {}
        for i, info in enumerate(outcomes_info):
            if info['is_bonus']:
                bonus = info['outcome']
                if bonus in last_bonus_index:
                    gap = i - last_bonus_index[bonus]
                    self.bonus_stats[bonus]['gap_history'].append(gap)
                last_bonus_index[bonus] = i
                self.bonus_stats[bonus]['count'] += 1
                
                # মাল্টিপ্লায়ার আপডেট
                old_avg = self.bonus_stats[bonus]['avg_multiplier']
                count = self.bonus_stats[bonus]['count']
                self.bonus_stats[bonus]['avg_multiplier'] = (old_avg * (count-1) + info['multiplier']) / count
                self.bonus_stats[bonus]['last_seen'] = info['timestamp']
        
        # ডিলার প্যাটার্ন আপডেট
        for i in range(len(outcomes_info) - 1):
            dealer = outcomes_info[i]['dealer']
            current = outcomes_info[i]['outcome']
            next_outcome = outcomes_info[i+1]['outcome']
            self.dealer_patterns[f"{dealer}|{current}"][next_outcome] += 1
        
        # টাইম প্যাটার্ন আপডেট
        for info in outcomes_info:
            hour_group = info['hour'] // 4
            self.time_patterns[f"hour_{hour_group}"][info['outcome']] += 1
            
            if info['multiplier'] > 10:
                self.time_patterns["high_multiplier"][info['outcome']] += 1
            
            if info['is_bonus'] and info['multiplier'] > 20:
                self.time_patterns["big_bonus"][info['outcome']] += 1
    
    def get_markov_prediction(self, last_n_outcomes, level=3):
        """মার্কোভ চেইন থেকে প্রেডিকশন - মাল্টি-লেভেল"""
        if len(last_n_outcomes) < level:
            return []
        
        if level == 2:
            pattern = f"{last_n_outcomes[-2]}|{last_n_outcomes[-1]}"
            chain = self.markov_chains_2gram
        elif level == 3:
            pattern = f"{last_n_outcomes[-3]}|{last_n_outcomes[-2]}|{last_n_outcomes[-1]}"
            chain = self.markov_chains_3gram
        elif level == 4:
            pattern = f"{last_n_outcomes[-4]}|{last_n_outcomes[-3]}|{last_n_outcomes[-2]}|{last_n_outcomes[-1]}"
            chain = self.markov_chains_4gram
        else:
            return []
        
        if pattern in chain:
            total = sum(chain[pattern].values())
            predictions = []
            for outcome, count in chain[pattern].items():
                predictions.append((outcome, count / total))
            
            # প্যাটার্ন কনফিডেন্স আপডেট
            self.pattern_confidence[pattern] = max(p[1] for p in predictions)
            
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        return []
    
    def get_bonus_prediction(self, last_bonuses):
        """বোনাস-স্পেসিফিক প্রেডিকশন"""
        if len(last_bonuses) < 2:
            return []
        
        predictions = defaultdict(float)
        
        # বোনাস ট্রানজিশন থেকে
        current_bonus = last_bonuses[-1]
        if current_bonus in self.bonus_transitions:
            total = sum(self.bonus_transitions[current_bonus].values())
            for bonus, count in self.bonus_transitions[current_bonus].items():
                predictions[bonus] += (count / total) * 0.4
        
        # বোনাস মার্কোভ থেকে
        if len(last_bonuses) >= 2:
            pattern = f"{last_bonuses[-2]}|{last_bonuses[-1]}"
            for bonus_type in self.bonus_markov:
                if pattern in self.bonus_markov[bonus_type]:
                    total = sum(self.bonus_markov[bonus_type][pattern].values())
                    for bonus, count in self.bonus_markov[bonus_type][pattern].items():
                        predictions[bonus] += (count / total) * 0.3
        
        # বোনাস গ্যাপ থেকে
        now = datetime.now()
        for bonus, stats in self.bonus_stats.items():
            if stats['last_seen'] and stats['gap_history']:
                hours_since = (now - stats['last_seen']).total_seconds() / 3600
                avg_gap = np.mean(stats['gap_history']) * 3  # 3 স্পিন = ~1 মিনিট
                
                if hours_since > avg_gap * 1.5:  # অনেকক্ষণ পর
                    predictions[bonus] += 0.2
                elif hours_since < avg_gap * 0.5:  # খুব তাড়াতাড়ি
                    predictions[bonus] -= 0.1
        
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    def calculate_time_decay_weight(self, timestamp):
        """টাইম-ডিকে ওয়েট ক্যালকুলেশন"""
        if not timestamp:
            return 1.0
        
        now = datetime.now()
        if timestamp.tzinfo:
            now = now.replace(tzinfo=timestamp.tzinfo)
        
        hours_old = (now - timestamp).total_seconds() / 3600
        return math.exp(-hours_old / 24)  # ২৪ ঘন্টায় অর্ধেক
    
    def get_ensemble_prediction(self, data, weights=None):
        """এনসেম্বল প্রেডিকশন - সব বোনাস সহ"""
        if not data or len(data) < 10:
            return [('1', 0.3), ('2', 0.2), ('5', 0.15), ('CF', 0.15), ('CH', 0.1), ('CB', 0.05), ('PC', 0.05)]
        
        # ডিফল্ট ওয়েট - নতুন ব্যালেন্স
        if weights is None:
            weights = {
                'frequency': 0.15,
                'markov_2gram': 0.15,
                'markov_3gram': 0.20,
                'markov_4gram': 0.15,
                'bonus_specific': 0.15,
                'dealer': 0.10,
                'time': 0.05,
                'rf': 0.05
            }
        
        # লাস্ট ডাটা
        last_10 = data[-10:]
        outcomes_info = [self.extract_outcome_info(d) for d in last_10]
        outcomes = [info['outcome'] for info in outcomes_info]
        last_bonuses = [info['outcome'] for info in outcomes_info if info['is_bonus']]
        
        # টাইম-ডিকে ওয়েট
        time_weights = [self.calculate_time_decay_weight(info['timestamp']) for info in outcomes_info]
        
        # ফ্রিকোয়েন্সি বেসড
        total = sum(self.outcome_frequencies.values())
        freq_pred = {}
        if total > 0:
            for outcome, count in self.outcome_frequencies.most_common(10):
                freq_pred[outcome] = count / total
        
        # মার্কোভ প্রেডিকশন (মাল্টি-লেভেল)
        markov_pred_2gram = dict(self.get_markov_prediction(outcomes, 2))
        markov_pred_3gram = dict(self.get_markov_prediction(outcomes, 3))
        markov_pred_4gram = dict(self.get_markov_prediction(outcomes, 4))
        
        # বোনাস স্পেসিফিক প্রেডিকশন
        bonus_pred = dict(self.get_bonus_prediction(last_bonuses))
        
        # ডিলার প্রেডিকশন
        dealer_pred = {}
        last_info = outcomes_info[-1]
        if last_info['dealer'] != 'Unknown':
            dealer_pattern = f"{last_info['dealer']}|{last_info['outcome']}"
            if dealer_pattern in self.dealer_patterns:
                total = sum(self.dealer_patterns[dealer_pattern].values())
                for outcome, count in self.dealer_patterns[dealer_pattern].items():
                    dealer_pred[outcome] = count / total
        
        # টাইম প্রেডিকশন
        time_pred = {}
        hour_group = last_info['hour'] // 4
        time_pattern = f"hour_{hour_group}"
        if time_pattern in self.time_patterns:
            total = sum(self.time_patterns[time_pattern].values())
            for outcome, count in self.time_patterns[time_pattern].items():
                time_pred[outcome] = count / total
        
        # আরএফ প্রেডিকশন (যদি ট্রেন করা থাকে)
        rf_pred = {}
        if self.rf_trained and self.scaler_fitted:
            try:
                features, _ = self.prepare_features(last_10)
                if len(features) > 0:
                    features_scaled = self.scaler.transform([features[-1]])
                    proba = self.rf_model.predict_proba(features_scaled)[0]
                    for i, prob in enumerate(proba):
                        outcome = self.label_encoder.inverse_transform([i])[0]
                        rf_pred[outcome] = float(prob)
            except Exception as e:
                logger.error(f"RF প্রেডিকশন ত্রুটি: {e}")
        
        # এনসেম্বল
        all_outcomes = set(self.all_outcomes)
        ensemble = []
        
        for outcome in all_outcomes:
            score = (
                weights['frequency'] * freq_pred.get(outcome, 0) +
                weights['markov_2gram'] * markov_pred_2gram.get(outcome, 0) +
                weights['markov_3gram'] * markov_pred_3gram.get(outcome, 0) +
                weights['markov_4gram'] * markov_pred_4gram.get(outcome, 0) +
                weights['bonus_specific'] * bonus_pred.get(outcome, 0) +
                weights['dealer'] * dealer_pred.get(outcome, 0) +
                weights['time'] * time_pred.get(outcome, 0) +
                weights['rf'] * rf_pred.get(outcome, 0)
            )
            
            # টাইম-ডিকে অ্যাডজাস্টমেন্ট
            if outcome in [o['outcome'] for o in outcomes_info]:
                idx = outcomes.index(outcome) if outcome in outcomes else -1
                if idx >= 0:
                    score *= (1 - time_weights[idx] * 0.3)  # সম্প্রতি আসলে কম গুরুত্ব
            
            ensemble.append((outcome, score))
        
        # সর্ট ও নরমালাইজ
        ensemble.sort(key=lambda x: x[1], reverse=True)
        total_score = sum(s for _, s in ensemble[:5])
        
        if total_score > 0:
            ensemble = [(o, s/total_score) for o, s in ensemble[:8]]
        else:
            ensemble = [('1', 0.2), ('2', 0.15), ('5', 0.15), ('10', 0.1),
                       ('CF', 0.15), ('CH', 0.1), ('CB', 0.08), ('PC', 0.07)]
        
        return ensemble
    
    def calculate_markov_accuracy(self, outcomes_info):
        """মার্কোভ চেইনের অ্যাকুরেসি ক্যালকুলেশন - সব বোনাস সহ"""
        if len(outcomes_info) < 100:
            return 0.3
        
        outcomes = [info['outcome'] for info in outcomes_info]
        correct_3gram = 0
        total_3gram = 0
        
        for i in range(len(outcomes) - 4):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}"
            actual = outcomes[i+3]
            
            if pattern in self.markov_chains_3gram:
                predicted = max(self.markov_chains_3gram[pattern].items(), key=lambda x: x[1])[0]
                if predicted == actual:
                    correct_3gram += 1
                total_3gram += 1
        
        return correct_3gram / total_3gram if total_3gram > 0 else 0.3
    
    def prepare_features(self, data):
        """ফিচার তৈরি - বোনাস ফিচার সহ"""
        features = []
        outcomes = []
        
        sample_size = min(len(data), 2000)
        data = data[-sample_size:]
        
        for i, result in enumerate(data):
            info = self.extract_outcome_info(result)
            outcomes.append(info['outcome'])
            
            # বেসিক ফিচার
            feature = [
                info['hour'] / 24.0,
                info['weekday'] / 7.0,
                info['multiplier'] / 100.0,
                1.0 if info['is_bonus'] else 0.0,  # বোনাস ফ্ল্যাগ
            ]
            
            # বোনাস টাইপ এনকোডিং
            for bonus in ['CF', 'CH', 'CB', 'PC']:
                feature.append(1.0 if info['outcome'] == bonus else 0.0)
            
            # শেষ ৫টি আউটকামের হিস্ট্রি
            for j in range(1, 6):
                if i >= j:
                    prev_info = self.extract_outcome_info(data[i-j])
                    outcome_code = hash(prev_info['outcome']) % 100 / 100.0
                    feature.append(outcome_code)
                    feature.append(prev_info['multiplier'] / 100.0)
                    feature.append(1.0 if prev_info['is_bonus'] else 0.0)
                else:
                    feature.extend([0, 0, 0])
            
            # প্যাডিং
            while len(feature) < self.n_features:
                feature.append(0)
            
            features.append(feature[:self.n_features])
        
        return np.array(features, dtype=np.float32), outcomes
    
    def train(self, data):
        """মডেল ট্রেন - আপডেটেড এবং ফিক্সড"""
        if self.is_training or len(data) < Config.MIN_DATA_FOR_TRAINING:
            return
        
        self.is_training = True
        logger.info(f"🚀 মডেল ট্রেনিং শুরু... ({len(data)} টি ডাটা, সব বোনাস সহ)")
        
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
            
            # আউটকাম লিমিট (সব বোনাস রাখা)
            valid_outcomes = [o for o in outcomes if o in self.all_outcomes]
            
            # চেক করুন যে একাধিক ইউনিক আউটকাম আছে কিনা
            unique_outcomes = set(valid_outcomes)
            if len(unique_outcomes) < 2:
                logger.warning(f"⚠️ শুধু {len(unique_outcomes)} টি ইউনিক আউটকাম আছে, ট্রেনিং স্কিপ করা হচ্ছে")
                self.is_training = False
                return
            
            # এনকোডিং
            self.label_encoder.fit(valid_outcomes)
            y = self.label_encoder.transform(valid_outcomes)
            
            # ফিচার ম্যাচ করা
            X = features[:len(y)]
            
            # ট্রেন-টেস্ট স্প্লিট
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
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
            
            # বোনাস স্ট্যাটস প্রিন্ট
            bonus_counts = {b: self.bonus_stats[b]['count'] for b in self.bonus_stats}
            
            logger.info(f"✅ র্যান্ডম ফরেস্ট অ্যাকুরেসি: {accuracy:.2%}")
            logger.info(f"✅ মার্কোভ চেইন অ্যাকুরেসি: {markov_accuracy:.2%}")
            logger.info(f"📊 বোনাস কাউন্ট: {bonus_counts}")
            
            self.last_trained = datetime.now()
            
            # মডেল সেভ
            self.save_model()
            
        except Exception as e:
            logger.error(f"❌ ট্রেনিং ত্রুটি: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
    
    def save_model(self):
        """মডেল সেভ - সব বোনাস ডাটা সহ"""
        try:
            model_data = {
                'label_encoder': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
                'accuracy_history': self.accuracy_history[-100:],
                'last_trained': datetime.now().isoformat(),
                'outcome_frequencies': dict(self.outcome_frequencies.most_common(50)),
                'scaler_fitted': self.scaler_fitted,
                'bonus_stats': self.bonus_stats
            }
            
            # পিকল ফাইল সেভ
            with open('models/statistical_model.pkl', 'wb') as f:
                pickle.dump({
                    'markov_chains_2gram': dict(self.markov_chains_2gram),
                    'markov_chains_3gram': dict(self.markov_chains_3gram),
                    'markov_chains_4gram': dict(self.markov_chains_4gram),
                    'bonus_markov': {k: dict(v) for k, v in self.bonus_markov.items()},
                    'bonus_transitions': dict(self.bonus_transitions),
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
            
            # রেডিসে সেভ
            if self.redis:
                self.redis.setex('ai_model_meta', 86400, json.dumps(model_data))
            
            logger.info("💾 মডেল সফলভাবে সংরক্ষিত (সমস্ত বোনাস সহ)")
            
        except Exception as e:
            logger.error(f"❌ মডেল সেভে ত্রুটি: {e}")
    
    def load_model(self):
        """মডেল লোড - সব বোনাস ডাটা সহ"""
        try:
            # পিকল ফাইল লোড
            if os.path.exists('models/statistical_model.pkl'):
                with open('models/statistical_model.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.markov_chains_2gram = defaultdict(lambda: defaultdict(int), data.get('markov_chains_2gram', {}))
                    self.markov_chains_3gram = defaultdict(lambda: defaultdict(int), data.get('markov_chains_3gram', {}))
                    self.markov_chains_4gram = defaultdict(lambda: defaultdict(int), data.get('markov_chains_4gram', {}))
                    
                    # বোনাস ডাটা লোড
                    bonus_markov = data.get('bonus_markov', {})
                    for k, v in bonus_markov.items():
                        self.bonus_markov[k] = defaultdict(lambda: defaultdict(int), v)
                    
                    self.bonus_transitions = defaultdict(lambda: defaultdict(int), data.get('bonus_transitions', {}))
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
                    if data.get('bonus_stats'):
                        self.bonus_stats.update(data['bonus_stats'])
            
            logger.info(f"✅ মডেল লোড করা হয়েছে (শেষ ট্রেন: {self.last_trained}, সব বোনাস সহ)")
            
        except Exception as e:
            logger.error(f"❌ মডেল লোডে ত্রুটি: {e}")
    
    def get_model_stats(self):
        """মডেল পরিসংখ্যান - বোনাস ডাটা সহ"""
        return {
            'total_outcomes': sum(self.outcome_frequencies.values()),
            'unique_outcomes': len(self.outcome_frequencies),
            'markov_patterns': len(self.markov_chains_3gram),
            'bonus_patterns': sum(len(v) for v in self.bonus_markov.values()),
            'dealer_patterns': len(self.dealer_patterns),
            'accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.125,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'rf_trained': self.rf_trained,
            'tf_available': self.tf_available,
            'bonus_counts': {b: self.bonus_stats[b]['count'] for b in self.bonus_stats}
        }
