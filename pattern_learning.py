import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
import pickle
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternLearning:
    """প্যাটার্ন লার্নিং সিস্টেম"""
    
    def __init__(self, engine, analyzer):
        self.engine = engine
        self.analyzer = analyzer
        self.learning_rate = 0.01
        self.min_samples = 20
        
        # লার্নিং ডাটা
        self.prediction_history = []
        self.pattern_accuracy = defaultdict(list)
        self.weight_adjustments = defaultdict(float)
        
        # মডেল ফাইল
        self.model_path = 'models/pattern_learning.pkl'
        self.load_model()
        
        logger.info("✅ প্যাটার্ন লার্নিং সিস্টেম ইনিশিয়ালাইজড")
    
    def learn_from_result(self, predicted: List[Tuple], actual: str, patterns_used: Dict):
        """প্রতি ফলাফল থেকে শেখা"""
        
        # ১. কোন প্রেডিকশন সঠিক ছিল
        correct = any(p[0] == actual for p in predicted[:3])
        
        # ২. প্রতিটি প্যাটার্নের পারফরমেন্স ট্র্যাক
        for pattern_name, was_used in patterns_used.items():
            if was_used:
                self.engine.update_performance(pattern_name, correct)
        
        # ৩. ওয়েট অ্যাডজাস্টমেন্ট
        self._adjust_weights(actual, predicted)
        
        # ৪. হিস্টোরিতে সেভ
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predicted': predicted,
            'actual': actual,
            'correct': correct
        })
        
        # ৫. মডেল সেভ
        if len(self.prediction_history) % 50 == 0:
            self.save_model()
    
    def _adjust_weights(self, actual: str, predicted: List[Tuple]):
        """প্যাটার্ন ওয়েট অ্যাডজাস্টমেন্ট"""
        
        for pattern_name, current_weight in self.engine.pattern_weights.items():
            # প্যাটার্নের পারফরমেন্স
            perf = self.engine.pattern_performance[pattern_name]
            
            if perf['total'] >= self.min_samples:
                accuracy = perf['correct'] / perf['total']
                
                # অ্যাকুরেসি অনুযায়ী ওয়েট অ্যাডজাস্ট
                target_weight = 0.05 + (accuracy * 0.15)  # 0.05 to 0.20
                
                # গ্র্যাজুয়াল অ্যাডজাস্টমেন্ট
                adjustment = (target_weight - current_weight) * self.learning_rate
                
                new_weight = current_weight + adjustment
                # লিমিটের মধ্যে রাখা
                new_weight = max(0.03, min(0.25, new_weight))
                
                self.engine.pattern_weights[pattern_name] = new_weight
                self.weight_adjustments[pattern_name] = adjustment
    
    def discover_new_patterns(self, data):
        """নতুন প্যাটার্ন ডিসকভারি"""
        
        # ১. অস্বাভাবিক সিকোয়েন্স খোঁজা
        unusual_sequences = self._find_unusual_sequences(data)
        
        # ২. নতুন করিলেশন খোঁজা
        new_correlations = self._find_new_correlations(data)
        
        # ৩. প্যাটার্ন লাইব্রেরিতে যোগ
        if unusual_sequences or new_correlations:
            logger.info(f"🔍 নতুন প্যাটার্ন পাওয়া গেছে!")
            
            if unusual_sequences:
                logger.info(f"  - {len(unusual_sequences)} টি অস্বাভাবিক সিকোয়েন্স")
            
            if new_correlations:
                logger.info(f"  - {len(new_correlations)} টি নতুন করিলেশন")
    
    def _find_unusual_sequences(self, data) -> List:
        """অস্বাভাবিক সিকোয়েন্স খোঁজা"""
        unusual = []
        
        outcomes = []
        for result in data[-200:]:
            info = self._extract_info(result)
            if info:
                outcomes.append(info['outcome'])
        
        # ৩-গ্রাম ফ্রিকোয়েন্সি
        sequences = defaultdict(int)
        for i in range(len(outcomes) - 3):
            seq = '|'.join(outcomes[i:i+3])
            sequences[seq] += 1
        
        # এক্সপেক্টেড ফ্রিকোয়েন্সি থেকে বেশি/কম
        for seq, count in sequences.items():
            if count > 3:  # কমপক্ষে ৩ বার দেখা গেছে
                expected = len(outcomes) / (len(set(outcomes)) ** 3)
                if count > expected * 2:  # ২x বেশি
                    unusual.append({
                        'sequence': seq,
                        'count': count,
                        'type': 'frequent'
                    })
                elif count < expected / 2 and count > 1:  # ২x কম
                    unusual.append({
                        'sequence': seq,
                        'count': count,
                        'type': 'rare'
                    })
        
        return unusual
    
    def _find_new_correlations(self, data) -> List:
        """নতুন করিলেশন খোঁজা"""
        correlations = []
        
        # বোনাস এবং মাল্টিপ্লায়ারের সম্পর্ক
        bonus_mult_map = defaultdict(list)
        
        for result in data[-100:]:
            info = self._extract_info(result)
            if info and info['outcome'] in ['CoinFlip', 'CashHunt', 'CrazyBonus', 'Pachinko']:
                bonus_mult_map[info['outcome']].append(info['multiplier'])
        
        for bonus, mults in bonus_mult_map.items():
            if len(mults) > 5:
                avg_mult = sum(mults) / len(mults)
                if avg_mult > 50:
                    correlations.append({
                        'type': 'high_multiplier_bonus',
                        'bonus': bonus,
                        'avg_multiplier': avg_mult
                    })
        
        return correlations
    
    def _extract_info(self, result) -> Dict:
        """রেজাল্ট থেকে ইনফো এক্সট্র্যাক্ট"""
        try:
            data = result.get('data', {})
            outcome = data.get('result', {}).get('outcome', {})
            wheel = outcome.get('wheelResult', {})
            
            if wheel.get('type') == 'Number':
                outcome_val = str(wheel.get('wheelSector', '0'))
            else:
                outcome_val = wheel.get('wheelSector', 'Unknown')
            
            return {
                'outcome': outcome_val,
                'multiplier': float(outcome.get('maxMultiplier', 1))
            }
        except:
            return None
    
    def save_model(self):
        """মডেল সেভ"""
        try:
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                'pattern_weights': dict(self.engine.pattern_weights),
                'pattern_performance': dict(self.engine.pattern_performance),
                'weight_adjustments': dict(self.weight_adjustments),
                'prediction_history': self.prediction_history[-500:]  # শেষ ৫০০
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("💾 প্যাটার্ন লার্নিং মডেল সেভ করা হয়েছে")
            
        except Exception as e:
            logger.error(f"❌ মডেল সেভে ত্রুটি: {e}")
    
    def load_model(self):
        """মডেল লোড"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # ওয়েট রিস্টোর
                if 'pattern_weights' in model_data:
                    for pattern, weight in model_data['pattern_weights'].items():
                        if pattern in self.engine.pattern_weights:
                            self.engine.pattern_weights[pattern] = weight
                
                # পারফরমেন্স রিস্টোর
                if 'pattern_performance' in model_data:
                    self.engine.pattern_performance.update(model_data['pattern_performance'])
                
                # হিস্টোরি রিস্টোর
                if 'prediction_history' in model_data:
                    self.prediction_history = model_data['prediction_history']
                
                logger.info(f"✅ প্যাটার্ন লার্নিং মডেল লোড করা হয়েছে")
            
        except Exception as e:
            logger.error(f"❌ মডেল লোডে ত্রুটি: {e}")
    
    def get_learning_stats(self) -> Dict:
        """লার্নিং পরিসংখ্যান"""
        stats = {
            'total_predictions': len(self.prediction_history),
            'recent_accuracy': 0,
            'pattern_performance': {},
            'weight_adjustments': dict(self.weight_adjustments)
        }
        
        if self.prediction_history:
            recent = self.prediction_history[-100:]
            correct = sum(1 for p in recent if p['correct'])
            stats['recent_accuracy'] = correct / len(recent) if recent else 0
        
        return stats
