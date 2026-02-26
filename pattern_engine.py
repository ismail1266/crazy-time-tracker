import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternEngine:
    """প্যাটার্ন ভিত্তিক প্রেডিকশন ইঞ্জিন"""
    
    def __init__(self):
        # প্যাটার্ন ওয়েটস (ডায়নামিক)
        self.pattern_weights = {
            'streak': 0.12,
            'gap': 0.08,
            'sequence_2gram': 0.10,
            'sequence_3gram': 0.15,
            'sequence_4gram': 0.10,
            'cycle': 0.07,
            'bonus': 0.10,
            'multiplier': 0.08,
            'time': 0.06,
            'hot_cold': 0.07,
            'correlation': 0.07
        }
        
        # পারফরমেন্স ট্র্যাকার
        self.pattern_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # ক্যাশ
        self.cache = {}
        self.cache_ttl = 300  # ৫ মিনিট
        
        logger.info("✅ প্যাটার্ন ইঞ্জিন ইনিশিয়ালাইজড")
    
    def predict(self, data: List, analyzer) -> List[Tuple[str, float]]:
        """সব প্যাটার্ন থেকে প্রেডিকশন"""
        
        if len(data) < 20:
            return self._get_default_prediction()
        
        # ক্যাশ চেক
        cache_key = f"pattern_pred_{len(data)}"
        if cache_key in self.cache:
            timestamp, pred = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return pred
        
        # আউটকাম ইনফো এক্সট্র্যাক্ট
        outcomes_info = []
        for result in data[-200:]:  # শেষ ২০০ স্পিন
            info = self._extract_info(result)
            if info:
                outcomes_info.append(info)
        
        outcomes = [info['outcome'] for info in outcomes_info]
        
        # সব প্যাটার্ন থেকে প্রেডিকশন
        predictions = {}
        
        # ১. স্ট্রিক প্যাটার্ন
        streak_pred = self._predict_from_streak(outcomes)
        self._add_weighted_prediction(predictions, streak_pred, 'streak')
        
        # ২. গ্যাপ প্যাটার্ন
        gap_pred = self._predict_from_gap(outcomes)
        self._add_weighted_prediction(predictions, gap_pred, 'gap')
        
        # ৩. সিকোয়েন্স প্যাটার্ন
        seq_2_pred = self._predict_from_sequence(outcomes, 2)
        seq_3_pred = self._predict_from_sequence(outcomes, 3)
        seq_4_pred = self._predict_from_sequence(outcomes, 4)
        
        self._add_weighted_prediction(predictions, seq_2_pred, 'sequence_2gram')
        self._add_weighted_prediction(predictions, seq_3_pred, 'sequence_3gram')
        self._add_weighted_prediction(predictions, seq_4_pred, 'sequence_4gram')
        
        # ৪. সাইকেল প্যাটার্ন
        if hasattr(analyzer, 'cycle_patterns'):
            cycle_pred = self._predict_from_cycle(outcomes, analyzer)
            self._add_weighted_prediction(predictions, cycle_pred, 'cycle')
        
        # ৫. বোনাস প্যাটার্ন
        bonus_pred = self._predict_from_bonus(outcomes_info)
        self._add_weighted_prediction(predictions, bonus_pred, 'bonus')
        
        # ৬. মাল্টিপ্লায়ার প্যাটার্ন
        mult_pred = self._predict_from_multiplier(outcomes_info)
        self._add_weighted_prediction(predictions, mult_pred, 'multiplier')
        
        # ৭. টাইম প্যাটার্ন
        if outcomes_info:
            time_pred = self._predict_from_time(outcomes_info[-1], analyzer)
            self._add_weighted_prediction(predictions, time_pred, 'time')
        
        # ৮. হট/কোল্ড
        hot_cold_pred = self._predict_from_hot_cold(outcomes)
        self._add_weighted_prediction(predictions, hot_cold_pred, 'hot_cold')
        
        # ৯. করিলেশন
        corr_pred = self._predict_from_correlation(outcomes_info)
        self._add_weighted_prediction(predictions, corr_pred, 'correlation')
        
        # ফাইনাল প্রেডিকশন
        result = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # নরমালাইজ
        total = sum(score for _, score in result)
        if total > 0:
            result = [(outcome, score/total) for outcome, score in result]
        else:
            result = self._get_default_prediction()
        
        # ক্যাশে সেভ
        self.cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def _add_weighted_prediction(self, predictions: Dict, pred_dict: Dict, pattern_name: str):
        """ওয়েটেড প্রেডিকশন যোগ করা"""
        weight = self.pattern_weights.get(pattern_name, 0.05)
        
        # পারফরমেন্স অ্যাডজাস্টমেন্ট
        perf = self.pattern_performance[pattern_name]
        if perf['total'] > 10:
            accuracy = perf['correct'] / perf['total']
            weight *= (0.5 + accuracy)  # 0.5x to 1.5x based on accuracy
        
        for outcome, prob in pred_dict.items():
            predictions[outcome] = predictions.get(outcome, 0) + (prob * weight)
    
    def _predict_from_streak(self, outcomes: List[str]) -> Dict[str, float]:
        """স্ট্রিক থেকে প্রেডিকশন"""
        if len(outcomes) < 3:
            return {}
        
        # বর্তমান স্ট্রিক চেক
        current = outcomes[-1]
        streak_length = 1
        
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == current:
                streak_length += 1
            else:
                break
        
        # স্ট্রিকের উপর ভিত্তি করে প্রেডিকশন
        predictions = {}
        
        if streak_length >= 3:
            # লম্বা স্ট্রিকের পর পরিবর্তনের সম্ভাবনা
            predictions[current] = 0.4  # 40% chance same continues
            for outcome in set(outcomes[-10:]):
                if outcome != current:
                    predictions[outcome] = 0.6 / (len(set(outcomes[-10:])) - 1)
        
        return predictions
    
    def _predict_from_gap(self, outcomes: List[str]) -> Dict[str, float]:
        """গ্যাপ থেকে প্রেডিকশন"""
        if len(outcomes) < 10:
            return {}
        
        last_outcome = outcomes[-1]
        last_pos = len(outcomes) - 1
        
        # আগেরবার কখন এসেছিল
        prev_pos = None
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == last_outcome:
                prev_pos = i
                break
        
        if prev_pos:
            gap = last_pos - prev_pos
            
            # এই গ্যাপের পর সাধারণত কি হয়
            predictions = {}
            # TODO: গ্যাপ প্যাটার্ন থেকে প্রেডিকশন
            pass
        
        return {}
    
    def _predict_from_sequence(self, outcomes: List[str], gram: int) -> Dict[str, float]:
        """সিকোয়েন্স থেকে প্রেডিকশন"""
        if len(outcomes) < gram + 1:
            return {}
        
        pattern = '|'.join(outcomes[-(gram):])
        
        # TODO: মার্কোভ চেইন থেকে প্রেডিকশন
        # এই অংশ pattern_analyzer থেকে নিবে
        
        return {}
    
    def _predict_from_cycle(self, outcomes: List[str], analyzer) -> Dict[str, float]:
        """সাইকেল থেকে প্রেডিকশন"""
        predictions = {}
        
        for outcome in set(outcomes[-20:]):
            if outcome in analyzer.cycle_lengths:
                cycle = analyzer.cycle_lengths[outcome]
                positions = [i for i, o in enumerate(outcomes) if o == outcome]
                
                if positions and cycle > 0:
                    last_pos = positions[-1]
                    current_pos = len(outcomes) - 1
                    gap = current_pos - last_pos
                    
                    if gap >= cycle * 0.8:  # সাইকেলের ৮০% হয়ে গেছে
                        predictions[outcome] = 0.7  # 70% chance আসবে
        
        return predictions
    
    def _predict_from_bonus(self, outcomes_info: List[Dict]) -> Dict[str, float]:
        """বোনাস প্যাটার্ন থেকে প্রেডিকশন"""
        predictions = {}
        
        bonus_types = ['CoinFlip', 'CashHunt', 'CrazyBonus', 'Pachinko']
        last_5 = [info['outcome'] for info in outcomes_info[-5:] if info]
        
        # শেষ ৫-এ বোনাস কম থাকলে
        bonus_count = sum(1 for o in last_5 if o in bonus_types)
        
        if bonus_count == 0:
            # বোনাস আসার সম্ভাবনা বাড়ে
            for bonus in bonus_types:
                predictions[bonus] = 0.25
        
        return predictions
    
    def _predict_from_multiplier(self, outcomes_info: List[Dict]) -> Dict[str, float]:
        """মাল্টিপ্লায়ার থেকে প্রেডিকশন"""
        if not outcomes_info:
            return {}
        
        last_mult = outcomes_info[-1]['multiplier']
        predictions = {}
        
        # হাই মাল্টিপ্লায়ারের পর
        if last_mult >= 50:
            predictions['1'] = 0.5
            predictions['2'] = 0.3
            predictions['5'] = 0.2
        elif last_mult >= 20:
            predictions['1'] = 0.4
            predictions['2'] = 0.3
            predictions['5'] = 0.2
            predictions['10'] = 0.1
        
        return predictions
    
    def _predict_from_time(self, last_info: Dict, analyzer) -> Dict[str, float]:
        """টাইম থেকে প্রেডিকশন"""
        if not last_info:
            return {}
        
        hour = last_info['hour']
        hour_group = hour // 4
        
        predictions = {}
        
        # এই সময়ে কি হওয়ার সম্ভাবনা বেশি
        time_key = f"group_{hour_group}"
        if time_key in analyzer.time_patterns:
            total = sum(analyzer.time_patterns[time_key].values())
            if total > 0:
                for outcome, count in analyzer.time_patterns[time_key].items():
                    predictions[outcome] = count / total
        
        return predictions
    
    def _predict_from_hot_cold(self, outcomes: List[str]) -> Dict[str, float]:
        """হট/কোল্ড থেকে প্রেডিকশন"""
        if len(outcomes) < 30:
            return {}
        
        recent = outcomes[-20:]
        recent_counts = Counter(recent)
        
        total = len(recent)
        predictions = {}
        
        for outcome, count in recent_counts.items():
            prob = count / total
            # হট জোনে বেশি ওয়েট
            if prob > 0.3:  # হট
                predictions[outcome] = prob * 1.2
            elif prob < 0.1:  # কোল্ড - রিবাউন্ড চান্স
                predictions[outcome] = 0.15
        
        return predictions
    
    def _predict_from_correlation(self, outcomes_info: List[Dict]) -> Dict[str, float]:
        """করিলেশন থেকে প্রেডিকশন"""
        predictions = {}
        
        # মাল্টিপ্লায়ার এবং আউটকামের সম্পর্ক
        mult_outcome_map = defaultdict(list)
        
        for info in outcomes_info[-50:]:
            mult_outcome_map[info['outcome']].append(info['multiplier'])
        
        # যে আউটকামে বেশি মাল্টিপ্লায়ার আসে
        for outcome, mults in mult_outcome_map.items():
            avg_mult = sum(mults) / len(mults)
            if avg_mult > 20:  # হাই মাল্টিপ্লায়ার আউটকাম
                predictions[outcome] = 0.3
        
        return predictions
    
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
                'multiplier': float(outcome.get('maxMultiplier', 1)),
                'timestamp': data.get('settledAt')
            }
        except:
            return None
    
    def _get_default_prediction(self) -> List[Tuple[str, float]]:
        """ডিফল্ট প্রেডিকশন"""
        return [('1', 0.5), ('2', 0.3), ('5', 0.2)]
    
    def update_performance(self, pattern_name: str, was_correct: bool):
        """প্যাটার্নের পারফরমেন্স আপডেট"""
        self.pattern_performance[pattern_name]['total'] += 1
        if was_correct:
            self.pattern_performance[pattern_name]['correct'] += 1
    
    def get_pattern_stats(self) -> Dict:
        """প্যাটার্ন পরিসংখ্যান"""
        stats = {}
        for pattern, perf in self.pattern_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                stats[pattern] = {
                    'accuracy': accuracy,
                    'weight': self.pattern_weights.get(pattern, 0.05),
                    'samples': perf['total']
                }
        return stats
