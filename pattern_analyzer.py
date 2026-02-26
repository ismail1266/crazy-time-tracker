import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """সমস্ত ধরনের প্যাটার্ন এনালাইসিসের মাস্টার ক্লাস"""
    
    def __init__(self):
        # স্ট্রিক প্যাটার্ন
        self.streak_patterns = defaultdict(lambda: defaultdict(int))
        
        # গ্যাপ প্যাটার্ন
        self.gap_patterns = defaultdict(lambda: defaultdict(int))
        self.last_seen = {}
        
        # সিকোয়েন্স প্যাটার্ন (২-গ্রাম, ৩-গ্রাম, ৪-গ্রাম)
        self.sequence_patterns = {
            '2gram': defaultdict(lambda: defaultdict(int)),
            '3gram': defaultdict(lambda: defaultdict(int)),
            '4gram': defaultdict(lambda: defaultdict(int))
        }
        
        # সাইকেল প্যাটার্ন
        self.cycle_patterns = defaultdict(list)
        self.cycle_lengths = {}
        
        # বোনাস প্যাটার্ন
        self.bonus_patterns = defaultdict(lambda: defaultdict(int))
        self.bonus_sequences = defaultdict(lambda: defaultdict(int))
        
        # মাল্টিপ্লায়ার প্যাটার্ন
        self.multiplier_patterns = defaultdict(lambda: defaultdict(int))
        self.multiplier_ranges = {
            'low': (1, 5),
            'medium': (6, 20),
            'high': (21, 100),
            'ultra': (101, 1000)
        }
        
        # টাইম প্যাটার্ন
        self.time_patterns = defaultdict(lambda: defaultdict(int))
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        
        # হট/কোল্ড ট্র্যাকিং
        self.hot_cold = defaultdict(int)
        self.recent_window = 50
        
        # করিলেশন প্যাটার্ন
        self.correlations = defaultdict(lambda: defaultdict(float))
        
        logger.info("✅ প্যাটার্ন এনালাইজার ইনিশিয়ালাইজড")
    
    def analyze_all_patterns(self, outcomes_info: List[Dict]) -> Dict:
        """সমস্ত প্যাটার্ন একসাথে এনালাইসিস"""
        
        outcomes = [info['outcome'] for info in outcomes_info if info]
        multipliers = [info['multiplier'] for info in outcomes_info if info]
        
        patterns = {
            'streak': self._analyze_streak_patterns(outcomes),
            'gap': self._analyze_gap_patterns(outcomes),
            'sequence': self._analyze_sequence_patterns(outcomes),
            'cycle': self._analyze_cycle_patterns(outcomes),
            'bonus': self._analyze_bonus_patterns(outcomes_info),
            'multiplier': self._analyze_multiplier_patterns(outcomes, multipliers),
            'time': self._analyze_time_patterns(outcomes_info),
            'hot_cold': self._analyze_hot_cold(outcomes),
            'correlation': self._analyze_correlations(outcomes, multipliers)
        }
        
        return patterns
    
    def _analyze_streak_patterns(self, outcomes: List[str]) -> Dict:
        """স্ট্রিক প্যাটার্ন এনালাইসিস"""
        streaks = defaultdict(list)
        
        current_streak = 1
        current_outcome = outcomes[0] if outcomes else None
        
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                current_streak += 1
            else:
                if current_streak > 1:
                    streaks[current_outcome].append({
                        'length': current_streak,
                        'position': i - current_streak,
                        'next': outcomes[i] if i < len(outcomes) else None
                    })
                current_streak = 1
                current_outcome = outcomes[i]
        
        # স্ট্রিক এনালাইসিস
        streak_analysis = {}
        for outcome, streak_list in streaks.items():
            if streak_list:
                lengths = [s['length'] for s in streak_list]
                streak_analysis[outcome] = {
                    'max_streak': max(lengths),
                    'avg_streak': sum(lengths) / len(lengths),
                    'total_streaks': len(streak_list),
                    'streak_break_pattern': self._analyze_streak_breaks(streak_list)
                }
        
        return streak_analysis
    
    def _analyze_streak_breaks(self, streak_list: List) -> Dict:
        """স্ট্রিক ভাঙার প্যাটার্ন"""
        break_patterns = defaultdict(int)
        for streak in streak_list:
            if streak['next']:
                break_patterns[streak['next']] += 1
        return dict(break_patterns)
    
    def _analyze_gap_patterns(self, outcomes: List[str]) -> Dict:
        """গ্যাপ প্যাটার্ন এনালাইসিস"""
        gaps = defaultdict(list)
        last_pos = {}
        
        for i, outcome in enumerate(outcomes):
            if outcome in last_pos:
                gap = i - last_pos[outcome]
                gaps[outcome].append(gap)
            last_pos[outcome] = i
        
        gap_analysis = {}
        for outcome, gap_list in gaps.items():
            if gap_list:
                gap_analysis[outcome] = {
                    'min_gap': min(gap_list),
                    'max_gap': max(gap_list),
                    'avg_gap': sum(gap_list) / len(gap_list),
                    'common_gaps': Counter(gap_list).most_common(3)
                }
        
        return gap_analysis
    
    def _analyze_sequence_patterns(self, outcomes: List[str]) -> Dict:
        """মাল্টি-গ্রাম সিকোয়েন্স প্যাটার্ন"""
        sequences = {}
        
        # ২-গ্রাম
        for i in range(len(outcomes) - 2):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}"
            next_outcome = outcomes[i+2]
            self.sequence_patterns['2gram'][pattern][next_outcome] += 1
        
        # ৩-গ্রাম
        for i in range(len(outcomes) - 3):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}"
            next_outcome = outcomes[i+3]
            self.sequence_patterns['3gram'][pattern][next_outcome] += 1
        
        # ৪-গ্রাম
        for i in range(len(outcomes) - 4):
            pattern = f"{outcomes[i]}|{outcomes[i+1]}|{outcomes[i+2]}|{outcomes[i+3]}"
            next_outcome = outcomes[i+4]
            self.sequence_patterns['4gram'][pattern][next_outcome] += 1
        
        # টপ প্যাটার্ন বের করা
        for gram in ['2gram', '3gram', '4gram']:
            top_patterns = {}
            for pattern, next_dict in self.sequence_patterns[gram].items():
                total = sum(next_dict.values())
                if total >= 3:  # মিনিমাম থ্রেশহোল্ড
                    top_next = max(next_dict.items(), key=lambda x: x[1])
                    top_patterns[pattern] = {
                        'next': top_next[0],
                        'probability': top_next[1] / total,
                        'count': total
                    }
            
            # সর্ট করে টপ ১০
            sequences[gram] = dict(sorted(
                top_patterns.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:10])
        
        return sequences
    
    def _analyze_cycle_patterns(self, outcomes: List[str]) -> Dict:
        """সাইকেল প্যাটার্ন এনালাইসিস"""
        cycles = {}
        
        for outcome in set(outcomes):
            positions = [i for i, o in enumerate(outcomes) if o == outcome]
            if len(positions) > 3:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                
                # সাইকেল ডিটেকশন
                from statsmodels.tsa.stattools import acf
                try:
                    cycle_length = self._detect_cycle(gaps)
                    cycles[outcome] = {
                        'cycle_length': cycle_length,
                        'confidence': 'high' if cycle_length > 0 else 'low',
                        'next_expected': positions[-1] + cycle_length if cycle_length > 0 else None
                    }
                except:
                    pass
        
        return cycles
    
    def _detect_cycle(self, gaps: List[int]) -> int:
        """সাইকেল ডিটেকশন অ্যালগরিদম"""
        if len(gaps) < 4:
            return 0
        
        # মোড (সবচেয়ে কমন গ্যাপ)
        common_gap = Counter(gaps).most_common(1)
        if common_gap and common_gap[0][1] >= len(gaps) * 0.3:
            return common_gap[0][0]
        
        return 0
    
    def _analyze_bonus_patterns(self, outcomes_info: List[Dict]) -> Dict:
        """বোনাস রাউন্ডের প্যাটার্ন"""
        bonus_patterns = {
            'bonus_frequency': defaultdict(int),
            'bonus_sequences': defaultdict(int),
            'bonus_multipliers': defaultdict(list)
        }
        
        bonus_types = ['CoinFlip', 'CashHunt', 'CrazyBonus', 'Pachinko']
        
        for i, info in enumerate(outcomes_info):
            if info and info['outcome'] in bonus_types:
                bonus_patterns['bonus_frequency'][info['outcome']] += 1
                bonus_patterns['bonus_multipliers'][info['outcome']].append(info['multiplier'])
                
                # বোনাসের পরের ফলাফল
                if i < len(outcomes_info) - 1 and outcomes_info[i+1]:
                    next_outcome = outcomes_info[i+1]['outcome']
                    bonus_patterns['bonus_sequences'][f"{info['outcome']}→{next_outcome}"] += 1
        
        # এভারেজ মাল্টিপ্লায়ার
        for bonus in bonus_types:
            if bonus_patterns['bonus_multipliers'][bonus]:
                avg_mult = sum(bonus_patterns['bonus_multipliers'][bonus]) / len(bonus_patterns['bonus_multipliers'][bonus])
                bonus_patterns['bonus_multipliers'][bonus] = {
                    'avg': avg_mult,
                    'max': max(bonus_patterns['bonus_multipliers'][bonus]),
                    'count': len(bonus_patterns['bonus_multipliers'][bonus])
                }
        
        return bonus_patterns
    
    def _analyze_multiplier_patterns(self, outcomes: List[str], multipliers: List[float]) -> Dict:
        """মাল্টিপ্লায়ার প্যাটার্ন"""
        multiplier_patterns = {}
        
        for range_name, (low, high) in self.multiplier_ranges.items():
            pattern_key = f"mult_{range_name}"
            multiplier_patterns[pattern_key] = defaultdict(int)
        
        for i in range(len(multipliers) - 1):
            current_mult = multipliers[i]
            next_outcome = outcomes[i+1] if i+1 < len(outcomes) else None
            
            if next_outcome:
                for range_name, (low, high) in self.multiplier_ranges.items():
                    if low <= current_mult <= high:
                        multiplier_patterns[f"mult_{range_name}"][next_outcome] += 1
                        break
        
        # প্রোবাবিলিটিতে কনভার্ট
        for key in multiplier_patterns:
            total = sum(multiplier_patterns[key].values())
            if total > 0:
                multiplier_patterns[key] = {
                    outcome: count/total 
                    for outcome, count in multiplier_patterns[key].items()
                }
        
        return multiplier_patterns
    
    def _analyze_time_patterns(self, outcomes_info: List[Dict]) -> Dict:
        """টাইম বেসড প্যাটার্ন"""
        time_patterns = {}
        
        for info in outcomes_info:
            if info:
                hour = info['hour']
                hour_group = hour // 4  # ৪ ঘন্টার গ্রুপ
                outcome = info['outcome']
                
                self.time_patterns[f"hour_{hour}"][outcome] += 1
                self.time_patterns[f"group_{hour_group}"][outcome] += 1
                
                if info['weekday'] < 5:
                    self.time_patterns['weekday'][outcome] += 1
                else:
                    self.time_patterns['weekend'][outcome] += 1
        
        # প্রোবাবিলিটি ক্যালকুলেশন
        for period, outcomes in self.time_patterns.items():
            total = sum(outcomes.values())
            if total > 0:
                time_patterns[period] = {
                    outcome: count/total 
                    for outcome, count in outcomes.items()
                }
        
        return time_patterns
    
    def _analyze_hot_cold(self, outcomes: List[str]) -> Dict:
        """হট/কোল্ড জোন এনালাইসিস"""
        if len(outcomes) < self.recent_window:
            return {}
        
        recent = outcomes[-self.recent_window:]
        older = outcomes[:-self.recent_window]
        
        recent_counts = Counter(recent)
        older_counts = Counter(older) if older else Counter()
        
        hot_cold = {}
        all_outcomes = set(recent_counts.keys()) | set(older_counts.keys())
        
        for outcome in all_outcomes:
            recent_pct = recent_counts.get(outcome, 0) / len(recent)
            older_pct = older_counts.get(outcome, 0) / max(len(older), 1)
            
            if recent_pct > older_pct * 1.3:  # 30% বেশি
                hot_cold[outcome] = 'HOT'
            elif recent_pct < older_pct * 0.7:  # 30% কম
                hot_cold[outcome] = 'COLD'
            else:
                hot_cold[outcome] = 'NORMAL'
        
        return hot_cold
    
    def _analyze_correlations(self, outcomes: List[str], multipliers: List[float]) -> Dict:
        """করিলেশন প্যাটার্ন"""
        correlations = {}
        
        # আউটকাম এবং মাল্টিপ্লায়ারের করিলেশন
        outcome_mult_map = defaultdict(list)
        for outcome, mult in zip(outcomes, multipliers):
            outcome_mult_map[outcome].append(mult)
        
        for outcome, mult_list in outcome_mult_map.items():
            if len(mult_list) > 5:
                correlations[outcome] = {
                    'avg_multiplier': sum(mult_list) / len(mult_list),
                    'max_multiplier': max(mult_list),
                    'min_multiplier': min(mult_list),
                    'volatility': np.std(mult_list) if len(mult_list) > 1 else 0
                }
        
        return correlations
    
    def get_important_patterns(self, min_confidence: float = 0.6) -> Dict:
        """শুধু গুরুত্বপূর্ণ প্যাটার্ন রিটার্ন করে"""
        important = {}
        
        # ৩-গ্রাম সিকোয়েন্স চেক
        for pattern, data in self.sequence_patterns['3gram'].items():
            total = sum(data.values())
            if total >= 5:  # কমপক্ষে ৫ বার দেখা গেছে
                top_next = max(data.items(), key=lambda x: x[1])
                confidence = top_next[1] / total
                if confidence >= min_confidence:
                    important[f"seq_3_{pattern}"] = {
                        'pattern': pattern,
                        'next': top_next[0],
                        'confidence': confidence,
                        'count': total
                    }
        
        # স্ট্রিক প্যাটার্ন চেক
        for outcome, streaks in self.streak_patterns.items():
            if streaks and len(streaks) >= 3:
                important[f"streak_{outcome}"] = {
                    'outcome': outcome,
                    'details': streaks
                }
        
        return important
