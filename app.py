# কনফিগারেশন ফাইল
import os
import sys
from datetime import timedelta

class Config:
    # API সেটিংস
    API_URL = "https://api-cs.casino.org/svc-evolution-game-events/api/crazytime"
    TABLE_ID = "CrazyTime0000001"
    
    # ডাটা কালেকশন সেটিংস
    PAGE_SIZE = 100
    MAX_PAGES = 200
    REQUEST_TIMEOUT = 5
    REQUEST_DELAY = 0.1
    
    # টেবিল স্ট্রাকচার
    ROWS_PER_COLUMN = 15
    
    # AI মডেল সেটিংস
    MODEL_SAVE_PATH = "models/"
    MIN_DATA_FOR_TRAINING = 50
    RETRAIN_INTERVAL = 3600
    
    # মেমোরি অপ্টিমাইজেশন
    MAX_DATA_POINTS = 3000
    BATCH_SIZE = 100

    # বোনাস গেমের তালিকা
    BONUS_GAMES = ['CoinFlip', 'CashHunt', 'CrazyBonus', 'Pachinko']
    BONUS_SHORTCODES = {
        'CoinFlip': 'CF',
        'CashHunt': 'CH', 
        'CrazyBonus': 'CB',
        'Pachinko': 'PC'
    }

    # সব সম্ভাব্য আউটকাম
    ALL_OUTCOMES = ['1', '2', '5', '10', 'CF', 'CH', 'CB', 'PC']

    # আউটকাম ক্যাটাগরি
    OUTCOME_CATEGORIES = {
        'number': ['1', '2', '5', '10'],
        'bonus': ['CF', 'CH', 'CB', 'PC']
    }

    # বোনাস রঙ (UI-র জন্য)
    BONUS_COLORS = {
        'CF': '#2d4d3a',
        'CH': '#4d3a2d',
        'CB': '#3a2d4d',
        'PC': '#2d4d4d'
    }
    
    # ক্যাশ সেটিংস
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    CACHE_TIMEOUT = 86400
    
    # সার্ভার সেটিংস
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = False
    
    # হেলথ চেক
    HEALTH_CHECK_INTERVAL = 300
