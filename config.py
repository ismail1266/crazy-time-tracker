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
    MAX_PAGES = 200  # কমিয়ে দিলাম মেমোরির জন্য
    REQUEST_TIMEOUT = 15
    REQUEST_DELAY = 0.3  # ৩০০ms ডেলে
    
    # টেবিল স্ট্রাকচার
    ROWS_PER_COLUMN = 15
    
    # AI মডেল সেটিংস
    MODEL_SAVE_PATH = "models/"
    MIN_DATA_FOR_TRAINING = 50  # কমিয়ে দিলাম
    RETRAIN_INTERVAL = 3600
    
    # মেমোরি অপ্টিমাইজেশন
    MAX_DATA_POINTS = 2000  # সর্বোচ্চ ডাটা পয়েন্ট
    BATCH_SIZE = 100
    
    # ক্যাশ সেটিংস
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    CACHE_TIMEOUT = 86400  # ২৪ ঘন্টা
    
    # সার্ভার সেটিংস
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = False
    
    # হেলথ চেক
    HEALTH_CHECK_INTERVAL = 300
