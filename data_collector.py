import requests
import time
from datetime import datetime
import json
import redis
from config import Config
import logging

# লগিং সেটআপ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.api_url = Config.API_URL
        self.table_id = Config.TABLE_ID
        self.redis_client = None
        self.local_cache = []
        self.is_collecting = False
        self.last_collected = None
        self.total_collected = 0
        
        # Redis কানেকশন
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL, socket_connect_timeout=5, socket_timeout=5)
            self.redis_client.ping()
            logger.info("✅ Redis কানেক্টেড")
        except Exception as e:
            logger.warning(f"⚠️ Redis কানেক্ট হয়নি: {e}, লোকাল মোডে চলবে")
            self.redis_client = None
    
    def fetch_all_data(self):
        """ডাটা সংগ্রহ"""
        if self.is_collecting:
            logger.info("⚠️ ডাটা কালেকশন ইতিমধ্যে চলছে...")
            return
        
        self.is_collecting = True
        logger.info("=" * 60)
        logger.info("🚀 ডাটা কালেকশন শুরু...")
        logger.info("=" * 60)
        
        try:
            all_data = []
            page = 0
            empty_count = 0
            
            while page < Config.MAX_PAGES and empty_count < 3:
                url = f"{self.api_url}?page={page}&size={Config.PAGE_SIZE}&sort=data.settledAt,desc&duration=72&wheelResults=Pachinko,CashHunt,CrazyBonus,CoinFlip,1,2,5,10&isTopSlotMatched=true,false&tableId={self.table_id}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Origin': 'https://www.casino.org',
                    'Referer': 'https://www.casino.org/'
                }
                
                try:
                    logger.info(f"📡 পৃষ্ঠা {page+1} থেকে ডাটা সংগ্রহ করা হচ্ছে...")
                    response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        data = response.json()
                        page_data = data if isinstance(data, list) else data.get('content', [])
                        
                        if len(page_data) == 0:
                            empty_count += 1
                            logger.info(f"📭 পৃষ্ঠা {page+1} খালি (empty count: {empty_count})")
                        else:
                            all_data.extend(page_data)
                            logger.info(f"✅ পৃষ্ঠা {page+1}: {len(page_data)} টি (মোট: {len(all_data)})")
                            empty_count = 0
                            
                        page += 1
                    else:
                        logger.error(f"❌ API error: {response.status_code}")
                        break
                        
                except requests.exceptions.Timeout:
                    logger.error(f"❌ পৃষ্ঠা {page+1} টাইমআউট")
                    break
                except requests.exceptions.ConnectionError:
                    logger.error(f"❌ পৃষ্ঠা {page+1} কানেকশন error")
                    break
                except Exception as e:
                    logger.error(f"❌ পৃষ্ঠা {page+1} ত্রুটি: {e}")
                    break
                
                # API ব্লক না করার জন্য ডেলay
                time.sleep(Config.REQUEST_DELAY)
            
            # মেমোরি লিমিট
            if len(all_data) > Config.MAX_DATA_POINTS:
                all_data = all_data[-Config.MAX_DATA_POINTS:]
                logger.info(f"📊 মেমোরি লিমিট: সর্বশেষ {Config.MAX_DATA_POINTS} টি রাখা হয়েছে")
            
            # ডাটা সর্ট (পুরোনো → নতুন)
            logger.info("🔄 ডাটা সাজানো হচ্ছে...")
            all_data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
            
            # সংরক্ষণ
            self.save_to_cache(all_data)
            
            self.last_collected = datetime.now()
            self.total_collected = len(all_data)
            
            logger.info("=" * 60)
            logger.info(f"✅ কালেকশন সম্পন্ন! মোট: {len(all_data)} টি")
            logger.info("=" * 60)
            
            return all_data
            
        except Exception as e:
            logger.error(f"❌ গুরুতর ত্রুটি: {e}")
            return []
        finally:
            self.is_collecting = False
    
    def check_new_data(self):
        """নতুন ডাটা চেক"""
        try:
            url = f"https://api-cs.casino.org/svc-evolution-game-events/api/crazytime/latest?tableId={self.table_id}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Origin': 'https://www.casino.org',
                'Referer': 'https://www.casino.org/'
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                new_data = response.json()
                
                if new_data and new_data.get('id'):
                    # চেক করুন ডাটা আগে আছে কিনা
                    exists = False
                    
                    if self.redis_client:
                        existing = self.redis_client.get(f"data:{new_data['id']}")
                        exists = existing is not None
                    else:
                        exists = any(d.get('id') == new_data['id'] for d in self.local_cache)
                    
                    if not exists:
                        logger.info(f"✨ নতুন ডাটা পাওয়া গেছে! ID: {new_data['id']}")
                        self.save_single_data(new_data)
                        return True
            return False
            
        except requests.exceptions.Timeout:
            logger.debug("⏱️ API টাইমআউট - নর্�াল ব্যাপার")
            return False
        except requests.exceptions.ConnectionError:
            logger.debug("🔌 কানেকশন error - নর্�াল ব্যাপার")
            return False
        except Exception as e:
            logger.error(f"❌ নতুন ডাটা চেকে ত্রুটি: {e}")
            return False
    
    def save_to_cache(self, data):
        """ক্যাশে সংরক্ষণ"""
        try:
            if self.redis_client:
                # Redis ক্লিয়ার
                try:
                    self.redis_client.flushdb()
                except:
                    pass
                
                # নতুন ডাটা সংরক্ষণ
                pipeline = self.redis_client.pipeline()
                for item in data:
                    pipeline.setex(
                        f"data:{item['id']}",
                        Config.CACHE_TIMEOUT,
                        json.dumps(item)
                    )
                    pipeline.lpush("data_list", item['id'])
                
                # লিস্ট সাইজ লিমিট
                pipeline.ltrim("data_list", 0, Config.MAX_DATA_POINTS - 1)
                
                # মেটাডাটা
                metadata = {
                    'total_count': len(data),
                    'last_update': datetime.now().isoformat(),
                    'oldest': data[0].get('data', {}).get('settledAt') if data else None,
                    'newest': data[-1].get('data', {}).get('settledAt') if data else None
                }
                pipeline.setex("metadata", Config.CACHE_TIMEOUT, json.dumps(metadata))
                
                pipeline.execute()
                logger.info(f"💾 Redis-এ {len(data)} টি ডাটা সংরক্ষিত")
            else:
                # লোকাল ক্যাশ
                self.local_cache = data
                logger.info(f"💾 লোকাল ক্যাশে {len(data)} টি ডাটা সংরক্ষিত")
                
        except Exception as e:
            logger.error(f"❌ ক্যাশ সংরক্ষণে ত্রুটি: {e}")
            self.local_cache = data  # fallback
    
    def save_single_data(self, data):
        """একক ডাটা সংরক্ষণ"""
        try:
            if self.redis_client:
                # Redis-এ সংরক্ষণ
                self.redis_client.setex(
                    f"data:{data['id']}",
                    Config.CACHE_TIMEOUT,
                    json.dumps(data)
                )
                self.redis_client.lpush("data_list", data['id'])
                self.redis_client.ltrim("data_list", 0, Config.MAX_DATA_POINTS - 1)
                
                # মেটাডাটা আপডেট
                metadata = self.get_metadata()
                if metadata:
                    metadata['total_count'] = min(metadata.get('total_count', 0) + 1, Config.MAX_DATA_POINTS)
                    metadata['last_update'] = datetime.now().isoformat()
                    metadata['newest'] = data.get('data', {}).get('settledAt')
                    self.redis_client.setex("metadata", Config.CACHE_TIMEOUT, json.dumps(metadata))
            else:
                # লোকাল ক্যাশ
                self.local_cache.append(data)
                if len(self.local_cache) > Config.MAX_DATA_POINTS:
                    self.local_cache = self.local_cache[-Config.MAX_DATA_POINTS:]
                    
        except Exception as e:
            logger.error(f"❌ ডাটা সংরক্ষণে ত্রুটি: {e}")
    
    def get_all_data(self):
        """সব ডাটা লোড"""
        try:
            if self.redis_client:
                # Redis থেকে ডাটা আইডি লোড
                data_ids = self.redis_client.lrange("data_list", 0, -1)
                data = []
                
                if data_ids:
                    # পাইপলাইনে ডাটা লোড
                    pipeline = self.redis_client.pipeline()
                    for data_id in data_ids:
                        pipeline.get(f"data:{data_id.decode()}")
                    
                    results = pipeline.execute()
                    
                    for item in results:
                        if item:
                            try:
                                data.append(json.loads(item))
                            except:
                                pass
                
                # সর্ট (পুরোনো প্রথমে)
                data.sort(key=lambda x: x.get('data', {}).get('settledAt', ''))
                return data
            else:
                return self.local_cache
                
        except Exception as e:
            logger.error(f"❌ ডাটা লোডে ত্রুটি: {e}")
            return self.local_cache
    
    def get_metadata(self):
        """মেটাডাটা লোড"""
        try:
            if self.redis_client:
                meta = self.redis_client.get("metadata")
                return json.loads(meta) if meta else None
        except:
            pass
        return None
    
    def get_latest_data(self):
        """সর্বশেষ ডাটা লোড"""
        try:
            data = self.get_all_data()
            if data:
                return data[-1]
            return None
        except:
            return None
    
    def get_data_count(self):
        """ডাটা কাউন্ট"""
        try:
            data = self.get_all_data()
            return len(data)
        except:
            return 0
    
    def clear_cache(self):
        """ক্যাশ ক্লিয়ার"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
                logger.info("🗑️ Redis ক্যাশ ক্লিয়ার করা হয়েছে")
            else:
                self.local_cache = []
                logger.info("🗑️ লোকাল ক্যাশ ক্লিয়ার করা হয়েছে")
        except Exception as e:
            logger.error(f"❌ ক্যাশ ক্লিয়ার ত্রুটি: {e}")
    
    def get_data_by_id(self, data_id):
        """আইডি দিয়ে ডাটা খুঁজে বের করা"""
        try:
            if self.redis_client:
                data = self.redis_client.get(f"data:{data_id}")
                return json.loads(data) if data else None
            else:
                return next((d for d in self.local_cache if d.get('id') == data_id), None)
        except:
            return None
    
    def get_data_range(self, start, end):
        """নির্দিষ্ট রেঞ্জের ডাটা"""
        try:
            data = self.get_all_data()
            if start < 0:
                start = 0
            if end > len(data):
                end = len(data)
            return data[start:end]
        except:
            return []
    
    def get_statistics(self):
        """পরিসংখ্যান"""
        data = self.get_all_data()
        if not data:
            return {
                'total': 0,
                'oldest': None,
                'newest': None,
                'outcome_stats': {}
            }
        
        # আউটকাম পরিসংখ্যান
        outcome_stats = {}
        for item in data:
            try:
                outcome = item.get('data', {}).get('result', {}).get('outcome', {})
                wheel = outcome.get('wheelResult', {})
                
                if wheel.get('type') == 'Number':
                    outcome_val = str(wheel.get('wheelSector', '0'))
                else:
                    outcome_val = wheel.get('wheelSector', 'Unknown')
                
                outcome_stats[outcome_val] = outcome_stats.get(outcome_val, 0) + 1
            except:
                pass
        
        return {
            'total': len(data),
            'oldest': data[0].get('data', {}).get('settledAt') if data else None,
            'newest': data[-1].get('data', {}).get('settledAt') if data else None,
            'outcome_stats': outcome_stats
        }
