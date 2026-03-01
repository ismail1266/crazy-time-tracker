web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 4
worker: python -c "from data_collector import DataCollector; import time; dc=DataCollector(); while True: dc.check_new_data(); time.sleep(3)"
