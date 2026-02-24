#!/bin/bash

echo "🚀 ক্রেজি টাইম AI সেটআপ শুরু..."

# ভার্চুয়াল এনভায়রনমেন্ট অ্যাক্টিভেট
echo "🔧 ভার্চুয়াল এনভায়রনমেন্ট অ্যাক্টিভেট করা হচ্ছে..."
source .venv/bin/activate

# pip আপডেট
echo "📦 pip আপডেট করা হচ্ছে..."
pip install --upgrade pip setuptools wheel

# NumPy আলাদাভাবে ইনস্টল
echo "🔢 NumPy ইনস্টল করা হচ্ছে..."
pip install numpy==1.26.3

# বাকি ডিপেন্ডেন্সি
echo "📚 বাকি ডিপেন্ডেন্সি ইনস্টল করা হচ্ছে..."
pip install flask==3.0.0 flask-cors==4.0.1 requests==2.31.0 gunicorn==21.2.0 werkzeug==3.0.1
pip install pandas==2.1.4 scikit-learn==1.3.2 joblib==1.3.2 redis==5.0.1 apscheduler==3.10.4
pip install scipy==1.11.4 python-dotenv==1.0.0 python-dateutil==2.8.2
pip install tensorflow-cpu==2.15.0 keras==2.15.0

# Procfile রিনেম
if [ -f "Procfile.txt" ]; then
    mv Procfile.txt Procfile
    echo "✅ Procfile.txt → Procfile"
fi

# Redis সেটআপ
echo "🗄️ Redis সেটআপ হচ্ছে..."
sudo apt-get update
sudo apt-get install redis-server -y
redis-server --daemonize yes
echo "✅ Redis স্টার্ট করা হয়েছে"

# অ্যাপ রান
echo "🎯 অ্যাপ রান হচ্ছে..."
python app.py