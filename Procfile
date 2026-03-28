web: find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; uvicorn main:app --host 0.0.0.0 --port $PORT
