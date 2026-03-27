# Day 21: Convenience commands for development

.PHONY: install test lint docker run clean

install:
	pip install -r requirements.txt

test:
	pytest test_day19.py -v

test-coverage:
	pytest test_day19.py -v --cov=src --cov-report=html

lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

docker-build:
	docker build -t crypto-predictor .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data crypto-predictor

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage

run-api:
	python -c "import sys; sys.path.insert(0, 'src'); from api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"

collect-data:
	python collect_500.py