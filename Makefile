.PHONY: install lint format test validate-data train eval export api docker-build docker-up clean

PYTHON := python
UVICORN := uvicorn
PYTEST := pytest
RUFF := ruff

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

lint:
	$(RUFF) check src tests scripts api

format:
	$(RUFF) format src tests scripts api

test:
	$(PYTEST) -v

validate-data:
	$(PYTHON) scripts/validate_dataset.py --data dataset/data.yaml --output-dir outputs/dataset_reports

train:
	$(PYTHON) scripts/train.py --config configs/train.yaml

eval:
	$(PYTHON) scripts/evaluate.py --model runs/checkpoints/best.pt --data dataset/data.yaml --output-dir outputs/evaluation

export:
	$(PYTHON) scripts/export_model.py --model runs/checkpoints/best.pt --format onnx --output-dir outputs/exports

api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t veyraxis-sentinel:latest .

docker-up:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
