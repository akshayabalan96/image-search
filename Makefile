# Create and activate virtual environment, then install requirements

.PHONY: setup run clean

VENV_NAME = venv

setup:
	python3 -m venv $(VENV_NAME)
	./$(VENV_NAME)/bin/pip install --upgrade pip
	./$(VENV_NAME)/bin/pip install -r requirements.txt

run:
	./$(VENV_NAME)/bin/uvicorn app:app --reload

clean:
	rm -rf $(VENV_NAME)
