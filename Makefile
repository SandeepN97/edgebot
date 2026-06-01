# EdgeBot Makefile
# Targets: install, test, backtest, paper, live, lint

PYTHON   := python3
PIP      := pip3
PYTEST   := python3 -m pytest
RUFF     := ruff
MYPY     := mypy
DOCKER   := docker compose

.PHONY: install test backtest paper live lint clean help

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "EdgeBot — available make targets:"
	@echo ""
	@echo "  install   Install all Python dependencies"
	@echo "  test      Run full test suite with coverage"
	@echo "  backtest  Run historical backtest (2022-2024)"
	@echo "  paper     Start bot in paper-trading mode"
	@echo "  live      Start bot in live-trading mode (requires .env)"
	@echo "  lint      Run ruff linter + mypy type checker"
	@echo "  clean     Remove caches, .pyc files, and build artefacts"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ \
		--cov=domain \
		--cov=application \
		--cov=infrastructure \
		--cov=strategies \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

# ── Backtest ──────────────────────────────────────────────────────────────────
backtest:
	$(PYTHON) backtesting/run_backtest.py \
		--symbol BTC-USD \
		--start 2022-01-01 \
		--end   2024-12-31 \
		--cash  10000

backtest-eth:
	$(PYTHON) backtesting/run_backtest.py \
		--symbol ETH-USD \
		--start 2022-01-01 \
		--end   2024-12-31 \
		--cash  10000

# ── Run modes ─────────────────────────────────────────────────────────────────
paper:
	PAPER_TRADE=true $(PYTHON) main.py

live:
	@echo "WARNING: Live trading uses real funds. Ensure .env is configured."
	@read -p "Type 'yes' to proceed: " confirm && [ "$$confirm" = "yes" ] || exit 1
	PAPER_TRADE=false $(PYTHON) -m edgebot.main

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	$(DOCKER) up -d

docker-down:
	$(DOCKER) down

docker-logs:
	$(DOCKER) logs -f edgebot

docker-build:
	$(DOCKER) build

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check . --fix
	$(RUFF) format .
	$(MYPY) domain application strategies --ignore-missing-imports

lint-check:
	$(RUFF) check .
	$(MYPY) domain application strategies --ignore-missing-imports

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage