# residual-shield — top-level interoperability Makefile.
#
# Mixed-language repository: a Python simulation/backend package under
# backend/ (pythalab-uavsim-backend, the authoritative uavsim simulator —
# see PAPER.yml notes) and a React/TypeScript/Three.js frontend at the repo
# root (src/, public/). Targets below wire the real, pre-existing commands
# documented in README.md and REPRODUCIBILITY.md; nothing here reimplements
# what those already describe. Verbs that do not apply are honest no-ops,
# per the collection standard.

.PHONY: setup test lint type smoke falsify paper check-paths repro check clean

setup:
	python3 -m pip install -e './backend[dev,experiments]'
	npm install

test:
	cd backend && python3 -m pytest -q

lint:
	cd backend && ruff check .

type:
	cd backend && pyright src tests
	npm run lint

smoke:
	cd backend && PYTHONPATH=src python3 -m uavsim.experiment_runner benchmark \
		--output-dir /tmp/uavsim-benchmark-smoke \
		--max-scenarios 1 \
		--duration 2 \
		--step-log-stride 10

falsify:
	@echo "falsify: no bespoke counterexample suite in this repo — running the backend regression tests (dynamics/controllers/server) as the closest executable check; see /falsify for an ad hoc counterexample pass on a specific change"
	cd backend && python3 -m pytest -q

paper:
	@echo "paper: no manuscript source in this repo — the LaTeX source lives in the companion repository (PAPER.yml:latex_home, PhiniteLab/sharq-twin); this repo is the companion-code artifact, not the manuscript home"

check-paths:
	python3 scripts/check_paths.py

repro:
	@echo "repro: fast smoke benchmark only — see REPRODUCIBILITY.md for the full raw-result refresh and hash-manifest procedure"
	cd backend && PYTHONPATH=src python3 -m uavsim.experiment_runner benchmark \
		--output-dir /tmp/uavsim-benchmark-smoke \
		--max-scenarios 1 \
		--duration 2 \
		--step-log-stride 10

check: lint type test check-paths

clean:
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache
	rm -rf dist node_modules/.vite
