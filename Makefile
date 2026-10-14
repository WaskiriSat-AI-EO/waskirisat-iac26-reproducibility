.PHONY: test lint verify-models table5 table6 table7 table8 parity test2-summary figures

test:
	pytest -q

lint:
	black --check . && ruff check .

verify-models:
	python reproduce/verify_models.py

# Targets below are stubs until each owner delivers. See reproduce/paper_map.md.
table5 table6 table7 table8 parity test2-summary figures:
	@echo "$@: not implemented yet - see reproduce/paper_map.md" && exit 1
