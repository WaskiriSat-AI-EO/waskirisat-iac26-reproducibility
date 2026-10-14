.PHONY: test lint verify-models reproduce table2 table5 table6 table7 table8 parity test2-summary figures

test:
	pytest -q

lint:
	black --check . && ruff check .

verify-models:
	python reproduce/verify_models.py

# Run every paper item and compare with the paper's values (PENDING = not delivered yet).
reproduce:
	python -m reproduce.check

table5 table6 table7 table8:
	python -m reproduce.$@

parity:
	python -m reproduce.thresholds

test2-summary:
	python -m reproduce.test2_summary

figures:
	cd reproduce && python make_figures.py

# Table 2 (degradation profiles) has no script yet; see reproduce/paper_map.md.
table2:
	@echo "table2: not implemented yet - see reproduce/paper_map.md" && exit 1
