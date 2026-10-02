# Reproduce every number, table and figure of the paper, then build the PDF.
PY ?= python3
SCRIPTS := $(sort $(wildcard scripts/s*_*.py))

.PHONY: all data test paper clean

all: test data paper

test:
	$(PY) -m pytest -q tests

data:
	@for s in $(SCRIPTS); do echo ">> $$s"; (cd scripts && $(PY) -u $$(basename $$s)) || exit 1; done

paper:
	@if [ -f paper/main.tex ]; then cd paper && tectonic -X compile main.tex; \
	else echo "paper/main.tex not present (the manuscript is not part of this repository)"; fi

clean:
	rm -f data/*.json paper/figures/*.pdf paper/figures/*.png paper/tables/*.tex paper/main.pdf
