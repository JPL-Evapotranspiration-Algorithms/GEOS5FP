# Define the package name, environment name, and Python version
PACKAGE_NAME = GEOS5FP
MODULE_NAME = $(subst -,_,$(PACKAGE_NAME))
ENVIRONMENT_NAME = $(PACKAGE_NAME)
PYTHON_VERSION = $(if $(PYTHON),$(PYTHON),3.14)

# Allow passing extra flags via command line: make <target> ARGS="-k test_name"
ARGS ?=

# Include a Makefile.local to override things, if found
-include Makefile.local

.PHONY: clean test build wheel sdist twine-upload dist install uninstall reinstall \
        environment remove-environment colima-start colima-reset \
        repomix generate-history show-issues

# Clean up build artifacts and temporary files
clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

# Run Python unit tests using pytest
test:
	pytest -vv -p no:warnings $(ARGS)

# Build the Python wheel and source distribution
wheel:
	python -m build --wheel

sdist:
	python -m build --sdist

build:
	python -m build

# Upload the built package to PyPI using Twine
twine-upload:
	twine upload dist/*

# Clean, build, and upload the package
dist:
	$(MAKE) clean
	$(MAKE) build
	$(MAKE) twine-upload

# Install the package in editable mode with development dependencies
install:
	pip install -e .[dev]

# Uninstall the package
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

# Reinstall the package (uninstall and then install)
reinstall:
	$(MAKE) uninstall
	$(MAKE) install

# Create a new Conda/Mamba environment with the specified Python version
environment:
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=$(PYTHON_VERSION)

# Remove the Conda/Mamba environment
remove-environment:
	mamba env remove -y -n $(ENVIRONMENT_NAME)

# Start Colima with the required settings to prevent Rosetta 2 crashes
colima-start:
	colima start --vm-type=vz --vz-rosetta=true --cpu 6 --memory 16 --disk 100

# Completely reset the Colima VM before starting to prevent vz/qemu driver hanging issues
colima-reset:
	colima delete -f
	colima start --vm-type=vz --vz-rosetta=true --cpu 6 --memory 16 --disk 100

# Pack repository context for LLM review
repomix:
	npx repomix -o $(PACKAGE_NAME)-repomix.xml --ignore "**/*.csv,!GEOS5FP/variables.csv,**/*.ipynb,experiments/**,guides/**,notes/**,performance/**,LICENSE,*-repomix*.xml" --copy

# Generate release commit history patch
generate-history:
	fish -c ' \
		git fetch --tags --quiet; \
		set first_commit (git rev-list --max-parents=0 HEAD | head -n 1); \
		set releases (gh release list --limit 1000 --json tagName,createdAt --jq "sort_by(.createdAt) | .[].tagName"); \
		begin; \
			echo "========================================"; \
			echo "REPOSITORY RELEASE TIMELINE"; \
			echo "========================================"; \
			gh release list --limit 1000; \
			echo ""; \
			if test (count $$releases) -ge 1; \
				set first_rel $$releases[1]; \
				echo "========================================"; \
				echo "INITIAL COMMIT ($$first_commit) -> FIRST RELEASE ($$first_rel)"; \
				echo "========================================"; \
				git log --reverse --stat -p $$first_rel -- "*.py" "pyproject.toml"; \
				echo ""; \
			end; \
			if test (count $$releases) -ge 2; \
				for i in (seq 2 (count $$releases)); \
					set prev_idx (math $$i - 1); \
					set prev $$releases[$$prev_idx]; \
					set curr $$releases[$$i]; \
					echo "========================================"; \
					echo "RELEASE COMMIT HISTORY: $$prev -> $$curr"; \
					echo "========================================"; \
					git log --reverse --stat -p $$prev..$$curr -- "*.py" "pyproject.toml"; \
					echo ""; \
				end; \
			end; \
			if test (count $$releases) -ge 1; \
				set latest_rel $$releases[-1]; \
				echo "========================================"; \
				echo "LATEST RELEASE ($$latest_rel) -> CURRENT WORKING TREE (HEAD + UNCOMMITTED)"; \
				echo "========================================"; \
				git log --reverse --stat -p $$latest_rel..HEAD -- "*.py" "pyproject.toml"; \
				echo ""; \
				git status -s -- "*.py" "pyproject.toml"; \
				echo ""; \
				git diff HEAD -- "*.py" "pyproject.toml"; \
			else; \
				echo "========================================"; \
				echo "INITIAL COMMIT ($$first_commit) -> CURRENT WORKING TREE"; \
				echo "========================================"; \
				git log --reverse --stat -p -- "*.py" "pyproject.toml"; \
				echo ""; \
				git status -s -- "*.py" "pyproject.toml"; \
				echo ""; \
				git diff HEAD -- "*.py" "pyproject.toml"; \
			end; \
		end > release_diffs.patch; \
	'

# List open GitHub issues
show-issues:
	gh issue list --state open --json number,title,body --template '{{range .}}## #{{.number}}: {{.title}}{{"\n\n"}}{{.body}}{{"\n\n---\n\n"}}{{end}}'