ENVIRONMENT_NAME = GEOS5FP
DOCKER_IMAGE_NAME = $(ENVIRONMENT_NAME)

.PHONY: clean test build twine-upload dist remove-environment install uninstall reinstall environment colima-start docker-build docker-build-environment docker-build-installation docker-interactive docker-remove repomix

clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest

build:
	python -m build

twine-upload:
	twine upload dist/*

dist:
	make clean
	make build
	make twine-upload

remove-environment:
	mamba env remove -y -n $(ENVIRONMENT_NAME)

install:
	pip install -e .[dev]

uninstall:
	pip uninstall -y $(ENVIRONMENT_NAME)

reinstall:
	make uninstall
	make install

environment:
	mamba create -y -n $(ENVIRONMENT_NAME) -c conda-forge python=3.14

colima-start:
	colima start -m 16 -a x86_64 -d 100

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-build-environment:
	docker build --target environment -t $(DOCKER_IMAGE_NAME):latest .

docker-build-installation:
	docker build --target installation -t $(DOCKER_IMAGE_NAME):latest .

docker-interactive:
	docker run -it $(DOCKER_IMAGE_NAME) fish

docker-remove:
	docker rmi -f $(DOCKER_IMAGE_NAME)

repomix:
	npx repomix --ignore "**/*.csv,!GEOS5FP/variables.csv,**/*.ipynb,experiments/**,guides/**,notes/**,performance/**,LICENSE" --copy