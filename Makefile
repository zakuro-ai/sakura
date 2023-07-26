# PHONY are targets with no files to check, all in our case
.PHONY: build
.DEFAULT_GOAL := build

PACKAGE_NAME=sakura

build:
	docker compose build vanilla
	docker compose build sandbox
	docker compose down
	docker compose up sandbox -d
	docker exec -it $(PACKAGE_NAME) bash

build_wheel: 
	# Build the wheels
	@mv dist/$(PACKAGE_NAME)*.whl dist/legacy/ || true; python setup.py bdist_wheel