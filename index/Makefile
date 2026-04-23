IMAGE ?= local-index
PORT ?= 8001
PLATFORM ?= linux/amd64

.PHONY: build run

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) ./

run: build
	docker run --rm -p $(PORT):8000 $(IMAGE)
