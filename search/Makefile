IMAGE ?= local-search
PORT ?= 8002
QDRANT_URL ?= http://host.docker.internal:6333
QDRANT_COLLECTION_NAME ?= messages
PLATFORM ?= linux/amd64

.PHONY: build run

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) ./

run: build
	docker run --rm -p $(PORT):8000 \
		--add-host=host.docker.internal:host-gateway \
		-e QDRANT_URL=$(QDRANT_URL) \
		-e QDRANT_COLLECTION_NAME=$(QDRANT_COLLECTION_NAME) \
		$(IMAGE)
