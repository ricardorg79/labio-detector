
NAME=labio
DEVIMG=$(NAME)-dev

dev: $(DEVIMG)
	docker run --rm -it \
		-e HUID=$(shell id -u) \
		-e HGID=$(shell id -u) \
		-v $(shell pwd):/workspace \
		$(DEVIMG)

$(DEVIMG): .build/$(DEVIMG)

.build/$(DEVIMG): .build .docker/Dockerfile.dev
	docker build -t $(DEVIMG) -f .docker/Dockerfile.dev .
	touch .build/$(DEVIMG)

.build:
	mkdir -p .build

.PHONY clean:
	rm -fr .build
	rm -fr .vim
