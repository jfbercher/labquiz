PKG_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

#make release v=0.3.9
#make build
#make clean

.PHONY: build clean version

build: 
	cd $(PKG_DIR) && python -m build

version:
ifndef v
	$(error Utilisation: make version v=X.Y.Z)
endif
	cd $(PKG_DIR) && python bump_version.py $(v)

release: clean version build

clean:
	cd $(PKG_DIR) && \
	rm -rf __pycache__  quiz_editor.egg-info
