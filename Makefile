PKG_DIR='/Users/bercherj/JFB/dev/labquizdev' # WAS #'/Users/bercherj/JFB/dev/LabQuizPkg'
SRC_DIR=src/labquiz
DEV_DIR='/Users/bercherj/JFB/dev/labquizdev/src/labquiz'

#make release v=0.3.9
#make build
#make clean

.PHONY: build clean version hash

build: 
	cd $(PKG_DIR) && python -m build

version:
ifndef v
	$(error Utilisation: make version v=X.Y.Z)
endif
	cd $(PKG_DIR) && python bump_version.py $(v)

hash:
	cd $(PKG_DIR) && python hash_package.py -o hashes.txt

release: clean version hash build

prepare:
	cd $(PKG_DIR) && \
	mkdir -p $(SRC_DIR) && \
	cp $(DEV_DIR)/main.py \
	   $(DEV_DIR)/utils.py \
	   $(DEV_DIR)/putils.py \
	   $(DEV_DIR)/__init__.py \
	   $(SRC_DIR)/

clean:
	cd $(PKG_DIR) && \
	rm -rf __pycache__  labquiz.egg-info
