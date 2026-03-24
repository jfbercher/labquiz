# ===== CONFIG =====
PKG_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECTS = quiz_nb quiz_editor quiz_dash
# mapping dossier -> nom package
PKG_NAMES = \
    quiz_nb=labquiz \
    quiz_editor=quiz_editor \
    quiz_dash=quiz_dash

META = meta

PYTHON = python
BASE_BRANCH = main
LAST_TAG := $(shell git describe --tags --abbrev=0 2>/dev/null)

# ===== INPUTS =====
# usage:
# make release                → projets modifiés
# make release target=projet1 → ciblé
# make release target=meta    → meta seul
# make release target=projet1 v=0.9

# ===== UTILS =====

define has_changed_previous
git diff --name-only $(BASE_BRANCH)...HEAD | grep -q "^$(1)/"
endef

define has_changed_previous_tag
if [ -n "$(LAST_TAG)" ]; then \
	echo "LAST_TAG defined. Comparing $(1) to $(LAST_TAG)"; \
	test -n "$$(git diff --name-only $(LAST_TAG)..HEAD -- $(1))"; \
else \
	echo "Comparing $(1) to $(BASE_BRANCH)"; \
	test -n "$$(git ls-files $(1))"; \
fi
endef

define has_changed
LAST_COMMIT_FILE=.${1}_last_release_commit; \
if [ -f $$LAST_COMMIT_FILE ]; then \
	test -n "$$(git diff --name-only $$(cat $$LAST_COMMIT_FILE)..HEAD -- $(1))"; \
else \
	echo "Comparing $(1) to $(BASE_BRANCH)"; \
	test -n "$$(git ls-files $(1))"; \
fi
endef

define get_pkg_name
$(word 2,$(subst =, ,$(filter $(1)=%,$(PKG_NAMES))))
endef

define get_version
grep '^version' $(1)/pyproject.toml | head -1 | cut -d '"' -f2
endef

define bump_patch
v=$$($(call get_version,$(1))); \
IFS='.' read -r major minor patch <<< "$$v"; \
echo "$$major.$$minor.$$((patch+1))"
endef

# ===== TARGET RESOLUTION =====

resolve_targets:
	@targets=""
	@if [ -n "$(target)" ]; then \
		targets="$(target)"; \
	else \
		for p in $(PROJECTS); do \
			echo "Checking $$p for modifications"; \
			if $(call has_changed,$$p); then \
				echo "Found changes in $$p"; \
				targets="$$targets $$p"; \
			fi; \
		done; \
	fi; \
	if [ -z "$$targets" ]; then \
		echo "⚠️ No targets to release"; exit 1; \
	else \
		echo "Targets to release: $$targets"; \
	fi; \
	echo "$$targets" > .targets


# ===== VERSION =====

version:
	@for p in $$(cat .targets); do \
		old=$$( $(call get_version,$$p) ); \
		if [ -n "$(v)" ]; then \
			new="$(v).0"; \
			echo "Set $$p: $$old → $$new"; \
		else \
			new=$$( \
				v=$$old; \
				IFS='.' read -r a b c <<< "$$v"; \
				echo "$$a.$$b.$$((c+1))" \
			); \
			echo "Bump $$p: $$old → $$new"; \
		fi; \
		sed -i "" "s/^version = .*/version = \"$$new\"/" $$p/pyproject.toml; \
	done

# ===== BUILD / PUBLISH =====

build:
	@for p in $$(cat .targets); do \
		# echo "Building $$p"; \
		# cd $$p && $(PYTHON) -m build && cd -; \
		echo "Releasing $$p"; \
		$(MAKE) -C $$p release_simple; \
		echo "$$(git rev-parse HEAD)" > .${p}_last_release_commit; \
	done

publish:
	@for p in $$(cat .targets); do \
		echo "Publishing $$p"; \
		cd $(PKG_DIR)/$$p && twine upload dist/* --verbose && cd $(PKG_DIR); \
	done

# ===== META =====

update-meta_old:
	@if [ "$(target)" != "meta" ]; then \
		echo "Updating meta dependencies"; \
		for p in $(PROJECTS); do \
			v=$$( $(call get_version,$$p) ); \
			sed -i "" "s|\"$$p==.*\"|\"$$p==$$v\"|g" $(META)/pyproject.toml; \
		done; \
	fi

update-meta:
	@if [ "$(target)" != "meta" ]; then \
		echo "Updating meta dependencies"; \
		$(foreach p,$(PROJECTS), \
			name="$(call get_pkg_name,$(p))"; \
			[ -z "$$name" ] && name="$(p)"; \
			v=$$( $(call get_version,$(p)) ); \
			sed -i "" "s|\"$$name==[^\"]*\"|\"$$name==$$v\"|g" $(META)/pyproject.toml; \
		) \
	fi

bump-meta:
	@if [ "$(target)" != "meta" ]; then \
		old=$$( $(call get_version,$(META)) ); \
		IFS='.' read -r a b c <<< "$$old"; \
		new="$$a.$$b.$$((c+1))"; \
		echo "Meta bump: $$old → $$new"; \
		sed -i "" "s/^version = .*/version = \"$$new\"/" $(META)/pyproject.toml; \
	fi

build-meta:
	@if [ "$(target)" != "meta" ]; then \
		cd $(META) && $(PYTHON) -m build && cd -; \
	fi

publish-meta:
	@if [ "$(target)" != "meta" ]; then \
		cd $(META) && twine upload dist/* && cd -; \
	fi

# ===== MAIN =====

release: resolve_targets version build publish update-meta bump-meta build-meta publish-meta tag
	@rm -f .targets
	@echo "✅ Release complete"

release-dry: resolve_targets
	$(eval TARGETS_CONTENT := $(shell cat .targets))
	@echo "Would release: $(TARGETS_CONTENT)"
	@rm -f .targets


# ===== TAG =====
tag:
	@echo "Tagging v$$new"; \
	git tag v$$new; \
	git push origin v$$new

tag_old:
	@v=$$(grep '^version' $(META)/pyproject.toml | cut -d '"' -f2); \
	echo "Tagging v$$v"; \
	git tag v$$v; \
	git push origin v$$v

# ===== CLEAN =====

clean:
	cd $(PKG_DIR) && \
	rm -rf */dist */build */*.egg-info .targets

.PHONY: release resolve_targets version build publish update-meta bump-meta build-meta publish-meta tag clean 