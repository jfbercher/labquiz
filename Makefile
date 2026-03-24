# ===== CONFIG =====
PROJECTS = labquiz quiz_editor quiz_dash
META = meta

PYTHON = python3
BASE_BRANCH = main

# ===== INPUTS =====
# usage:
# make release                → projets modifiés
# make release target=projet1 → ciblé
# make release target=meta    → meta seul
# make release target=projet1 v=0.9

# ===== UTILS =====

define has_changed
git diff --name-only $(BASE_BRANCH)...HEAD | grep -q "^$(1)/"
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
		sed -i "s/^version = .*/version = \"$$new\"/" $$p/pyproject.toml; \
	done

# ===== BUILD / PUBLISH =====

build:
	@for p in $$(cat .targets); do \
		# echo "Building $$p"; \
		# cd $$p && $(PYTHON) -m build && cd -; \
		echo "Releasing $$p"; \
		$(MAKE) -C $$p release_simple; \
	done

publish:
	@for p in $$(cat .targets); do \
		echo "Publishing $$p"; \
		cd $$p && twine upload dist/* && cd -; \
	done

# ===== META =====

update-meta:
	@if [ "$(target)" != "meta" ]; then \
		echo "Updating meta dependencies"; \
		for p in $(PROJECTS); do \
			v=$$( $(call get_version,$$p) ); \
			sed -i "s|\"$$p==.*\"|\"$$p==$$v\"|g" $(META)/pyproject.toml; \
		done; \
	fi

bump-meta:
	@if [ "$(target)" != "meta" ]; then \
		old=$$( $(call get_version,$(META)) ); \
		IFS='.' read -r a b c <<< "$$old"; \
		new="$$a.$$b.$$((c+1))"; \
		echo "Meta bump: $$old → $$new"; \
		sed -i "s/^version = .*/version = \"$$new\"/" $(META)/pyproject.toml; \
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

release: resolve_targets version build publish update-meta bump-meta build-meta publish-meta
	@rm -f .targets
	@echo "✅ Release complete"

release-dry: resolve_targets
	@echo "Would release:" $(cat .targets)
	@rm -f .targets


# ===== CLEAN =====

clean:
	rm -rf */dist */build */*.egg-info .targets

.PHONY: release resolve_targets version build publish update-meta bump-meta build-meta publish-meta clean