#LANG = es
#LANG = fr
#LANG = de
# pass the language in the command line
# make -f make_language.mak extract LANG=es

PKG = labquiz
MAINDIR = /Users/bercherj/JFB/dev/labquizdev/src/labquiz
ALL_LOCALES := $(wildcard $(MAINDIR)/locales/*)
DIR_LOCALES := $(foreach f,$(ALL_LOCALES),\
                $(if $(wildcard $(f)/*),$(f)))
LANGUAGES := $(notdir $(DIR_LOCALES))



.PHONY: update_all $(LANGUAGES)

print-debug:
	@echo MAINDIR=$(MAINDIR)
	@echo WILDCARD=$(wildcard $(MAINDIR)/locales/*)
	@echo DIR_LOCALES=$(DIR_LOCALES)
	@echo LANGUAGES=$(LANGUAGES)

print-langs:
	@echo $(LANGUAGES)

extract:
	xgettext -o $(MAINDIR)/locales/$(PKG).pot \
		--language=Python \
		--keyword=_ \
		$$(find $(MAINDIR) -name "*.py")

create:
	mkdir -p $(MAINDIR)/locales/$(LANG)/LC_MESSAGES

	msginit \
	--locale=$(LANG) \
	--input=$(MAINDIR)/locales/$(PKG).pot \
	--output-file=$(MAINDIR)/locales/$(LANG)/LC_MESSAGES/$(PKG).po

update:
	msgmerge --update $(MAINDIR)/locales/$(LANG)/LC_MESSAGES/$(PKG).po \
	         $(MAINDIR)/locales/$(PKG).pot

compile:
	msgfmt $(MAINDIR)/locales/$(LANG)/LC_MESSAGES/$(PKG).po \
	       -o $(MAINDIR)/locales/$(LANG)/LC_MESSAGES/$(PKG).mo

.PHONY: update_all compile_all $(LANGUAGES)

# --- Partie Update ---
update_all: $(addprefix update-,$(LANGUAGES))

update-%:
	$(MAKE) -f make_language.mak update LANG=$*

# --- Partie Compile ---
compile_all: $(addprefix compile-,$(LANGUAGES))

compile-%:
	$(MAKE) -f make_language.mak compile LANG=$*