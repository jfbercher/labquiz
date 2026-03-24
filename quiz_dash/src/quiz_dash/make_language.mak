#LANG = es
#LANG = fr
#LANG = de
# pass the language in the command line
# make -f make_language.mak extract LANG=es

PKG = quiz_dash
#MAINDIR = /Users/bercherj/JFB/dev/quiz_dash
MAINDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
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