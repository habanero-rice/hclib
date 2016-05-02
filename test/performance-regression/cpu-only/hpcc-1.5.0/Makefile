# -*- Makefile -*-

arch = UNKNOWN
include hpl/Make.$(arch)

all:
	- $(MKDIR) hpl/lib/$(arch)
	( $(CD) hpl/lib/arch/build ; $(MAKE) arch=$(arch) -f Makefile.hpcc )

clean:
	- $(MKDIR) hpl/lib/$(arch)
	( $(CD) hpl/lib/arch/build ; $(MAKE) arch=$(arch) -f Makefile.hpcc clean )

readme: README.html README.txt

README.html: README.tex
	hevea -fix -O README.tex
	python tools/readme.py README.html

README.info: README.tex
	hevea -fix -info README.tex

README.txt: README.tex
	hevea -fix -text README.tex

.PHONY: all clean readme
