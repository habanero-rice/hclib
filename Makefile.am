
SUBDIRS = src

AUX_DIST = \
	$(ac_aux_dir)/config.guess		\
	$(ac_aux_dir)/config.sub		\
	$(ac_aux_dir)/install-sh		\
	$(ac_aux_dir)/ltmain.sh			\
	$(ac_aux_dir)/missing			\
	$(ac_aux_dir)/depcomp

library_includedir=$(includedir)

if WITH_OCR
library_include_HEADERS = inc/ocr-rt.h inc/hcpp.h inc/ocr-async.h inc/ocr-mak/hcpp.mak inc/ocr-ddf.h inc/hcpp-forasync.h inc/hcpp-asyncAwait.h
else
library_include_HEADERS = inc/hcpp-rt.h inc/hcpp.h inc/hcpp-mak/hcpp.mak inc/hcpp-async.h inc/hcpp-hpt.h inc/hcpp-mm.h inc/hcpp-forasync.h inc/hcpp-ddf.h inc/hcpp-asyncAwait.h src/inc/hcpp-atomics.h inc/hcpp-place.h
endif

MAINTAINERCLEANFILES = Makefile.in \
	aclocal.m4 \
	configure  \
	$(AUX_DIST)
