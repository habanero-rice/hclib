
SUBDIRS = src

AUX_DIST = \
	$(ac_aux_dir)/config.guess		\
	$(ac_aux_dir)/config.sub		\
	$(ac_aux_dir)/install-sh		\
	$(ac_aux_dir)/ltmain.sh			\
	$(ac_aux_dir)/missing			\
	$(ac_aux_dir)/depcomp

library_includedir=$(includedir)

library_include_HEADERS = inc/hcpp-timer.h inc/hcpp-rt.h inc/hcpp.h \
						  inc/hcpp-mak/hcpp.mak inc/hcpp-async.h src/inc/hcpp-hpt.h \
						  inc/hcpp-forasync.h inc/hcpp-ddf.h inc/hcpp-asyncAwait.h \
						  src/inc/hcpp-atomics.h inc/hcpp-place.h inc/hcupc-support.h \
						  inc/hcpp-asyncStruct.h inc/hclib.h inc/hclib_cpp.h \
						  inc/hcpp-task.h inc/hclib_common.h src/inc/litectx.h \
						  src/fcontext/fcontext.h src/inc/hcpp-tree.h src/inc/hcpp-cuda.h

MAINTAINERCLEANFILES = Makefile.in \
	aclocal.m4 \
	configure  \
	$(AUX_DIST)
