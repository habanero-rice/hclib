
SUBDIRS = src

AUX_DIST = \
	$(ac_aux_dir)/config.guess		\
	$(ac_aux_dir)/config.sub		\
	$(ac_aux_dir)/install-sh		\
	$(ac_aux_dir)/ltmain.sh			\
	$(ac_aux_dir)/missing			\
	$(ac_aux_dir)/depcomp

library_includedir=$(includedir)

library_include_HEADERS = inc/hclib-timer.h inc/hclib-rt.h inc/hclib.h \
						  inc/hclib_config.h \
						  inc/hclib-mak/hclib.mak inc/hclib-async.h \
						  inc/hclib-forasync.h inc/hclib-promise.h inc/hclib_promise.h \
						  src/inc/hclib-atomics.h \
						  inc/hclib-async-struct.h inc/hclib_cpp.h inc/hclib_future.h \
						  inc/hclib-task.h inc/hclib_common.h src/inc/litectx.h \
						  src/fcontext/fcontext.h src/inc/hclib-tree.h \
						  inc/hclib-locality-graph.h inc/hclib-module.h src/inc/hclib-fptr-list.h \
						  inc/hclib_atomic.h inc/hclib-instrument.h

MAINTAINERCLEANFILES = Makefile.in \
	aclocal.m4 \
	configure  \
	$(AUX_DIST)
