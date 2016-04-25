# Post Makefile includes are the main part of a module's build system, allowing
# it to add flags to the overall project compile and link flags.
PROJECT_CFLAGS+=-I$(HCLIB_ROOT)/../modules/openshmem/inc -I$(OPENSHMEM_INSTALL)/include
PROJECT_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/openshmem/inc -I$(OPENSHMEM_INSTALL)/include
PROJECT_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/openshmem/lib -L$(OPENSHMEM_INSTALL)/lib
PROJECT_LDLIBS += $(call GET_LINK_FLAG,--whole-archive) \
				  $(call GET_LINK_FLAG,-lhclib_openshmem) \
				  $(call GET_LINK_FLAG,--no-whole-archive) \
				  $(call GET_LINK_FLAG,-lopenshmem) \
				  $(call GET_LINK_FLAG,-lrt) \
				  $(call GET_LINK_FLAG,-lmpi) \
				  $(call GET_LINK_FLAG,-lelf) \
				  $(call GET_LINK_FLAG,/scratch/jmg3/gasnet-ibv-install/lib/libgasnet-ibv-par.a) \
				  $(call GET_LINK_FLAG,/scratch/jmg3/gasnet-ibv-install/lib/libammpi.a) \
				  $(call GET_LINK_FLAG,/usr/lib64/libibverbs.so) \
				  $(call GET_LINK_FLAG,/usr/lib64/libpmi2.so)
