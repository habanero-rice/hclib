BASE_TARGET := uts_main

CFLAGS   := -Wno-write-strings -DBRG_RNG
CXXFLAGS := $(CFLAGS)

COMMON_SOURCES := uts.c rng/brg_sha1.c
OBJECTS := $(COMMON_SOURCES:.c=.o)
OBJECTS := $(OBJECTS:.cpp=.o)
OBJECTS := $(patsubst %.o,build/%.o,$(OBJECTS))

PROJECT_EXTRA_DEPS := build
LDFLAGS := $(OBJECTS)

include ../common/common.mk

build: $(OBJECTS)

build/%.o: %.c
	@mkdir -p $(@D)
	@echo Build support file $@ from $<
	$(CC) -c $(CFLAGS) $< -o $@


