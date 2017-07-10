COMMON_SOURCES := getoptions.c
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

