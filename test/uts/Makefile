include $(HCLIB_ROOT)/include/hclib.mak

TARGET=UTS

.PHONY: all clean

all: clean $(TARGET)

FLAGS := -Wno-write-strings -DBRG_RNG

SOURCES := uts_main.cpp uts.c rng/brg_sha1.c

OBJECTS := $(SOURCES:.c=.o)
OBJECTS := $(OBJECTS:.cpp=.o)
OBJECTS := $(patsubst %.o,build/%.o,$(OBJECTS))

$(TARGET): $(OBJECTS)
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

build/%.o: %.c
	@mkdir -p $(@D)
	$(CC) -c $(FLAGS) $< -o $@

build/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(FLAGS) $(PROJECT_CXXFLAGS) $< -o $@

clean:
	rm -rf $(TARGET) build
