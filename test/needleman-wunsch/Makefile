include $(HCLIB_ROOT)/include/hclib.mak

TARGET=needleman-wunsch

all: clean $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $^ $(PROJECT_LDLIBS)

clean:
	rm -rf $(TARGET)
