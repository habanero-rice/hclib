include $(HCLIB_ROOT)/include/hclib.mak

TARGET=cholesky

.PHONY: all clean

all: clean $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) -o $@ $< $(PROJECT_LDLIBS)

clean:
	rm -rf $(TARGET) *.out
