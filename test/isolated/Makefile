include $(HCLIB_ROOT)/include/hclib.mak

EXE=test1 test2 BankTransactions

all: clean $(EXE) clean-obj

test1: test1.cpp
	g++ $(PROJECT_CXXFLAGS) -o $@ $^ -L . $(PROJECT_LDLIBS) $(PROJECT_LDFLAGS)

test2: test2.cpp
	g++ $(PROJECT_CXXFLAGS) -o $@ $^ -L . $(PROJECT_LDLIBS) $(PROJECT_LDFLAGS)

BankTransactions: BankTransactions.cpp
	g++ $(PROJECT_CXXFLAGS) -o $@ $^ -L . $(PROJECT_LDLIBS) $(PROJECT_LDFLAGS)

clean-obj:
	rm -rf *.o *.dSYM

clean:
	rm -rf *.o $(EXE) *.dSYM
