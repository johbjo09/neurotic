CC := c++ -std=c++11
SRCDIR := src
BUILDDIR := build
TARGET := bin/runner
TEST_BINARY := bin/test

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -Wall
LIB := -L lib -lpthread
INC := -I include

$(TARGET): $(OBJECTS)
	@mkdir -p bin
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(TEST_BINARY): build/util.o test/tester.cpp
	$(CC) build/util.o $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o $(TEST_BINARY)

build: $(TARGET)

test: $(TEST_BINARY)
	@./$(TEST_BINARY)

run: $(TARGET)
	@./$(TARGET)

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: build test run clean
