CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC
LDFLAGS = -ldl

# Paths
SRCDIR = .
BUILDDIR = build
OCG_LIB_PATH = /home/ecast229/Applications/EDOPro/libocgcore.so

# Source files
SOURCES = ocg_headless_duel.cpp
OBJECTS = $(SOURCES:%.cpp=$(BUILDDIR)/%.o)
TARGET = ocg_headless_duel

.PHONY: all clean test

all: $(BUILDDIR) $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)

test: $(TARGET)
	@echo "Testing OCG Headless Duel with sample decks..."
	@echo "Make sure you have valid .ydk files to test with"
	@echo "Usage: ./$(TARGET) deck1.ydk deck2.ydk"

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	chmod +x /usr/local/bin/$(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Check if OCG library exists
check-ocg:
	@if [ ! -f "$(OCG_LIB_PATH)" ]; then \
		echo "OCG library not found at $(OCG_LIB_PATH)"; \
		echo "Please update OCG_LIB_PATH in Makefile"; \
		exit 1; \
	fi
	@echo "OCG library found at $(OCG_LIB_PATH)"

# Show OCG library symbols
show-symbols:
	@echo "OCG library symbols:"
	@objdump -T $(OCG_LIB_PATH) | grep -E "OCG_|MSG_" | head -20

help:
	@echo "Available targets:"
	@echo "  all       - Build the headless duel simulator"
	@echo "  clean     - Remove build files"
	@echo "  test      - Show test usage"
	@echo "  debug     - Build with debug symbols"
	@echo "  check-ocg - Check if OCG library is available"
	@echo "  show-symbols - Show OCG library symbols"
	@echo "  install   - Install to /usr/local/bin"
	@echo "  help      - Show this help message"
