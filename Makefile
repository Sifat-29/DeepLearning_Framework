# ==========================================
#          Compiler & Flags
# ==========================================
# The compiler to use
CC = gcc

# Compiler Flags:
# -Wall -Wextra: Enable all warnings (helps find bugs)
# -O3: Maximum optimisation (makes training fast)
# -march=native: Optimises code specifically for your CPU
# -Iinclude: Look for header files (.h) in the 'include' folder
# -fPIC: Position Independent Code (Required for Shared Libraries)
CFLAGS = -Wall -Wextra -O3 -march=native -Iinclude -fPIC

# Linker Flags:
# -lm: Link the standard Math library (required for sqrt, exp, etc.)
LDFLAGS = -lm

# ==========================================
#          Directory Variables
# ==========================================
SRC_DIR = src
OBJ_DIR = build
LIB_DIR = lib
BIN_DIR = bin
TEST_DIR = tests

# ==========================================
#          Files & Paths
# ==========================================
# 1. Find all .c files in src/ (e.g., src/tensor.c, src/layer.c)
SRCS = $(wildcard $(SRC_DIR)/*.c)

# 2. Convert source paths to object paths (e.g., src/tensor.c -> build/tensor.o)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

TEST_NAME = mnist_test

TEST_SRC = $(TEST_DIR)/$(TEST_NAME).c
TEST_OBJ = $(OBJ_DIR)/$(TEST_NAME).o

# 4. Output Names
LIB_NAME = libneural.so
TARGET_LIB = $(LIB_DIR)/$(LIB_NAME)
TARGET_BIN = $(BIN_DIR)/neural_net

# Installation path
PREFIX = /usr/local

# ==========================================
#          Build Rules
# ==========================================

# Default Target: Run 'make' to do all of this
all: directories $(TARGET_LIB) $(TARGET_BIN)

# Creates the Shared Library (.so)
# Links all library object files into one shared binary
$(TARGET_LIB): $(OBJS)
	@echo "Creating Shared Library: $@"
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

# Linking the Main Executable
# Links the test object file WITH the shared library we just made
# -L$(LIB_DIR): Look for libraries in 'lib/'
# -lneural: Link against 'libneural.so'
# -Wl,-rpath=$(LIB_DIR): Tell executable to look in 'lib/' at runtime (Local usage)
$(TARGET_BIN): $(TEST_OBJ) $(TARGET_LIB)
	@echo "Linking Executable: $@"
	$(CC) $(TEST_OBJ) -o $@ $(LDFLAGS) -L$(LIB_DIR) -lneural -Wl,-rpath=$(LIB_DIR)

# Compiling Library .c files to .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling Library: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Compiling Test .c file to .o
$(OBJ_DIR)/$(TEST_NAME).o: $(TEST_SRC)
	@echo "Compiling Test: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Create the missing directories
directories:
	@mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

# ==========================================
#          Install / Uninstall
# ==========================================

# Installs headers and library to system folders so friends can use #include <neural/network.h>
install: all
	@echo "Installing to $(PREFIX)..."
	@mkdir -p $(PREFIX)/include/neural
	@mkdir -p $(PREFIX)/lib
	@cp include/*.h $(PREFIX)/include/neural/
	@cp $(TARGET_LIB) $(PREFIX)/lib/
	@ldconfig
	@echo "Installation complete."

uninstall:
	@echo "Removing library..."
	@rm -rf $(PREFIX)/include/neural
	@rm -f $(PREFIX)/lib/$(LIB_NAME)
	@ldconfig
	@echo "Uninstalled."

# ==========================================
#          Cleanup
# ==========================================
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean directories install uninstall
