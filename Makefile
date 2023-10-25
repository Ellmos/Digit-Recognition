CXX = g++
CXXFLAGS = -Iinclude -Wall -Wextra -Wvla -MMD -g
LDLIBS = -lm
LDFLAGS = -Llib

SRC_DIR = src
BUILD_DIR = build

BIN = digit_recognition

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEP = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.d)


.PHONY: all run main clean

all: main

run: main
	./$(BIN)

main: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(BIN) $^ $(LDLIBS)

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN)

-include $(DEP)
