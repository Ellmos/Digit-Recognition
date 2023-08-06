CXX = g++
CXXFLAGS = -Wall -Wextra -g
LDLIBS = -lm
LDFLAGS = -fsanitize=address

SRC = main.cpp neural.cpp layer.cpp hyperParameters.cpp activationFunctions.cpp costFunctions.cpp data/dataLoader.cpp  data/data.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(SRC:.cpp=.d)

main: $(OBJ)
	$(CXX) $(LDFLAGS) -o main $(OBJ) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

.PHONY: clean

clean:
	$(RM) $(OBJ) $(DEP) main

-include $(DEP)


