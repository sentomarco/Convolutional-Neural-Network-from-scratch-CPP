CNN: main.o CNN.o Filters.o Datasets.o Volumes.o MLP.o
	g++ main.o CNN.o Filters.o Datasets.o Volumes.o MLP.o -o CNN
	clear
	./CNN

main.o: main.cpp CNN.cpp CNN.h
	g++ -c -Wall main.cpp CNN.h CNN.cpp 

CNN.o: CNN.h CNN.cpp Volumes.h Volumes.cpp Filters.h Filters.cpp Datasets.h Datasets.cpp MLP.cpp MLP.h
	g++ -c -Wall CNN.h CNN.cpp Volumes.h Volumes.cpp Filters.h Filters.cpp Datasets.h Datasets.cpp MLP.h MLP.cpp 

Filters.o: Filters.h Filters.cpp Volumes.h Volumes.cpp
	g++ -c -Wall Volumes.h Volumes.cpp Filters.h Filters.cpp

Datasets.o: Datasets.cpp Datasets.h Volumes.h Volumes.cpp
	g++ -c -Wall Datasets.h Datasets.cpp Volumes.h Volumes.cpp

Volumes.o: Volumes.cpp Volumes.h
	g++ -c -Wall Volumes.h Volumes.cpp

MLP.o: MLP.h MLP.cpp
	g++ -c -Wall MLP.h MLP.cpp
