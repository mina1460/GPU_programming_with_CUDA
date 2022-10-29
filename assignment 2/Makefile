RED=\033[0;31m
NC=\033[0m
GREEN='\033[0;32m'
YELLOW='\033[1;33m'

all:
	nvcc task1.cu -o task1
	nvcc task2.cu -o task2
	nvcc task3.cu -o task3

build_task1:
	g++ task1.c -o task1

run_task1: build_task1
	@echo "********************************"
	@echo "\033[1;33mRunning - Task1 \033[0m"
	./task1
	@echo "\033[92mDone\033[0m"
	@echo "********************************\n\n"
