RED=\033[0;31m
NC=\033[0m
GREEN='\033[0;32m'
YELLOW='\033[1;33m'

all:
	nvcc ./Task_1/task1.cu -o ./Task_1/task1
	nvcc ./Task_2/task2.cu -o ./Task_2/task2
	nvcc ./Task_3/task3.cu -o ./Task_3/task3

build_task1:
	nvcc ./Task_1/task1.cu -o ./Task_1/task1

run_task1: build_task1
	@echo "********************************"
	@echo "\033[1;33mRunning - Task1 \033[0m"
	./Task_1/task1
	@echo "\033[92mDone\033[0m"
	@echo "********************************\n\n"

build_task2: 
	nvcc ./Task_2/task2.cu -o ./Task_2/task2

run_task2: build_task2
	@echo "********************************"
	@echo "\033[1;33mRunning - Task2 \033[0m"
	./Task_2/task2
	@echo "\033[92mDone\033[0m"
	@echo "********************************\n\n"

build_task3:
	nvcc ./Task_3/task3.cu -o ./Task_3/task3

run_task3: build_task3
	@echo "********************************"
	@echo "\033[1;33mRunning - Task3 \033[0m"
	./Task_3/task3
	@echo "\033[92mDone\033[0m"
	@echo "********************************\n\n"
	
