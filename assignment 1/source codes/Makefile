RED=\033[0;31m
NC=\033[0m
GREEN='\033[0;32m'
YELLOW='\033[1;33m'

all:
	nvcc task1.cu -o task1
	nvcc task2.cu -o task2
	nvcc task3.cu -o task3

build_task1:
	nvcc task1.cu -o task1

build_task2:
	nvcc task2.cu -o task2

build_task3:
	nvcc task3.cu -o task3

clean:
	rm -f task1 task2 task3

run_task1: build_task1
	@echo "********************************"
	@echo "\033[1;33mRunning - Task1 \033[0m"
	./task1
	@echo "\033[92mDone\033[0m"
	@echo "********************************\n\n"


run_task2: build_task2
	@echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
	@echo "\033[1;33mRunning - Task2 \033[0m"
	./task2
	@echo "\033[92mDone\033[0m"
	@echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n"

run_task3: build_task3	
	@echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
	@echo "\033[1;33mRunning - Task3 \033[0m"
	./task3
	@echo "\033[92mDone\033[0m"
	@echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n\n"

run: run_task1 run_task2 run_task3

