NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17

TARGET = sgemm_naive
SRC = sgemm_naive.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
