#include <cstdlib>  // For aligned_alloc and free on some systems
#include <list>
#include <utility>  // For std::pair
#include <algorithm> // For std::max

// Define cache line size (example: 64 bytes for modern CPUs)
#ifndef PBRT_L1_CACHE_LINE_SIZE
#define PBRT_L1_CACHE_LINE_SIZE 64
#endif

class MemoryArena {
private:
    // Memory arena's block size (default: 262144 bytes or 256KB).
    const size_t blockSize;

    // Position within the current memory block.
    size_t currentBlockPos = 0;

    // Size of the current memory block.
    size_t currentAllocSize = 0;

    // Pointer to the current memory block.
    uint8_t* currentBlock = nullptr;

    // List of memory blocks that have been used.
    std::list<std::pair<size_t, uint8_t*>> usedBlocks;

    // List of memory blocks that are available for reuse.
    std::list<std::pair<size_t, uint8_t*>> availableBlocks;

    // Aligned memory allocation
    void* AllocAligned(size_t size) {
        return _aligned_malloc(size, PBRT_L1_CACHE_LINE_SIZE);
    }

    // Aligned memory deallocation
    void FreeAligned(void* ptr) {
        if (ptr) {
            _aligned_free(ptr);
        }
    }

public:
    // Constructor to initialize the memory arena with a default block size.
    MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) { }

    // Destructor to free all allocated memory blocks.
    ~MemoryArena() {
        FreeAligned(currentBlock);  // Free the current memory block.
        // Free all used memory blocks.
        for (auto& block : usedBlocks)
            FreeAligned(block.second);
        // Free all available memory blocks.
        for (auto& block : availableBlocks)
            FreeAligned(block.second);
    }

    // Allocates nBytes of memory, returning a pointer to the allocated memory.
    void* Alloc(size_t nBytes) {
        // Round up nBytes to the machine's minimum alignment.
        nBytes = (nBytes + 7) & (~7); 

        // If the current block does not have enough space, get a new memory block.
        if (currentBlockPos + nBytes > currentAllocSize) {
            // Move the current block to the usedBlocks list if valid.
            if (currentBlock) {
                usedBlocks.push_back(std::make_pair(currentAllocSize, currentBlock));
            }

            // Try to get a new block from availableBlocks or allocate a new one.
            for (auto it = availableBlocks.begin(); it != availableBlocks.end(); ++it) {
                if (it->first >= nBytes) {
                    currentAllocSize = it->first;
                    currentBlock = it->second;
                    availableBlocks.erase(it);
                    break;
                }
            }

            // If no suitable available block, allocate a new block.
            if (!currentBlock) {
                currentAllocSize = std::max(nBytes, blockSize);
                currentBlock = (uint8_t*)AllocAligned(currentAllocSize);
            }
            currentBlockPos = 0;
        }

        // Allocate the memory from the current block.
        void* ret = currentBlock + currentBlockPos;
        currentBlockPos += nBytes;
        return ret;
    }

    // Template method to allocate memory for `n` objects of type `T`.
    template<typename T>
    T* Alloc(size_t n = 1, bool runConstructor = true) {
        // Allocate memory for n objects.
        T* ret = (T*)Alloc(n * sizeof(T));

        // Optionally run the constructor for each object.
        if (runConstructor)
            for (size_t i = 0; i < n; ++i)
                new (&ret[i]) T();
        return ret;
    }

    // Resets the memory arena by reusing all the used blocks.
    void Reset() {
        currentBlockPos = 0;  // Reset the position in the current block.
        // Move all used blocks to the availableBlocks list.
        availableBlocks.splice(availableBlocks.begin(), usedBlocks);
    }

    // Returns the total amount of allocated memory (current + used + available blocks).
    size_t TotalAllocated() const {
        size_t total = currentAllocSize;
        // Add the size of all used blocks.
        for (const auto& alloc : usedBlocks)
            total += alloc.first;
        // Add the size of all available blocks.
        for (const auto& alloc : availableBlocks)
            total += alloc.first;
        return total;
    }

};
