// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include <algorithm>

#include <stdio.h>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

bool g_verbose = false; // Whether to display input/output to console
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Simple key-value pairing for floating point types.
 * Treats positive and negative zero as equivalent.
 */
struct Pair
{
  float key;
  int value;

  bool operator<(const Pair& b) const
  {
    return key < b.key;
  }
};

/**
 * Initialize key-value sorting problem.
 */
void Initialize(float* h_keys, int* h_values, float* h_reference_keys, int* h_reference_values, int num_items)
{
  Pair* h_pairs = new Pair[num_items];

  for (int i = 0; i < num_items; ++i)
  {
    h_pairs[i].key = static_cast<float>(rand()) / RAND_MAX;
    h_pairs[i].value = i;
    h_keys[i] = h_pairs[i].key;
    h_values[i] = h_pairs[i].value;
  }

  std::stable_sort(h_pairs, h_pairs + num_items);

  for (int i = 0; i < num_items; ++i)
  {
    h_reference_keys[i]   = h_pairs[i].key;
    h_reference_values[i] = h_pairs[i].value;
  }

  delete[] h_pairs;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */

// Helper function to compare device results
template <typename T>
int CompareDeviceResults(T* h_reference, T* d_data, int num_items, bool verbose = true, bool display_data = false)
{
  // Allocate array on host for copying results from device
  T* h_data = new T[num_items];
  cudaMemcpy(h_data, d_data, num_items * sizeof(T), cudaMemcpyDeviceToHost);

  int num_errors = 0;
  for (int i = 0; i < num_items; i++)
  {
    if (h_data[i] != h_reference[i])
    {
      num_errors++;
      if (display_data && (num_errors < 10))
        printf("Data error at %d: reference %f, data %f\n", 
             i, float(h_reference[i]), float(h_data[i]));
    }
  }

  delete[] h_data;
  return num_errors;
}

// Helper function to check for errors
void AssertEquals(int expected, int actual)
{
  if (expected != actual)
  {
    printf("ERROR: Expected %d but got %d\n", expected, actual);
    exit(1);
  }
}

int main(int argc, char** argv)
{
  int num_items = 150;
  // set device id to 0
  cudaSetDevice(0);

  printf("cub::DeviceRadixSort::SortPairs() %d items (%d-byte keys %d-byte values)\n",
         num_items,
         int(sizeof(float)),
         int(sizeof(int)));
  fflush(stdout);

  // Allocate host arrays
  float* h_keys           = new float[num_items];
  float* h_reference_keys = new float[num_items];
  int* h_values           = new int[num_items];
  int* h_reference_values = new int[num_items];

  // Initialize problem and solution on host
  Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items);

  // Allocate device arrays
  DoubleBuffer<float> d_keys;
  DoubleBuffer<int> d_values;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_values.d_buffers[0], sizeof(int) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_values.d_buffers[1], sizeof(int) * num_items));

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void* d_temp_storage      = nullptr;

  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Initialize device arrays
  CubDebugExit(
    cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
  CubDebugExit(
    cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));

  // Run
  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

  // Check for correctness (and display results, if specified)
  int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
  printf("\t Compare keys (selector %d): %s\n", d_keys.selector, compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);
  compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
  printf("\t Compare values (selector %d): %s\n", d_values.selector, compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  

  // Cleanup
  if (h_keys)
  {
    delete[] h_keys;
  }
  if (h_reference_keys)
  {
    delete[] h_reference_keys;
  }
  if (h_values)
  {
    delete[] h_values;
  }
  if (h_reference_values)
  {
    delete[] h_reference_values;
  }

  if (d_keys.d_buffers[0])
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
  }
  if (d_keys.d_buffers[1])
  {
    CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
  }
  if (d_values.d_buffers[0])
  {
    CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
  }
  if (d_values.d_buffers[1])
  {
    CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
  }
  if (d_temp_storage)
  {
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  }

  printf("\n\n");

  return 0;
}