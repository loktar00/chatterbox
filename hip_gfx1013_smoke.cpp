#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void add_one(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] += 1.0f;
  }
}

static void check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

int main() {
  int device_count = 0;
  check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  std::printf("device_count=%d\n", device_count);
  if (device_count < 1) {
    return 2;
  }

  hipDeviceProp_t prop{};
  check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::printf("device_name=%s gcnArchName=%s\n", prop.name, prop.gcnArchName);

  constexpr int n = 16;
  float host[n] = {};
  for (int i = 0; i < n; ++i) {
    host[i] = static_cast<float>(i);
  }

  float* device = nullptr;
  check(hipMalloc(&device, sizeof(host)), "hipMalloc");
  check(hipMemcpy(device, host, sizeof(host), hipMemcpyHostToDevice), "hipMemcpy H2D");
  add_one<<<1, 32>>>(device, n);
  check(hipGetLastError(), "kernel launch");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
  check(hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost), "hipMemcpy D2H");
  check(hipFree(device), "hipFree");

  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i + 1);
    if (host[i] != expected) {
      std::fprintf(stderr, "bad result at %d: got %f expected %f\n", i, host[i], expected);
      return 3;
    }
  }

  std::puts("native_gfx1013_hip_smoke=ok");
  return 0;
}
