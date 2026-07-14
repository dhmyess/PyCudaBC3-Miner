// sha3_gpu_miner.cu
// Optimized GPU miner for triple-SHA3-256 (BitcoinIII)
// Build: nvcc -O3 -arch=sm_61 -shared -Xcompiler -fPIC -o libsha3miner.so sha3_gpu_miner.cu

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Constants for Keccak-f
__device__ __constant__ uint64_t d_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int y) {
    return (x << y) | (x >> (64 - y));
}

// Keccak-f with hardcoded rotations (no array lookups)
__device__ __forceinline__ void keccakf(uint64_t st[25]) {
    uint64_t t, bc[5];
    for (int round = 0; round < 24; round++) {
        // Theta
        #pragma unroll
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5) st[j+i] ^= t;
        }

        // Rho and Pi (hardcoded sequence)
        t = st[1];
        st[1] = rotl64(st[6], 44);
        st[6] = rotl64(st[9], 20);
        st[9] = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2] = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4] = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8] = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5] = rotl64(st[3], 28);
        st[3] = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7] = rotl64(st[10], 3);
        st[10] = rotl64(t, 1);

        // Chi
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            uint64_t a0 = st[j], a1 = st[j+1], a2 = st[j+2], a3 = st[j+3], a4 = st[j+4];
            st[j]   = a0 ^ ((~a1) & a2);
            st[j+1] = a1 ^ ((~a2) & a3);
            st[j+2] = a2 ^ ((~a3) & a4);
            st[j+3] = a3 ^ ((~a4) & a0);
            st[j+4] = a4 ^ ((~a0) & a1);
        }

        // Iota
        st[0] ^= d_rndc[round];
    }
}

// Single-block SHA3-256 (input < 136 bytes)
__device__ __forceinline__ void sha3_256_single(const uint8_t *in, int inlen, uint8_t out[32]) {
    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0ULL;
    uint8_t *b = (uint8_t*)st;
    for (int i = 0; i < inlen; i++) b[i] = in[i];
    b[inlen] ^= 0x06;          // SHA3-256 domain separator
    b[135] ^= 0x80;            // final bit
    keccakf(st);
    #pragma unroll
    for (int i = 0; i < 32; i++) out[i] = b[i];
}

// Kernel: each thread hashes one nonce
extern "C" __global__ __launch_bounds__(512) void scan_kernel(
    const uint8_t* __restrict__ header76,
    uint32_t start_nonce,
    const uint8_t* __restrict__ target32,
    uint32_t* found_nonces,
    unsigned int* found_count,
    unsigned int max_found)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;

    // Build 80-byte header (76 fixed + 4 nonce bytes little-endian)
    uint8_t header[80];
    #pragma unroll
    for (int i = 0; i < 76; i++) header[i] = header76[i];
    header[76] = (uint8_t)(nonce & 0xFF);
    header[77] = (uint8_t)((nonce >> 8) & 0xFF);
    header[78] = (uint8_t)((nonce >> 16) & 0xFF);
    header[79] = (uint8_t)((nonce >> 24) & 0xFF);

    // Triple SHA3-256
    uint8_t h1[32], h2[32], h3[32];
    sha3_256_single(header, 80, h1);
    sha3_256_single(h1, 32, h2);
    sha3_256_single(h2, 32, h3);

    // Reverse bytes to get blockhash (big-endian)
    uint8_t bh[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) bh[i] = h3[31 - i];

    // Compare bh <= target32 (both big-endian)
    bool le = true;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (bh[i] < target32[i]) { break; }
        if (bh[i] > target32[i]) { le = false; break; }
    }

    if (le) {
        unsigned int pos = atomicAdd(found_count, 1u);
        if (pos < max_found) {
            found_nonces[pos] = nonce;
        }
    }
}

// Host function: set launch config and call kernel
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return -1; \
        } \
    } while (0)

extern "C" int scan_batch(
    const uint8_t* header76,
    uint32_t start_nonce,
    const uint8_t* target32,
    uint32_t* out_nonces,
    uint32_t max_found)
{
    uint8_t *d_header76 = nullptr, *d_target32 = nullptr;
    uint32_t *d_found_nonces = nullptr;
    unsigned int *d_found_count = nullptr;
    unsigned int h_found_count = 0;

    const int threadsPerBlock = 512;
    const int blocksPerGrid = 32768;  // 256 * 32768 = 8,388,608 nonces per batch

    CUDA_CHECK(cudaMalloc(&d_header76, 76));
    CUDA_CHECK(cudaMalloc(&d_target32, 32));
    CUDA_CHECK(cudaMalloc(&d_found_nonces, max_found * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_header76, header76, 76, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target32, target32, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(unsigned int)));

    scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_header76, start_nonce, d_target32, d_found_nonces, d_found_count, max_found);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_found_count, d_found_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int n_copy = (h_found_count < max_found) ? h_found_count : max_found;
    if (n_copy > 0) {
        CUDA_CHECK(cudaMemcpy(out_nonces, d_found_nonces, n_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_header76);
    cudaFree(d_target32);
    cudaFree(d_found_nonces);
    cudaFree(d_found_count);

    return (int)h_found_count;
}
