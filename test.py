import ctypes
import hashlib
import os
import sys
import time

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsha3miner.so")

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 32768
BATCH_SIZE = THREADS_PER_BLOCK * BLOCKS_PER_GRID  # 8388608
TOTAL_BATCHES = (0xFFFFFFFF + 1) // BATCH_SIZE     # 512
MAX_FOUND_PER_BATCH = 1024


def rev_hex(hexstr):
    return "".join([hexstr[i:i + 2] for i in range(0, len(hexstr), 2)][::-1])


def triple_sha3_256(header_bytes):
    h1 = hashlib.sha3_256(header_bytes)
    h2 = hashlib.sha3_256(h1.digest())
    h3 = hashlib.sha3_256(h2.digest())
    return h3.hexdigest()


def build_header76(version, previousblockhash, merkleroot, ntime, nbits):
    """Everything except the 4-byte nonce, in on-wire (already byte-reversed) order."""
    hex76 = (
        rev_hex(version)
        + rev_hex(previousblockhash)
        + rev_hex(merkleroot)
        + rev_hex(f"{ntime:08x}")
        + rev_hex(nbits)
    )
    b = bytes.fromhex(hex76)
    assert len(b) == 76, f"expected 76 bytes, got {len(b)}"
    return b


def load_lib():
    if not os.path.exists(LIB_PATH):
        sys.exit(
            f"libsha3miner.so not found at {LIB_PATH}.\n"
            "Build it first on a machine with the CUDA toolkit + NVIDIA GPU:\n"
            "  nvcc -O3 -shared -Xcompiler -fPIC -o libsha3miner.so sha3_gpu_miner.cu"
        )
    lib = ctypes.CDLL(LIB_PATH)
    lib.scan_batch.argtypes = [
        ctypes.c_char_p,                    # header76
        ctypes.c_uint32,                    # start_nonce
        ctypes.c_char_p,                    # target32
        ctypes.POINTER(ctypes.c_uint32),    # out_nonces
        ctypes.c_uint32,                    # max_found
    ]
    lib.scan_batch.restype = ctypes.c_int
    return lib


def mining_nonce(lib, header76, target32, start_nonce, batch_size,
                  max_found=MAX_FOUND_PER_BATCH):
    # threadsPerBlock=512 / blocksPerGrid=16384 are hardcoded in the .cu file;
    # batch_size here must match that (8388608) or the loop math in main() drifts.
    assert batch_size == BATCH_SIZE
    out_buf = (ctypes.c_uint32 * max_found)()
    n_found = lib.scan_batch(
        header76, ctypes.c_uint32(start_nonce), target32, out_buf, max_found,
    )
    if n_found == -1:
        sys.exit(
            "GPU kernel reported a CUDA error (see the CUDA error line printed "
            "above, from stderr of libsha3miner.so). Scan aborted."
        )
    return [out_buf[i] for i in range(min(n_found, max_found))], n_found


def verify_on_cpu(header76, nonce, target32):
    header80 = header76 + bytes.fromhex(rev_hex(f"{nonce:08x}"))
    hasil = triple_sha3_256(header80)
    bh = rev_hex(hasil)
    return bh, bytes.fromhex(bh) <= target32


def selftest():
    """Reproduces the known-good block from hashlib_sha3-256.py to sanity-check
    the header construction before trusting the GPU kernel's output."""
    header76 = build_header76(
        version="20001000",
        previousblockhash="00000000000b12363a67e32ca2d6e25f35649da78af676baf89246c74862a6ba",
        merkleroot="c72460285178bdbdb5db3efc75c42167f57482bd59d161c62cd1d68ba4353b3e",
        ntime=1783975923,
        nbits="1b0fa11e",
    )
    nonce = 705414469
    header80 = header76 + bytes.fromhex(rev_hex(f"{nonce:08x}"))
    hasil = triple_sha3_256(header80)
    bh = rev_hex(hasil)
    expected = "00000000000f87b165ecce8863338593e141f592d5e357daf402dd10aff4bba4"
    print("selftest blockhash:", bh)
    print("expected          :", expected)
    print("MATCH" if bh == expected else "MISMATCH", "\n")
    return bh == expected


def main():
    if "--selftest" in sys.argv:
        ok = selftest()
        sys.exit(0 if ok else 1)

    # --- fill these in for the block you're actually mining ---
    header76 = build_header76(
        version="20001000",
        previousblockhash="00000000000b12363a67e32ca2d6e25f35649da78af676baf89246c74862a6ba",
        merkleroot="c72460285178bdbdb5db3efc75c42167f57482bd59d161c62cd1d68ba4353b3e",
        ntime=1783975923,
        nbits="1b0fa11e",
    )
    target32 = bytes.fromhex("00000000FFFF0000000000000000000000000000000000000000000000000000"[-64:])

    lib = load_lib()

    print(f"Batch size: {BATCH_SIZE} (threadsPerBlock={THREADS_PER_BLOCK} x blocksPerGrid={BLOCKS_PER_GRID})")
    print(f"Total batches: {TOTAL_BATCHES} covering nonce 0 .. 0xFFFFFFFF\n")

    t0 = time.time()
    for i in range(TOTAL_BATCHES):
        start_nonce = i * BATCH_SIZE
        t_batch_start = time.time()
        candidates, n_found = mining_nonce(lib, header76, target32, start_nonce, BATCH_SIZE)
        batch_elapsed = time.time() - t_batch_start
        rate = BATCH_SIZE / batch_elapsed if batch_elapsed > 0 else float("inf")
        print(f"Scanning nonce {start_nonce} to {start_nonce + BATCH_SIZE - 1} "
              f"[{rate/1e6:.2f} MH/s this batch, {batch_elapsed*1000:.1f} ms]")

        for nonce in candidates:
            bh, ok = verify_on_cpu(header76, nonce, target32)
            if ok:
                print(f"  >>> VALID nonce={nonce} blockhash={bh}")
            else:
                print(f"  (false positive on CPU re-check, nonce={nonce}, bh={bh})")

        if n_found > MAX_FOUND_PER_BATCH:
            print(f"  warning: {n_found} hits this batch exceeded max_found buffer "
                  f"({MAX_FOUND_PER_BATCH}); increase MAX_FOUND_PER_BATCH")


if __name__ == "__main__":
    main()
