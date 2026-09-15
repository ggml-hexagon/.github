# FastRPC‑based ggml‑hexagon backend
> Alternative co‑existing backend for Qualcomm Hexagon NPU (Android / WoS(Windows on Snapdragon) / Linux)

 
## Overview
Qualcomm Hexagon SDK exposes two distinct RPC transport mechanisms: native FastRPC and dspqueue.

[Project ggml‑hexagon](https://github.com/ggml-hexagon/ggml-hexagon) introduces FastRPC‑based ggml‑hexagon backend, which co‑exists alongside the existing dspqueue‑based ggml‑hexagon implementation.

This dual‑backend pattern mirrors existing patterns inside llama.cpp:
- ggml‑openvino vs ggml‑sycl (multiple backends targeting same hardware)
- ggml‑cuda vs ggml‑hip (re‑use of existing device kernels across host layers)

Both backends share the same Hexagon/HTP kernel source tree; users select transport layer via build‑time CMake option, zero breaking changes to upstream default dspqueue workflow.The real performance difference does not lie in the Hexagon/HTP kernels themselves, but in the scheduling framework, cache policy and offloading strategy
(See details at [The mempool/FastRPC and dspqueue ggml-hexagon variants: Architecture Analysis](https://github.com/ggml-hexagon/ggml-hexagon/blob/self-build-jz/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)).

FastRPC‑based ggml‑hexagon originated from upstream [PR #12326](https://github.com/ggml-org/llama.cpp/pull/12326), with follow‑up upstream PRs: [PR #26373](https://github.com/ggml-org/llama.cpp/pull/26373), [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642).


## Key Features

- PP and TG offer [some advantages](https://github.com/ggml-hexagon/ggml-hexagon/discussions/71) over Qualcomm's dspqueue‑based ggml‑hexagon for certain modern models.
  
- Single shared mempool + Native FastRPC transport

  Enables **effective** lm‑head offloading, which is the primary source of PP / TG throughput improvements observed in benchmarks. The optimization depends directly on the shared mempool architecture.

- NPU‑side role‑aware cache coherency

  Distinguishes long‑lived resident weights from short‑lived per‑batch activations. Cache invalidation for resident weights runs only once, eliminating redundant coherency traffic.

- Low‑footprint co‑existing backend

  Reuses 100 % of existing Hexagon/HTP kernels; both backends share the htp/ operator source tree. Selection is purely a compile‑time option, zero breaking changes for existing dspqueue users.

- Compatible with [Qualcomm's dspqueue-based ggml-hexagon](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hexagon)
   
### Design & Trade‑offs
> This section explains core design decisions, alternatives considered, and accepted trade‑offs.

1. Re‑use existing Hexagon kernels
    - Pro: Avoid duplicate maintenance of HTP operator implementations. All existing performance‑optimized HVX/HMX kernels are shared between the two backends. **Do not touch the Hexagon kernel code** to reduce maintenance burden, only focus on ggml-hexagon-fastrpc.cpp and entry.c.
    - Con: Backend improvements to HTP operators must land once and benefit both transports; host‑side logic differs for memory pool / RPC handling.

2. Build‑time variant selection, no ABI difference
    - Two backends live within one source directory and one build system. Output artifacts (libggml‑hexagon.so, libggml‑htp‑vXX.so) have identical filenames.
    - Pro: Existing downstream integrations do not need to change linkage / library loading logic.
    - Con: You must re‑compile to switch RPC transport and toggle runtime libs manually(refer to section:How to reproduce the benchmark results); cannot swap transports at runtime.

3. Single shared mempool + NPU‑side role‑aware cache coherency
    - Resident weights vs per‑batch activations are differentiated on NPU side; first‑touch cache invalidation is applied only once for resident weights. This is the key optimization enabling large lm‑head offloading and major PP / TG throughput gains for many models.
    - Pro: Reduces expensive repeated cache maintenance operations on AP side; enables full lm‑head offloading.
    - Con: Subject to Hexagon DSP 32‑bit virtual address space limit (4 GiB). Models exceeding this limit trigger fallback heap‑mirror memcpy path with performance penalty.


4. Minimal incremental code footprint
    - Most source files are unchanged; only a small set of new host‑side / DSP entry / IDL files are added.
    - Pro: Low ongoing maintenance burden; risk of regression is constrained to new files.
    - Con: FastRPC async capability is currently disabled (accepted limitation).



### Build option mapping
| CMake flags | Backend | Description |
|---|---|---|
| `-DGGML_HEXAGON=ON` (default) | dspqueue‑based | Original upstream implementation, unchanged behavior |
| `-DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_MEMPOOL=ON` | FastRPC‑based | New mempool / FastRPC transport backend |


### Changed / New source files
New files:
- ggml/src/ggml‑hexagon/ggml‑hexagon‑fastrpc.cpp — AP‑side host implementation
- ggml/src/ggml‑hexagon/htp/entry.c — HTP DSP entry point for FastRPC path
- ggml/src/ggml‑hexagon/htp/dsp‑ctx.h — HTP session context & descriptors (avoids name collision with existing htp‑ctx.h)
- ggml/src/ggml‑hexagon/htp/ggml_htp.idl — FastRPC IDL definition

Modified files:
- ggml/src/ggml‑hexagon/CMakeLists.txt
- ggml/src/ggml‑hexagon/htp/CMakeLists.txt

Optional utility script (for local verification & CI):
- scripts/build‑run‑ggmlhexagon‑android.sh ------ helper for building variants, pushing binaries to device, running batch AB‑benchmarks. **No docker dependency required**(The shell-based build & run CI script can be modified to fit your needs --- no black box).



## Benchmark results

### Test device

Snapdragon 8 Elite (aka 8 Gen 4), QCOM_HTP_V79, VTCM=8MB, HVX+HMX

### PP&TG in [dspqueue-based ggml-hexagon(aka Qualcomm's official ggml-hexagon)](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hexagon)

<img width="1905" height="401" alt="Screenshot from 2026-08-24 11-21-31" src="https://github.com/user-attachments/assets/657659c4-a341-4e47-9443-fe3b7e01629d" />
<img width="1907" height="398" alt="Screenshot from 2026-08-24 11-18-40" src="https://github.com/user-attachments/assets/f7500383-692a-4adb-adee-590b033ec67d" />
<img width="1916" height="335" alt="Image" src="https://github.com/user-attachments/assets/5b766ab7-aaf9-442f-b117-a7ffd12acd22" />

### PP&TG in [FastRPC-based ggml-hexagon](https://github.com/ggml-hexagon)

<img width="1938" height="884" alt="Screenshot from 2026-08-24 11-16-48" src="https://github.com/user-attachments/assets/3f6f9a70-30ba-4b60-b077-8e4cb6470b73" />
<img width="1908" height="884" alt="Screenshot from 2026-08-24 11-14-48" src="https://github.com/user-attachments/assets/e5875b15-5e65-4092-b316-03412b2a47f8" />
<img width="1913" height="881" alt="Image" src="https://github.com/user-attachments/assets/dc0f619d-0534-49bc-92ad-01013b904694" />

### Eight-Model AB Test Results (fastrpc vs dspqueue)


| Model        | Layers | Model Size | fastrpc PP | dspqueue PP |    PP Diff | fastrpc TG | dspqueue TG |    TG Diff |
| ------------ | -----: | ---------- | ---------: | ----------: | ---------: | ---------: | ----------: | ---------: |
| minicpm5-1b  |     24 |    635MiB    |     981.54 |      824.59 |     +19.0% |      52.54 |       37.99 |     +38.3% |
| Qwen3.5-2B   |     24 |       1.2GiB |     384.16 |      348.96 |     +10.1% |      22.39 |       15.76 |     +42.1% |
| Qwen3.5-4B   |     32 |       2.5GiB |     183.69 |      142.80 |     +28.6% |      10.47 |        9.19 |     +13.9% |
| Spark-X2.5-1.7B |  28 |       3.2GiB |     355.35 |       10.45 |  +3300.5% |      15.00 |        5.78 |    +159.5% |
| gemma-4-E2B  |     35 |       2.9GiB |     557.48 |      383.21 |     +45.5% |      27.82 |       22.51 |     +23.6% |
| Nanbeige-3B  |     22 |       2.4GiB |     286.56 |      166.08 |     +72.5% |       7.94 |        8.60 |      -7.7% |
| gemma-4-E4B  |     42 |       4.9GiB |     334.86 |      338.37 |      -1.0% |      14.72 |       10.03 |     +46.8% |
| Qwen3.5-9B   |     32 |       5.1GiB |      11.05 |       99.35 |     -88.9% |       3.91 |        5.95 |     -34.3% |



### How to reproduce the benchmark results

The build script (`scripts/build-run-ggmlhexagon-android.sh`) **automatically downloads** and sets up dependencies(**8 models** and dependent SDKs) on first run.

<details>
<summary> usage of ./scripts/build-run-ggmlhexagon-android.sh </summary>

```
Usage:
  ./scripts/build-run-ggmlhexagon-android.sh help
  ./scripts/build-run-ggmlhexagon-android.sh build                    (build the mempool/FastRPC-invoke ggml-hexagon backend for performance comparision)
  ./scripts/build-run-ggmlhexagon-android.sh build_dspqueue           (build the dspqueue ggml-hexagon backend for performance comparison, Qualcomm's official dspqueu-based ggml-hexagon)
  ./scripts/build-run-ggmlhexagon-android.sh build_armcpu             (build Android CPU-only reference for correctness check and troulbeshooting trick issues)
  ./scripts/build-run-ggmlhexagon-android.sh clean
  ./scripts/build-run-ggmlhexagon-android.sh update_fastrpc_libs      (push mempool/FastRPC runtime .so from out/ab-test/ to device, for build)
  ./scripts/build-run-ggmlhexagon-android.sh update_dspqueue_libs     (push dspqueue runtime .so from out/ab-test/ to device, for build_dspqueue)
  ./scripts/build-run-ggmlhexagon-android.sh update_cpu_libs          (push CPU-only runtime .so from out/ab-test/ to device, for build_armcpu)
  ./scripts/build-run-ggmlhexagon-android.sh update_ggml_libs         (incremental: push AP-side libs from bin/ to device only; keep DSP skels as-is)
  ./scripts/build-run-ggmlhexagon-android.sh run_llamaversion         (display llama-cpp version information, e.g. version: 0.2.0-dev (build 11120, commit 9f03708a9), built with Clang 21.0.0 for Android aarch64)
  ./scripts/build-run-ggmlhexagon-android.sh run_testops
  ./scripts/build-run-ggmlhexagon-android.sh run_testop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify accuracy    of ADD/MUL_MAT)
  ./scripts/build-run-ggmlhexagon-android.sh run_perfop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify performance of ADD/MUL_MAT)


  ./scripts/build-run-ggmlhexagon-android.sh run_abtest_all [rounds]
    Batch AB test across all 8 models (qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b).
    rounds: default 3
    Log capture example:
      ./scripts/build-run-ggmlhexagon-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt


  ./scripts/build-run-ggmlhexagon-android.sh run_llamacli     [model_alias]
  ./scripts/build-run-ggmlhexagon-android.sh run_llamabench   [model_alias]
  Model aliases for run_llamacli:
    qwen3-2b      -> Qwen3.5-2B-Q4_0.gguf
    qwen3-9b      -> Qwen3.5-9B-Q4_0.gguf
    gemma4-e2b    -> gemma-4-E2B-it-Q4_0.gguf
    gemma4-e4b    -> gemma-4-E4B_q4_0-it.gguf
    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf
    llama3        -> Llama-3.2-1B-Instruct-Q4_0.gguf
    nanbeige-3b   -> Nanbeige_Nanbeige4.2-3B-Q4_0.gguf
    minicpm5-1b   -> minicpm5-1b-q4_0.gguf
    (default)     -> gemma-4-E2B-it-Q4_0.gguf
  Examples:
    ./scripts/build-run-ggmlhexagon-android.sh run_llamacli/run_llamabench              # run gemma4-e2b inference test on an Qualcomm mobile SoC-based Android phone
    ./scripts/build-run-ggmlhexagon-android.sh run_llamacli/run_llamabench qwen3-2b     # test qwen3-2b
    ./scripts/build-run-ggmlhexagon-android.sh run_llamacli/run_llamabench gemma4-e2b   # test gemma4-e2b
    ./scripts/build-run-ggmlhexagon-android.sh run_llamacli/run_llamabench gemma4-e4b   # test gemma4-e4b

```
</details>


```
# build the dspqueue backend (Qualcomm's official) for performance comparison
./scripts/build-run-ggmlhexagon-android.sh build_dspqueue

# build Android CPU-only backend for performance comparison
./scripts/build-run-ggmlhexagon-android.sh build_armcpu

# build the fastrpc backend for performance comparison
./scripts/build-run-ggmlhexagon-android.sh build

# llama-bench with the dspqueue backend
./scripts/build-run-ggmlhexagon-android.sh update_dspqueue_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench

# llama-bench with the fastrpc backend
./scripts/build-run-ggmlhexagon-android.sh update_fastrpc_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench

# llama-bench with Android CPU-only backend
./scripts/build-run-ggmlhexagon-android.sh update_cpu_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench
```
### PP & TG Performance Comparison: dspqueue-based ggml-hexagon vs FastRPC-based ggml-hexagon

Pls refer to: https://github.com/ggml-hexagon/ggml-hexagon/discussions/71

## Verified SoCs

| SoC | HTP Arch  | VTCM | Status |
|-----|----------|----------------|--------|
| Snapdragon 8 Gen 2 | v73  | unknown | Not tested |
| Snapdragon 8 Gen 3 | v75 |  8MB | Tested & verified |
| Snapdragon 8 Elite (8 Gen 4) | v79 |  8MB | Tested & verified (**Recommended**) |
| Snapdragon 8 Elite Gen5 (8 Gen 5) | v81  | unknown | Not tested |
| Snapdragon X Elite | v79  | unknown | Not tested |
| Snapdragon X2 Elite | v81  | unknown | Not tested |

## Known Limitations
1. 4 GiB DSP virtual‑address‑space limit (HTP‑v75 / v79) due to limitations in the Qualcomm Hexagon SDK.
    > Hexagon user‑mode has a 32‑bit byte‑addressable address space; user‑mode code can only directly map up to 4 GiB of memory at one time (Qualcomm documentation reference: https://docs.qualcomm.com/doc/80‑N2040‑60/topic/memory.html).

    When model weight footprint exceeds 4 GiB (example: Qwen3.5‑9B ~5.1 GiB), the FastRPC backend cannot fit everything inside the shared mempool. It falls back to heap‑allocation with mirror‑buffer memcpy for overflow weights, which introduces substantial token‑generation overhead. This explains the large performance regression observed for Qwen3.5‑9B/Spark-X2.5-4B in benchmark.

2. FastRPC async is currently disabled due to limitations in the Qualcomm Hexagon SDK.

3. There are two large PRs from Qualcomm: [PR #26501](https://github.com/ggml-org/llama.cpp/pull/26501) (commit 192067b72d1b7a3653b3f0c59190303b18596637, "hexagon: support for multi-NPU devices (IQ9, IQ10) and fully asynchronous backend") and [PR #28589](https://github.com/ggml-org/llama.cpp/pull/28589) (commit eafe15a5e3d87dd68ae33acf6a7cbd9415a0ac5e, "hexagon: support for multi-device model split (aka row-split)"). The FastRPC-based ggml-hexagon has no real multi-NPU implementation due to the lack of suitable hardware for development and testing(implementation based on PR-26501 and PR-28589 will be done in less than 24 hours once suitable hardware is available).

## Contribution Notes

Follows [CONTRIBUTING.md](https://github.com/ggml-hexagon/ggml-hexagon/blob/self-build-jz/CONTRIBUTING.md).


## Disclosure on AI Use

Per project llama.cpp's policies, I am disclosing that significant portions of this work have been developed with AI assistance. All code has been meticulously reviewed, revised and tested by me to ensure the implementation aligns with the rest of the project.

The core ideas originate from my fully‑original, hand‑written PR‑12326. Starting in June 2026, I have used AI coding agents to assist with brainstorming, drafting code snippets, composing technical documentation (with questions and scope defined by me), and generating test reports.

I have inspected, tested, and fully understand all code within ggml-hexagon-fastrpc.cpp and htp/entry.c, excluding quantization-type conversion routines. Most importantly, all technical decisions, design directions and code adjustments are made solely by me.


## Acknowledgement

-    [07/01/2026, July 1 2026] Thanks to Trae and GLM‑5.2 for their great assistance. Qualcomm’s ggml‑hexagon implementation also provided valuable reference. GLM‑5.2 and I co‑designed the mempool‑based op‑batch solution after many hours of iteration. I intentionally avoid Qualcomm’s dspqueue within JZ's ggml‑hexagon, as this dspqueue‑free, mempool‑based op‑batch mechanism is one of the key highlights of this backend. GLM‑5.2 has acted as a co‑contributor to JZ's ggml‑hexagon starting June, 2026.

-   [07/06/2026, July 6 2026] Thanks for Trae + DeepSeek-V4-Pro's great help. DeepSeek-V4-Pro did a good job in performance optimization.

-    [07/09/2026, July 9 2026] Thanks for Trae + MiniMax-M3's breakthrough help and profound insights, the PP performance has been boosted from around 180 to over 300 based on Qualcomm's new operators/kernels.

-   [07/09/2026, July 9 2026] GLM-5.2, DeepSeek-V4-Pro, MiniMax-M3 are both China's top AI Coding Models, sincerely thanks for the original authors of them.

-    [07/14/2026, July 14 2026] Kimi-K2.7 also made solid contribution on 07/13/2026 although Kimi-K2.7-Code joined this project on late evening 07-13-2026.

-   [07/14/2026, July 14 2026] There would be no 100% upstream compatible FastRPC-based ggml-hexagon without the excellent operators/kernels implementation provided by Qualcomm, because the FastRPC-based ggml-hexagon can directly 100% re-use the highly-excellent hexagon kernels.

-    [07/17/2026, July 17 2026] Kimi-K3 joined this project on late evening 07-17-2026, Kimi-K3 did a breakthrough optimization(offload lm-head) in this backend.

-    [08, 2026] Xiaomi Mimo V2.5 joined this project.

## Language Policy

Both English and Chinese posts are accepted for discussions and issues in this project.

