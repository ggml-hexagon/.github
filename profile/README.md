# ggml‑hexagon
> FastRPC‑based ggml‑hexagon

 English | [中文](https://github.com/ggml-hexagon/.github/blob/main/profile/README-zh.md) 


## Overview
This organization maintains FastRPC‑based Qualcomm Hexagon NPU backend for ggml / llama.cpp, as an alternative to Qualcomm's dspqueue implementation.

### Key Features
- Compatible with [Qualcomm's dspqueue-based ggml-hexagon](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hexagon)
- PP and TG offer [some advantages](https://github.com/ggml-hexagon/ggml-hexagon/discussions/71) over Qualcomm's dspqueue‑based ggml‑hexagon for certain modern models
- Originated from upstream [PR #12326](https://github.com/ggml-org/llama.cpp/pull/12326), with follow‑up upstream PRs: [PR #26373](https://github.com/ggml-org/llama.cpp/pull/26373), [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642).


### Repositories
- [ggml‑hexagon](https://github.com/ggml-hexagon/ggml-hexagon): Main backend repository


## Acknowledgement

-    [07/01/2026, July 1 2026] Thanks to Trae and GLM‑5.2 for their great assistance. Qualcomm’s ggml‑hexagon implementation also provided valuable reference. GLM‑5.2 and I co‑designed the mempool‑based op‑batch solution after many hours of iteration. I intentionally avoid Qualcomm’s dspqueue within JZ's ggml‑hexagon, as this dspqueue‑free, mempool‑based op‑batch mechanism is one of the key highlights of this backend. GLM‑5.2 has acted as a co‑contributor to JZ's ggml‑hexagon starting June, 2026.

-   [07/06/2026, July 6 2026] Thanks for Trae + DeepSeek-V4-Pro's great help. DeepSeek-V4-Pro did a good job in performance optimization.

-    [07/09/2026, July 9 2026] Thanks for Trae + MiniMax-M3's breakthrough help and profound insights, the PP performance has been boosted from around 180 to over 300 based on Qualcomm's new operators/kernels.

-   [07/09/2026, July 9 2026] GLM-5.2, DeepSeek-V4-Pro, MiniMax-M3 are both China's top AI Coding Models, sincerely thanks for the original authors of them.

-    [07/14/2026, July 14 2026] Kimi-K2.7 also made solid contribution on 07/13/2026 although Kimi-K2.7-Code joined this project on late evening 07-13-2026.

-   [07/14/2026, July 14 2026] There would be no 100% upstream compatible FastRPC-based ggml-hexagon without the excellent operators/kernels implementation provided by Qualcomm, because the FastRPC-based ggml-hexagon can directly 100% re-use the highly-excellent hexagon kernels.

-    [07/17/2026, July 17 2026] Kimi-K3 joined this project on late evening 07-17-2026, Kimi-K3 did a breakthrough optimization(offload lm-head) in this backend.

-    [08, 2026] Xiaomi Mimo V2.5 joined this project.



## Disclosure on AI Use

Per project llama.cpp's policies, I am disclosing that significant portions of this work have been developed with AI assistance. All code has been meticulously reviewed, revised and tested by me to ensure the implementation aligns with the rest of the project.

The core ideas originate from my fully‑original, hand‑written PR‑12326. Starting in June 2026, I have used AI coding agents to assist with brainstorming, drafting code snippets, composing technical documentation (with questions and scope defined by me), and generating test reports.

I have inspected, tested, and fully understand all code within ggml-hexagon-fastrpc.cpp (formerly ggml-hexagon-jz.cpp) and htp/entry.c, excluding quantization-type conversion routines. Most importantly, all technical decisions, design directions and code adjustments are made solely by me.
