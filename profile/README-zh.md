# ggml-hexagon

> 让大语言模型推理跑在高通 Hexagon NPU 上——基于原生 FastRPC 的轻量、高效、长期维护的 ggml-hexagon 后端

`ggml-hexagon` 是一个面向 Qualcomm Hexagon NPU 的独立维护组织，目标是长期维护和发展一套基于**原生 FastRPC 机制**的 ggml-hexagon 后端变体（fastrpc 变体），并持续跟踪上游 `llama.cpp` / `ggml` 项目的演进。该后端面向基于Snapdragon SoC的Android / Linux / WoS（Windows on Snapdragon）设备，可与 Qualcomm 官方的 dspqueue 后端共存，二者关系类似 `llama.cpp` 内的 ggml-openvino vs ggml-sycl，或 ggml-cuda vs ggml-hip。本组织由独立开发人员 Jeff Zhou （ GitHub: [`zhouwg`](https://github.com/zhouwg) / [`jeffzhou-zhouwg`](https://github.com/jeffzhou-zhouwg) /  [`jeff-zhouwg`](https://github.com/jeff-zhouwg) ）发起，承载其从 2024 年 3 月起在 Hexagon NPU 后端上的持续工程实践。

***

## 这个组织为什么存在

Qualcomm Hexagon SDK提供了两种RPC机制：native FastRPC与dspqueue。Qualcomm 官方的 ggml-hexagon 后端使用 `dspqueue` 的异步队列框架，在 AP（应用处理器）与 DSP（aka cDSP / HTP / NPU）之间交换数据。`dspqueue` 本质上是原生 FastRPC 机制的高层封装 ------ 而 LLM 推理在本质上是同步的，且 AP 与 NPU 共享同一块 DDR 物理内存，两者能同时"看到"同一区域。

这意味着完全可以在原生 FastRPC 之上，实现一套更高效的方案：用一次 `fastrpc_mmap` 建立单个内存池，用同步 `invoke` 携带整张计算图批次，把算子卸载到 Hexagon NPU 上执行。

Jeff Zhou 多次尝试将该方案合入上游 `llama.cpp`：2025 年 7 月的 [PR #12326](https://github.com/ggml-org/llama.cpp/pull/12326) 被封禁；不知道什么原因，2026 年的 [PR #26373](https://github.com/ggml-org/llama.cpp/pull/26373) 与最新的 [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642)（2026/08/24 提交）相继被上游维护者关闭。为避免贡献反复被埋没、并保证这套基于 FastRPC 的实现能被长期维护和迭代，`ggml-hexagon` 组织2026年8月27日由此而生。

***

## 验证的 SoC

| SoC                              | HTP 架构 | VTCM | 状态        |
| -------------------------------- | ------ | ---- | --------- |
| Snapdragon 8 Gen 2               | v73    | 未知   | 未测试       |
| Snapdragon 8 Gen 3               | v75    | 8MB  | 已测试验证     |
| Snapdragon 8 Elite（8 Gen 4）      | v79    | 8MB  | 已测试验证（推荐） |
| Snapdragon 8 Elite Gen5（8 Gen 5） | v81    | 未知   | 未测试       |
| Snapdragon X Elite               | v79    | 未知   | 未测试       |
| Snapdragon X2 Elite              | v81    | 未知   | 未测试       |

***

## fastrpc 变体

fastrpc 变体（即原"JZ's ggml-hexagon"）与 dspqueue 变体（Qualcomm 官方后端）共享同一棵 DSP 算子源码树 `ggml/src/ggml-hexagon/htp/*.c`，fastrpc 变体**100% 复用现有 Hexagon 内核**，二者只是构建时的两个变体，由一个 CMake 选项切换：

```cmake
option(GGML_HEXAGON_USE_MEMPOOL
  "mempool/FastRPC-invoke 实现（单个 ION 内存池，同步 invoke）；OFF = dspqueue/per-buffer 实现"
  OFF)
```

* `GGML_HEXAGON=ON` + `GGML_HEXAGON_USE_MEMPOOL=ON` → fastrpc 变体：`ggml-hexagon-fastrpc.cpp` + `htp/entry.c` + `htp/ggml_htp.idl`

* `GGML_HEXAGON=ON`（默认）→ dspqueue 变体：`ggml-hexagon.cpp` + `htp/main.c` + `htp/htp_iface.idl`，即上游原有行为

两个变体最终都产出**同名**产物：AP 侧的 `libggml-hexagon.so` 与 DSP 侧的 `libggml-htp-vXX.so`。该切换**零破坏性变更**，官方 dspqueue 后端照常构建与运行。

数据面两者几乎一致：都通过 AP-DSP 共享内存搬运张量数据，都需要 cache flush / invalidate 同步，最终都跑同一套 HVX/HMX 内核。真正的性能差异不在算子内核本身，而在**调度框架、cache 策略与卸载策略**（详见 [架构分析文档](https://github.com/zhouwg/ggml-hexagon/blob/self-build-jz/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)）。

***

## 架构对比

| 维度                  | fastrpc 变体                    | dspqueue 变体                           |
| ------------------- | ----------------------------- | ------------------------------------- |
| 控制面                 | 原生 FastRPC `invoke`（同步）       | `dspqueue_write/read`（异步，最多 16 个并发批次） |
| 数据面                 | 单个 mempool + 偏移寻址             | per-chunk + `bi` 间接寻址                 |
| DSP 入口              | `htp/entry.c`                 | `htp/main.c`                          |
| AP 侧代码              | `ggml-hexagon-fastrpc.cpp`    | `ggml-hexagon.cpp`                    |
| 构建选项                | `GGML_HEXAGON_USE_MEMPOOL=ON` | `GGML_HEXAGON_USE_MEMPOOL=OFF`（默认）    |
| `fastrpc_mmap` 调用次数 | 初始化时 1 次                      | 每个 chunk 1 次                          |
| fd 数量               | 1 个                           | 每个 chunk 1 个                          |
| DSP 张量寻址            | 直接 `void *` 偏移                | `bi` → `htp_buf_desc[]` 间接寻址          |
| 生命周期                | 单次 alloc/free                 | 每个 chunk 的 alloc/mmap/munmap/free     |
| IOVA 空间局部性          | 连续、可预测                        | 跨 chunk 碎片化                           |
| cache 一致性           | NPU 侧：角色感知（权重 vs 激活）          | 内核态驱动标志（按批次统一处理）                      |
| lm-head 卸载到 DSP     | 可行                            | 不可行                                   |

***

## 关键优势：把 lm-head 卸载到 DSP

TG（token 生成）阶段在两个后端上都是带宽受限的——每生成一个 token，都要把全部权重从 DRAM 重新读一遍。lm-head 矩阵乘是 TG 单项最大开销。此前两个变体都拒绝 `ne[1] > 32768` 的量化权重矩阵，所以 lm-head 只能留在 CPU 上跑。

把 lm-head 卸载到 DSP，需要一份重排（分块）后的权重在 DSP 可寻址内存中驻留整个会话。在 dspqueue 变体的 per-chunk 设计下，每个驻留缓冲区都要背负自己的 fd、`fastrpc_mmap`、按批次的描述符重注册、DSP 侧有限的 vmem 预算内反复 eviction/remap，以及必须与 DSP 侧 unmap 协调的生命周期——这正是 dspqueue 变体保留 32768 行守卫、把 lm-head 留在 CPU 的原因。

fastrpc 变体在初始化时映射一次内存池（v79 上容量可探测到 4032 MiB）。在加载时把 lm-head 重排进内存池只需一次转换；之后这份重排就是一个已映射区域内的偏移区间——零循环 fd/mmap/生命周期成本。lm-head 卸载是 [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642) PP/TG 吞吐优势的主要来源，且高度依赖单一共享内存池设计。三个关键改动都因单一内存池而成为可能：

1. **移除** **`ne[1] > 32768`** **守卫**，允许 lm-head 卸载。
2. **Q4\_K 以 Q4\_0 分块重排存储**（约 214 MB 常驻），把 Q4\_K 权重转成 DSP 可直接执行的 Q4\_0 分块布局，使卸载成为可能（重排不降低带宽，二者数据量相同）。
3. **首次触碰式权重失效**。重排权重在加载时由 AP 写一次后不再触碰，首次失效后 DSP 跳过后续每 token 的重复失效，消除约 9.2 ms/token 的 DSP 侧 dcinva 扫描开销。

附带效应是 cache 一致性优势的反转：dspqueue 变体的按批次 cache 维护是统一且角色无关的，无法区分权重与激活。fastrpc 变体的**NPU 侧**角色感知 cache 一致性能区分常驻权重与按批次激活，只对常驻权重做一次首次失效。在最新 [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642) 中，AP 侧冗余的 cache 一致性维护已被移除，使 fastrpc 后端得以在 WoA 设备上运行。

***

## 性能表现

以下为 [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642) 在 **Snapdragon 8 Elite（8 Gen 4，HTP v79，VTCM=8MB，HVX+HMX）** 上对 8 个模型的 AB 测试结果，fastrpc 与 dspqueue 使用相同运行参数、相同 prompt 与相同模型文件：

| 模型           | 层数 | 注意力         | lm\_head | 模型大小      | fastrpc PP | dspqueue PP | PP 差异  | fastrpc TG | dspqueue TG | TG 差异  |
| ------------ | -- | ----------- | -------- | --------- | ---------- | ----------- | ------ | ---------- | ----------- | ------ |
| qwen1.5-1.8B | 24 | MHA 1:1     | Q6\_K    | 1.1 GiB   | 521.36     | 738.04      | -29.4% | 17.12      | 24.31       | -29.6% |
| minicpm5-1b  | 24 | GQA 8:1     | Q4\_0    | \~0.7 GiB | 1161.77    | 1016.62     | +14.3% | 51.95      | 36.14       | +43.8% |
| Llama-3.2-1B | 16 | GQA 4:1     | Q4\_K    | 0.7 GiB   | 1006.10    | 1063.43     | -5.4%  | 41.64      | 28.33       | +47.1% |
| Qwen3.5-2B   | 24 | GQA + Delta | Q6\_K    | 1.2 GiB   | 485.07     | 434.96      | +11.5% | 26.99      | 14.95       | +80.5% |
| gemma-4-E2B  | 35 | GQA 8:1     | Q4\_K    | 2.9 GiB   | 668.14     | 490.09      | +36.3% | 27.00      | 24.99       | +8.0%  |
| Nanbeige-3B  | 22 | GQA 6:1     | Q4\_0    | 2.4 GiB   | 224.14     | 136.73      | +63.9% | 8.03       | 9.11        | -11.9% |
| gemma-4-E4B  | 42 | GQA 4:1     | Q4\_K    | 4.9 GiB   | 403.76     | 426.82      | -5.4%  | 14.75      | 12.08       | +22.1% |
| Qwen3.5-9B   | 32 | GQA + Delta | Q6\_K    | 5.0 GiB   | 11.28      | 114.14      | -90.1% | 5.80       | 6.77        | -14.3% |

数据呈现两个清晰规律：

* **TG 普遍占优**：得益于 lm-head 卸载，fastrpc 在多数模型的 TG 上明显领先，最高达 +80.5%（Qwen3.5-2B）。唯一 TG 落后的两个模型（qwen1.5-1.8B、Nanbeige-3B）均为 Q4\_0 lm-head 的小模型。

* **大模型明显劣势**：Qwen3.5-9B（5.0 GiB）PP 落后 90.1%、TG 落后 14.3%，根因是 DSP 4 GiB 虚拟地址空间限制（见下节），溢出权重回退到堆分配 + mirror memcpy，产生显著开销。



***

## 已知限制

* **4 GiB DSP 虚拟地址空间限制（htp-v75 / htp-v79）**：Hexagon V79 用户态为 32 位字节可寻址内存地址空间，DSP 任意时刻只能 mmap/访问 ≤ 4 GiB。大于 4 GiB 的模型（如 Qwen3.5-9B 的 5.1 GiB）无法整体装入单个 mempool，fastrpc 后端对溢出权重回退到堆分配 + mirror memcpy，带来显著 TG 开销——这是 Qwen3.5-9B 性能远逊于 dspqueue 变体的根因。

* **FastRPC async 被禁用**。



***

## 仓库结构

主仓库 `ggml-hexagon` 由 `llama.cpp` fork 而来，维护两条分支：

* `master`：跟踪上游 `llama.cpp` 项目。

* `self-build-jz`：默认分支，Qualcomm 的 dspqueue-based ggml-hexagon 与 fastrpc-based ggml-hexagon的全部代码都在这里。

[PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642) 中真正新增或修改的文件极少：

| 文件                                               | 说明                                     |
| ------------------------------------------------ | -------------------------------------- |
| `ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp` | 新增；fastrpc 变体 AP 侧代码                   |
| `ggml/src/ggml-hexagon/htp/entry.c`              | 新增；HTP 入口                              |
| `ggml/src/ggml-hexagon/htp/dsp-ctx.h`            | 新增；HTP 会话上下文与描述符（避开与已有 `htp-ctx.h` 重名） |
| `ggml/src/ggml-hexagon/htp/ggml_htp.idl`         | 新增；FastRPC IDL                         |
| `ggml/src/ggml-hexagon/CMakeLists.txt`           | 修改；变体切换                                |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt`       | 修改；变体切换                                |
| `scripts/build-run-ggmlhexagon-android.sh`       | 可选；简化两后端的构建与 CI                        |

dspqueue 变体侧仍为 `ggml-hexagon.cpp` + `htp/main.c` + `htp/htp_iface.idl`。两个变体共享同一棵 `htp/*.c` 算子源码树。



***

## 如何构建

构建已在 Ubuntu 20.04（2025/05/31 EOL）与 Ubuntu 26.04 上验证，推荐 Ubuntu 26.04。构建脚本首次运行时会**自动下载并配置依赖**（8 个模型与所需 SDK）。

获取源码并构建 fastrpc 变体：

```bash
git clone https://github.com/ggml-hexagon/ggml-hexagon.git
cd ggml-hexagon
git checkout self-build-jz

./scripts/build-run-ggmlhexagon-android.sh build
```

在 Ubuntu 20.04 上需先把 cmake 升级到 4.2.3：

```bash
sudo apt remove -y cmake
wget https://cmake.org/files/v4.2/cmake-4.2.3-linux-x86_64.tar.gz
tar -zxf cmake-4.2.3-linux-x86_64.tar.gz
sudo mv cmake-4.2.3-linux-x86_64 /opt/
sudo ln -sf /opt/cmake-4.2.3-linux-x86_64/bin/* /usr/bin/
cmake --version
```

### 构建脚本完整用法

```text
./scripts/build-run-ggmlhexagon-android.sh help
./scripts/build-run-ggmlhexagon-android.sh build            # 构建 mempool/FastRPC 后端
./scripts/build-run-ggmlhexagon-android.sh build_dspqueue   # 构建 dspqueue 后端（Qualcomm 官方）
./scripts/build-run-ggmlhexagon-android.sh build_armcpu     # 构建 Android 纯 CPU 版本，用于正确性校验
./scripts/build-run-ggmlhexagon-android.sh clean
./scripts/build-run-ggmlhexagon-android.sh update_fastrpc_libs   # 推送 fastrpc 运行时 .so 到设备
./scripts/build-run-ggmlhexagon-android.sh update_dspqueue_libs # 推送 dspqueue 运行时 .so 到设备
./scripts/build-run-ggmlhexagon-android.sh update_cpu_libs      # 推送纯 CPU 运行时 .so 到设备
./scripts/build-run-ggmlhexagon-android.sh update_ggml_libs     # 增量：只推送 AP 侧库，保留 DSP skel
./scripts/build-run-ggmlhexagon-android.sh run_llamaversion     # 显示 llama-cpp 版本信息
./scripts/build-run-ggmlhexagon-android.sh run_testops
./scripts/build-run-ggmlhexagon-android.sh run_testop ADD/MUL_MAT/FLASH_ATTN_EXT  # 校验算子正确性
./scripts/build-run-ggmlhexagon-android.sh run_perfop ADD/MUL_MAT/FLASH_ATTN_EXT  # 校验算子性能
./scripts/build-run-ggmlhexagon-android.sh run_abtest_all [rounds]  # 跨全部 8 模型批量 AB 测试，默认 3 轮
./scripts/build-run-ggmlhexagon-android.sh run_llamacli [model_alias]
./scripts/build-run-ggmlhexagon-android.sh run_llamabench [model_alias]
```

模型别名：`qwen3-2b`、`qwen3-9b`、`gemma4-e2b`、`gemma4-e4b`、`qwen1`、`llama3`、`nanbeige-3b`、`minicpm5-1b`，默认 `gemma4-e2b`。

### 公平对比三个后端

为保证公平，三个后端（fastrpc / dspqueue / 纯 CPU）使用相同运行参数、相同 prompt、相同模型文件与同一台 8 Elite 手机：

```bash
# 构建三个后端
./scripts/build-run-ggmlhexagon-android.sh build_dspqueue
./scripts/build-run-ggmlhexagon-android.sh build_armcpu
./scripts/build-run-ggmlhexagon-android.sh build

# 跨 8 模型批量 AB 测试并记录日志
./scripts/build-run-ggmlhexagon-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt
```

单独跑某个模型的 llama-bench 时，先切换库再运行：

```bash
# dspqueue 变体
./scripts/build-run-ggmlhexagon-android.sh update_dspqueue_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench qwen3-2b

# fastrpc 变体
./scripts/build-run-ggmlhexagon-android.sh update_fastrpc_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench qwen3-2b

# 纯 CPU 参照
./scripts/build-run-ggmlhexagon-android.sh update_cpu_libs
./scripts/build-run-ggmlhexagon-android.sh run_llamabench qwen3-2b
```

***

## 如何参与

本组织欢迎在 Hexagon NPU 上做过真实测试而非仅跑 benchmark 的开发者与 AI 大模型团队参与。如果你认为自己的模型足够好，可以用 `self-build-jz` 分支做真机实测。

* 欢迎提交 PR 与参与 Discussions 讨论。

* 详细的工程背景与演进历史见 [about ggml-hexagon #18](https://github.com/zhouwg/ggml-hexagon/discussions/18)。

* 最新上游 PR 与 8 模型基准见 [PR #27642](https://github.com/ggml-org/llama.cpp/pull/27642)。

***

## 许可证

继承上游 `llama.cpp` 的 **MIT** 许可证。
