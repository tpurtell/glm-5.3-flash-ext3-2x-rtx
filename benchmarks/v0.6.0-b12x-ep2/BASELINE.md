# v0.6.0 B12x/EP2 baseline manifest

Captured 2026-09-01 before changing the v0.5.0 runtime. This records the
immutable release-A inputs. The controlled A/B measurements use the same
explicitly verified 400 W/GPU cap as the previous release measurements. At the
initial capture GPU 0 was owned by an unrelated service, so the historical
receipts below remain provenance rather than substitutes for the new isolated
comparison.

## Immutable release state

| Item | Value |
|---|---|
| Recipe commit/tag | `d46fdeddf8c6fec2d4595b65535a32d80a5af787` / `v0.5.0` |
| Public image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.5.0` |
| Public image digest | `sha256:001a45bd71bcf908a8c5e0a205d085f29ac7f3201529fa3eb75` |
| Target model | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4` |
| DFlash2 model | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| B12x | `tpurtell/sparkinfer-glmrt@988246c8b007c9c1c2006eb677f6fa4b26aeb561` |
| DFlash2 vLLM change | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` |
| EXL3 adapter source | `tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 source vLLM commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| GLM base image | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| Runtime vLLM version | `0.1.dev20051+g487ecf187` |
| Torch / CUDA / CUTLASS DSL | `2.13.x` / `13.0.1` / `4.6.2` |

The public image labels independently report the same recipe, B12x, DFlash2,
base-image, and EXL3-source identities. The base image itself labels its vLLM
commit `unknown`; the runtime package version above is therefore the strongest
available vLLM identity and this limitation is retained rather than invented
away.

## Default launcher profile

The exact launcher is `start.sh` with SHA-256
`8cd7875acbf70b1b7a06fbedd823b473c012fcb5acc6cd99e214fc3e03ca438a`.
With no overrides it resolves to:

- TP2, DCP2, `ag_rs`, and no expert parallelism.
- DFlash2 K5 with a replicated BF16 draft cache.
- FP8 target MLA (`fp8_ds_mla`) through `B12X_MLA_SPARSE`.
- B12x sparse indexer, k-pool indexer, PCIe all-reduce, DCP owner exchange,
  H64 query projection, and mHC enabled or auto-selected.
- EXL3 Trellis decode M=1..32, prefill Trellis enabled, prefill block M=64,
  and prefill capacity 1024.
- `max_model_len=1048576`, `max_num_batched_tokens=2048`,
  `max_num_seqs=16`, `max_cudagraph_capture_size=96`, and
  `gpu_memory_utilization=0.950`.
- Prefix caching with aligned recurrent state, multimodal execution with an
  exact 16-image request limit, GLM47 tool parser, and GLM45 reasoning parser.
- FlashInfer autotuning disabled.

The Dockerfile SHA-256 is
`f27db6b80679ddb28cff5c39fef5658497ff1fa9a2e1fa4c38dc3480508693b7`;
the build wrapper SHA-256 is
`853a2a90245bc599ab01277246362dcdb9c789875ef564fbdc83d5b1a753b7cd`.

## Patch hashes

```text
005a5a7dd241bbb558823a9a2692e858dd7e06f20ebca6916a2f3044a2aecd01  patches/port-replayssm-glm53.py
01a45bc45d64baddc68068aed169cefe8a21ca2bce583912161064bbbfdd58a2  patches/port-glm53-mtp-prefix-cache.py
0672336117f3ae40a3c5ac8002e66340ab42ef041acd9b17e32d8aaa5a367cf2  patches/port-b12x-dcp-a2a-glm53.py
086083bb8fdd0238615ecbcd1b836e4737ab64044c880fca5a31e280f4ffdd20  patches/port-b12x-dcp-owner-glm53.py
09ace5a2ce22537df3764526efaf86904503fb1c64a29e5d6a537b65672e0df1  patches/port-b12x-mhc-glm53.py
0c3fe0ef99dc12ee7b7e495f5fde804ddd0e5b790f9fb78f7027717100111fef  patches/port-b12x-nvfp4-glm53.py
11501adc323b52ce369289c663cdfa1d4781e50eec43baf6843c1395188a26d9  patches/port-dflash2-glm-eagle3.py
1797a80d0d8f425da58d4e6a7911a4300e7e3d4949da777c7653dc7debd227cf  patches/b12x_pcie_all_reduce.py
2285b1259e88ee73db1b0f8048274a8d10445730297c284a35fe90316f8b6bad  patches/adaptive_mtp.py
22e6be56c18de5b0f86cb5ec561501a5955cf648ccd8081710f4df85162a5ce4  patches/port-dflash2-glm53.py
23aa23802d416763204b45d30d214bf8e05bd8b88341e89d0637e451249ac933  patches/port-dflash2-glm-kv.py
2d7ba2fa090dcc2521fcb30f5674530492af564f7c2866d2e4b052149a593da4  patches/b12x_dcp_topk.py
35f1e01aa9e3332d9c0ed4099d63401a092fedc8887e844eb7d2a5de0dea0459  patches/port-b12x-dcp-glm53.py
3e50197963ca86e0c7f0a88ab690cb9f5e8059b549c187e21b24d959520f62af  patches/vllm-replayssm-spec.patch
44e6ae4aaf2c3fae56f79af36f719e9ac3e76bcd2c35f84b71564fa70f5dcbdf  patches/port-b12x-pcie-glm53.py
7c3bbd9779d9def6efedfbedadd3a818ba94786dd7ffb46e0e674be5b290432b  patches/port-b12x-glm53.py
7c5f2b6e4d50db49d9376f8007c2b58bfa13f8ed388b00e9fa9e559a3a523d9d  patches/vllm-dynamic-sd-cudagraph.patch
8497b96b7451fc0699e56953ebf7c538712bf9a1a56b65d52e16ea2fb4753ecc  patches/port-adaptive-mtp-glm53.py
9257f597c2bd73609bf42beac908ef4a44c7cb44d4ba34c1b93844bbe2f95afd  patches/port-exl3-mtp-glm53.py
a77a59962146bf9098b62ba08ee026ae17d3c93bf8c6f5f6b258e5c823c8659d  patches/port-b12x-kpool-glm53.py
b03940182270a546d4a053d1d20400312189b4c59b160fab477e7cabf2453a2e  patches/b12x_dcp_a2a.py
b39c2df1e1f5a40fa170bdbc7d00667318f2343bce2db924163eedf7d64b8bae  patches/port-glm53-sm12-stability.py
b41c5b77114ba53b05bc9f40f268ab41640a0428edf30815b72244c5009b5e90  patches/port-exl3-glm53.py
ce6ea1b25ac93bd3ccea3347f4ced29cde52dd104e6ae9e63f59ad6da8df2d06  patches/port-dflash2-replicated-dcp.py
da60ca81fc9b3fedb568f458467004ced85ab6700a6193c9ca45fa8e98245f22  patches/port-b12x-glm-h64-query.py
```

## Verified historical receipts

These values were re-read from the raw JSON, not copied from the README:

| Receipt | SHA-256 | Verified value |
|---|---|---|
| `dflash2-fp8-code-agent-final.json` | `47d4500203d3d402ff967598c11bf00ee75ea1e0712a342b2dbef0ca463ae2cb` | C1 199.142 tok/s; C16 aggregate 800.940 tok/s |
| `dflash2-fp8-code-agent-k5-block128-validation.json` | `d45322c70dc3ddb7a9aafdd0c94dbd73d4a76912624d84e57e886d7c832d6b49` | C1 199.792 tok/s; C16 aggregate 838.278 tok/s |
| `dflash2-fp8-prefill-c1.json` | `694c3995e3cdc0b6afa38fdf6272735d43fe34af359a4cd215701b031ea0f530` | 32K 4,480.552 tok/s; 64K 4,428.706 tok/s; 128K 4,382.072 tok/s |
| `dflash2-fp8-glmrt-seven-blend-final.json` | `d403df80358d97a0418ae03b1687798d0a1dc8176563c522708703f90e81ef8f` | seven-type content receipt |
| `dflash2-fp8-1m-multi-needle-final.json` | `7ebd60dac0bb5909aa53de9be55afc40ba604875d36ad4c3fe1669efe89b0558` | six of six needles; 266.773 s request |
| `dflash2-fp8-vision.json` | `678e52510cb6da341d1c3c497c46683df89ddb253f4695a1057d3c0f3c01080f` | exact 16-image pass; image 17 rejected |

The v0.5 report records 4,286,464 physical target FP8 tokens and 2,786,881
full-request-equivalent scheduler tokens. It does not contain the complete
steady GPU/PCIe telemetry required for the new release matrix.

## Capture host and controlled baseline condition

- Ubuntu 26.04 LTS, kernel `7.0.0-29-generic`, NVIDIA driver `595.71.05`.
- Two RTX PRO 6000 Blackwell Workstation Edition GPUs, 97,887 MiB each.
- Hardware default/max power limit reported as 600 W per GPU.
- The configured and required comparison limit is 400 W per GPU, matching all
  previous published measurements for this recipe.
- GPU 0 had 86,263 MiB allocated by the unrelated `glrmt-coordinator`; GPU 1
  was free. No service was stopped or modified for this capture.

Before candidate selection, rerun release A at a verified 400 W per GPU with
the fixed code-agent seed/prompt and three steady samples each for 32K/64K
cold prefill, zero-depth C1/C16 pure decode, FP8 capacity, and synchronized
GPU/PCIe telemetry.
