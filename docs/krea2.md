# Krea 2

## Overview / 概要

This document describes the usage of the Krea 2 (K2) architecture within the Musubi Tuner framework. Krea 2 is a text-to-image generation model based on a single-stream MMDiT, using **Qwen3-VL-4B-Instruct** as the text encoder and the **Qwen-Image VAE** as the autoencoder.

Two DiT checkpoints exist: a **RAW** model (full-step, CFG-based) and a distilled **Turbo** model (few-step, CFG-free). The recommended LoRA workflow is to **train on the RAW model and run inference on the Turbo model** (see [Sample image generation during training](#sample-image-generation-during-training--学習中のサンプル画像生成) and [Inference](#inference--推論)).

This feature is experimental.

> **References:** Official inference code and repository: [krea-ai/krea-2](https://github.com/krea-ai/krea-2). Technical report: [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report).

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのKrea 2 (K2) アーキテクチャの使用法について説明しています。Krea 2はsingle-stream MMDiTをベースとしたテキストから画像を生成するモデルで、テキストエンコーダーに **Qwen3-VL-4B-Instruct**、Autoencoderに **Qwen-Image VAE** を使用します。

DiTのチェックポイントは2種類あります。**RAW** モデル（フルステップ、CFGあり）と、蒸留された **Turbo** モデル（少ステップ、CFGなし）です。推奨されるLoRAのワークフローは、**RAWモデルで学習し、Turboモデルで推論する**ことです（[学習中のサンプル画像生成](#sample-image-generation-during-training--学習中のサンプル画像生成) および [推論](#inference--推論) を参照）。

この機能は実験的なものです。

> **参考:** 公式の推論コードおよびリポジトリ: [krea-ai/krea-2](https://github.com/krea-ai/krea-2)。テクニカルレポート: [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

</details>

## Download the model / モデルのダウンロード

You need to prepare the following models:

- **DiT (RAW)**: The MMDiT transformer model, full-step version. Used for training.
- **DiT (Turbo)** *(optional)*: The distilled few-step version. Used for inference, and optionally for sample image generation during training.
- **VAE**: The **Qwen-Image VAE** (`*.safetensors`). This is the same VAE used by the Qwen-Image integration; if you already have it, you can reuse it.
- **Text Encoder**: **Qwen3-VL-4B-Instruct** as a single `*.safetensors` file (official or ComfyUI key layout). A safetensors **file** is expected here, not a HuggingFace directory. The Comfy-Org single-file weight (`text_encoders/qwen3vl_4b_bf16.safetensors`) is convenient and can be shared with ComfyUI.

The Krea 2 DiT weights are provided as single `*.safetensors` files in the official HuggingFace repositories ([RAW](https://huggingface.co/krea/Krea-2-Raw), [Turbo](https://huggingface.co/krea/Krea-2-Turbo)); each repo also contains a diffusers-compatible checkpoint.

| type | model | file |
|------|-------|------|
| DiT (RAW) | Krea 2 RAW | `raw.safetensors` from https://huggingface.co/krea/Krea-2-Raw |
| DiT (Turbo) | Krea 2 Turbo | `turbo.safetensors` from https://huggingface.co/krea/Krea-2-Turbo |
| VAE | Qwen-Image VAE | `split_files/vae/qwen_image_vae.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI |
| Text Encoder | Qwen3-VL-4B-Instruct | `text_encoders/qwen3vl_4b_bf16.safetensors` from https://huggingface.co/Comfy-Org/Qwen3-VL |

<details>
<summary>日本語</summary>

以下のモデルを準備してください：

- **DiT (RAW)**: MMDiT transformerモデルのフルステップ版。学習に使用します。
- **DiT (Turbo)** *（オプション）*: 蒸留された少ステップ版。推論、および任意で学習中のサンプル画像生成に使用します。
- **VAE**: **Qwen-Image VAE**（`*.safetensors`）。Qwen-Image統合で使用するVAEと同じものです。すでにお持ちであれば再利用できます。
- **Text Encoder**: **Qwen3-VL-4B-Instruct** の単一の `*.safetensors` ファイル（公式またはComfyUIのキー配置）。ここではHuggingFaceのディレクトリではなく、safetensors **ファイル** を指定します。Comfy-Orgの単一ファイル版（`text_encoders/qwen3vl_4b_bf16.safetensors`）が扱いやすく、ComfyUIと共用できます。

Krea 2のDiTの重みは、公式のHuggingFaceリポジトリ（[RAW](https://huggingface.co/krea/Krea-2-Raw)、[Turbo](https://huggingface.co/krea/Krea-2-Turbo)）に単一の `*.safetensors` ファイルとして提供されています（各リポジトリにはdiffusers互換のチェックポイントも含まれます）。ファイル一覧は英語版の表を参照してください: RAWは [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) の `raw.safetensors`、Turboは [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) の `turbo.safetensors` です。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Krea 2. The VAE and latent normalization are identical to Qwen-Image.

```bash
python src/musubi_tuner/krea2_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/qwen_image_vae
```

- Uses `krea2_cache_latents.py`.
- The `--vae` argument is required (Qwen-Image VAE).
- The dataset should be an image dataset. Krea 2 is plain text-to-image only (no control/edit images), so only target image latents are cached.

<details>
<summary>日本語</summary>

latentの事前キャッシングはKrea 2専用のスクリプトを使用します。VAEとlatentの正規化はQwen-Imageと同一です。

- `krea2_cache_latents.py`を使用します。
- `--vae`引数（Qwen-Image VAE）が必要です。
- データセットは画像データセットである必要があります。Krea 2はテキストから画像生成のみ（コントロール/編集画像なし）のため、ターゲット画像のlatentのみがキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script. Krea 2 caches the multi-layer hidden-state stack from Qwen3-VL; only valid (non-padding) tokens are stored (varlen).

```bash
python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/qwen3_vl_4b \
    --batch_size 1
```

- Uses `krea2_cache_text_encoder_outputs.py`.
- Requires the `--text_encoder` (Qwen3-VL-4B-Instruct) argument.
- Larger batch sizes require more VRAM. Adjust `--batch_size` according to your VRAM capacity.
- Pass `--multi_caption` to treat every non-empty line in directory caption files (CR, LF, or CRLF) as a separate alternative. During training one alternative is chosen deterministically from `--seed`, the epoch, and the sample position. Without this flag, the whole file remains one caption, including line breaks. JSONL captions remain single prompts. Re-run this cache step after editing caption files; `--skip_existing` validates the stored alternatives.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。Krea 2はQwen3-VLの複数レイヤーのhidden state stackをキャッシュし、有効な（パディングでない）トークンのみを保存します（varlen）。

- `krea2_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder`（Qwen3-VL-4B-Instruct）引数が必要です。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。
- `--multi_caption`を指定すると、ディレクトリのcaptionファイルの空でない各行（CR、LF、CRLF）が別々の候補としてエンコード・保存されます。学習時には`--seed`、エポック、サンプル位置から決定的に1つ選ばれます。このフラグなしでは、改行を含むファイル全体が1つのcaptionのままです。JSONLのcaptionは単一プロンプトのままです。captionファイルを変更した後はこのキャッシュ手順を再実行してください。`--skip_existing`は保存済み候補を検証します。

</details>

## Training / 学習

Training uses a dedicated script `krea2_train_network.py`. Train on the **RAW** DiT.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \
    --dit path/to/raw_dit_model \
    --vae path/to/qwen_image_vae \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.5 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `krea2_train_network.py`.
- **Requires** specifying `--dit` (RAW model) and `--vae` (Qwen-Image VAE).
- **Requires** specifying `--network_module networks.lora_krea2`.
- `--text_encoder` is needed if you generate sample images during training or enable `caption_dropout_rate` (otherwise it is not needed for the training step itself, because text encoder outputs are pre-cached).
- `caption_dropout_rate` is an optional dataset setting in the range `0.0` to `1.0` (default `0.0`). It can be set in `[general]` or overridden per `[[datasets]]`. On each item read, the normal caption is replaced with the empty caption with this probability. The empty-caption embedding is encoded once at training startup, kept in shared CPU memory, and is not written to a cache file; enabling dropout does not require re-running `krea2_cache_text_encoder_outputs.py`.

  ```toml
  [general]
  caption_dropout_rate = 0.1
  ```
- Krea 2 uses flow matching. `--timestep_sampling shift` with `--discrete_flow_shift` is a reasonable starting point. The value `2.5` matches the K2 inference time-shift at 1024×1024 (the schedule is resolution-aware: it ranges from about `1.6` at 256×256 to `3.2` at 1280×1280, reaching ~`2.5` at 1024×1024). For varying-resolution training, `--timestep_sampling krea2_shift` reproduces the same resolution-aware schedule per sample, so each timestep is shifted exactly as K2 shifts it at inference (default resolution range 256–1280); no fixed `--discrete_flow_shift` is needed in that case. (`--timestep_sampling flux_shift` is similar but its high end saturates at 1024px instead of 1280px, giving a slightly stronger shift above 256px.) The optimal settings are not yet established; feedback is welcome.
- `--network_dim` / `--network_alpha` of 32 reproduces the model authors' recommended default. See [LoRA target layers](#lora-target-layers--loraの対象レイヤー) below.

<details>
<summary>日本語</summary>

学習は専用のスクリプト`krea2_train_network.py`を使用します。**RAW** のDiTで学習します。コマンド例は英語版を参照してください。

- `krea2_train_network.py`を使用します。
- `--dit`（RAWモデル）と`--vae`（Qwen-Image VAE）を指定する必要があります。
- `--network_module networks.lora_krea2`を指定する必要があります。
- `--text_encoder`は学習中にサンプル画像を生成する場合、または`caption_dropout_rate`を有効にする場合に必要です（それ以外ではテキストエンコーダー出力が事前キャッシュされるため、学習ステップ自体には不要です）。
- `caption_dropout_rate`は`0.0`から`1.0`までのオプションのデータセット設定です（デフォルトは`0.0`）。`[general]`に設定するか、各`[[datasets]]`で上書きできます。各項目の読み込み時に、この確率で通常のcaptionを空のcaptionに置き換えます。空captionのembeddingは学習開始時に一度だけエンコードされ、共有CPUメモリに保持されます。cacheファイルには書き込まれないため、dropoutを有効にするだけなら`krea2_cache_text_encoder_outputs.py`を再実行する必要はありません。
- Krea 2はflow matchingを使用します。`--timestep_sampling shift`と`--discrete_flow_shift`の組み合わせが出発点として妥当です。値 `2.5` は1024×1024でのK2推論時のtime-shiftに一致します（このスケジュールは解像度依存で、256×256で約 `1.6`、1280×1280で約 `3.2`、1024×1024で約 `2.5` です）。解像度を変えて学習する場合は、`--timestep_sampling krea2_shift` を使うと同じ解像度依存スケジュールをサンプルごとに再現し、各タイムステップがK2推論時とまったく同じようにシフトされます（デフォルトの解像度レンジ256〜1280）。この場合は固定の `--discrete_flow_shift` は不要です。（`--timestep_sampling flux_shift` も類似ですが、高解像度側が1024px（K2は1280px）で飽和するため、256pxより上ではやや強いshiftになります。）最適な設定はまだ確立されていません。フィードバックをお待ちしています。
- `--network_dim` / `--network_alpha` を32にすると、モデル作者が推奨するデフォルト設定を再現します。下記の[LoRAの対象レイヤー](#lora-target-layers--loraの対象レイヤー)を参照してください。

</details>

### Alpha-masked attention

`--alpha_masked_token_drop` averages **all** alpha pixels, including zeros, in
each 16×16 image-token cell. This mean sets both attention alpha and the uniform
MSE weight of its four latent cells. Only wholly transparent cells are physically
removed. Mixed boundary cells now survive: this intentionally replaces the older
"any zero excludes the whole cell" rule. Original-alpha caches remain valid.
Without token-drop, the existing loss-mask behavior is unchanged. The default
`--alpha_masked_attention_mode native` preserves the existing behavior: every
positive-alpha token participates fully in attention.

For probabilistic source suppression, use:

```bash
--flash_attn --alpha_masked_token_drop --alpha_masked_attention_mode sharedkv
```

`sharedkv` retains the Q, prediction, and soft-weighted loss of every positive
token, but conditionally removes its K/V. Once per image presentation it draws
one `u ~ Uniform[0,1)` and retains a token's K/V when
`u < mean(alpha over its 16x16 cell)`. Thus equal-alpha regions switch together,
higher-alpha retained sets contain lower-threshold sets, alpha 0 is always
removed, and alpha 255 is always retained. The same decision is reused by all
DiT blocks and gradient-checkpoint recomputation. Text K/V is always retained;
training-time sample generation remains native.

This mode requires FlashAttention 2 and does not support `--split_attn`. It uses
fixed B×N storage, so the random retained K/V count does not become a tensor-shape
or token-bucketing coordinate. A missing alpha mask is treated as fully opaque.

For deterministic soft source suppression, use `logbias` and explicitly choose
its strength:

```bash
--flash_attn --alpha_masked_token_drop \
--alpha_masked_attention_mode logbias --alpha_masked_attention_gamma 3
```

For each retained image token, `logbias` adds
`gamma * log(mean(alpha over its 16x16 cell))` to that key's attention logit.
Text keys receive zero bias. Exact-zero cells are still physically removed; a
positive-alpha token keeps its Q, prediction, and soft-weighted MSE even when its
outgoing K/V influence is strongly reduced. Missing alpha is equivalent to alpha
255 and therefore gives zero bias. The same fixed bias is reused by every main
DiT block and gradient-checkpoint recomputation; text fusion and training-time
sample generation remain native.

`logbias` requires the optional `fa2-alpha` CUDA extension in addition to the
normal FlashAttention installation. The currently tested wheel is specialized
for Linux x86_64, CPython 3.12, PyTorch 2.10.0+cu130, CUDA SM120, BF16,
head_dim=128, GQA varlen attention, and no attention dropout. Install a wheel
compatible with the actual Python/PyTorch/CUDA/GPU combination before selecting
this mode. `native` and `sharedkv` do not depend on `fa2-alpha`.

### Inverted-mask output preservation

Add `--alpha_masked_output_preservation` to train with complementary masks. It
requires `--alpha_masked_token_drop`, standard `networks.lora_krea2`, and
`blocks_to_swap=0`. It is off by default. All three attention modes are supported;
their existing backend and gamma requirements still apply.

Use `--alpha_masked_output_preservation_loss_balance k` to rebalance the two loss
terms while keeping their coefficients' sum equal to 2:

```text
loss = (1+k) * loss_target + (1-k) * loss_preservation
```

The default is `k=0`, which gives both terms their original coefficient of 1.
`k` must be finite and in the closed interval `[-1, 1]`, keeping both loss
coefficients non-negative. The `loss_target` and `loss_preservation` metrics
remain unscaled.

For the same noisy latent, timestep and caption, compute the base-model teacher
(LoRA off, no gradient) with `1-a`, then the target student with `a` and the
preservation student with `1-a`:

```text
loss = mean_full_latent(a * (target_student - flow_target)^2
                     + (1-a) * (preservation_student - base_teacher)^2)
```

Here `a = mean(alpha/255)` per 16×16 tile; the inverse is exactly `1-a`.

Add `--alpha_masked_output_preservation_extremes_only` to invert only exact
alpha endpoints for the preservation branch. Before the 16×16 tile mean is
computed, `0` becomes `1`, `1` becomes `0`, and every value strictly between
them is left unchanged. The target branch continues to use the original alpha
mask. The option requires `--alpha_masked_output_preservation`; full `1-alpha`
inversion remains the default. Let `b` denote the resulting preservation token
weight; thus `b=1-a` by default.

The transformed preservation mask is also used for attention influence. With
this option, `sharedkv` applies the same per-image random threshold directly to
the target and preservation token weights. In `logbias`, the transformed token
weight supplies the log bias. `native` does not apply alpha attention masking.

There is no area renormalization or extra factor of one half. Both terms use the
same timestep weighting and dataset multiplier; the optional loss balance above
is the only extra scaling between branches.
Mixed-resolution losses are averaged per full image first, then across images.
The metrics `loss_target` and `loss_preservation` report the two terms separately.
Missing alpha means `a=1`, so no preservation passes are needed. Entire empty
branches are skipped; empty rows retain their weight in the batch average.

In default `sharedkv`, one shared uniform per image enables target K/V when
`u<a` and preservation K/V when `u>=a`. With the extremes-only option, the
preservation condition is instead the direct transformed-mask threshold `u<b`.
Teacher and preservation student reuse the exact same packed plan. `native`
keeps all positive-weight K/V; `logbias` uses `gamma*ln(a)` and `gamma*ln(b)`
respectively. Zero-weight Q/K/V are physically removed in each branch; text is
retained. The teacher is therefore a **masked** base prediction, not a
full-context inference prediction.

Plans are built outside compiled blocks and reused during checkpointing. LoRA
multipliers and dropout state are restored before either student forward. The
target student runs forward/backward before the preservation student runs
forward/backward, releasing the first graph before building the second. Both
backwards accumulate into the same LoRA gradients; clipping, gradient reduction
and the optimizer step happen afterward, respecting gradient accumulation. The
teacher is computed once without gradients. This avoids retaining two student
graphs alongside optimizer state. Compile/checkpointing are supported, but
scaled-FP8/BF16 Inductor gradients can differ substantially from eager, including
without preservation. `--compile_backend aot_eager` provides a closer numerical
reference. These differences alone do not establish a semantic or visual change;
training-quality validation remains outstanding. See the
[validation report](krea2_preservation_validation.md). Extra teacher/student graphs
and extra computation/memory are expected; there is
no fixed overhead estimate. Sampling is unchanged. Bucketing still uses target-Q
counts only; dry-bucketing additionally reports inverse lengths and padding.

### Image-token bucketing

`--image_token_bucketing` groups Krea 2 images by the number of image tokens retained for DiT, even when their original resolutions differ. The width of each length group is `256 * --image_token_bucket_multiple` tokens (default multiplier: 1). Short group remainders move to the next smaller group. Every image is used once per epoch; if the dataset item count (including repeats) is not divisible by its microbatch size, training stops with an assertion instead of creating a partial microbatch. Each dataset is grouped separately.

With `--alpha_masked_token_drop`, keep counts use the same 16×16 alpha alignment as training. Without it, all image tokens count. Caption length is not included in the grouping. The original image geometry still determines resize/crop, RoPE positions, resolution-aware timesteps, and the full-size denominator of each image's loss.

In `sharedkv` and `logbias` modes, bucketing still uses the number of Q tokens
remaining after exact-zero token drop. The random number of enabled K/V in
`sharedkv`, and the bias values in `logbias`, do not change batch membership or
allocated sequence length.

Preview the grouping without loading model weights or starting training:

```bash
python src/musubi_tuner/krea2_train_network.py \
  --dataset_config /workspace/dataset.toml \
  --alpha_masked_token_drop --dry-bucketing
```

`--dry-bucketing` implies `--image_token_bucketing` and prints the image-only padding estimate. Run training with `--image_token_bucketing` instead of `--dry-bucketing`. These options also work in a TOML training config.

<details>
<summary>日本語</summary>

`--image_token_bucketing` は、元の解像度が異なっても、DiTに残る画像トークン数でKrea 2のマイクロバッチを組みます。区間幅は `256 * --image_token_bucket_multiple`（デフォルト倍率1）です。端数は次の小さい区間に移し、各画像を1エポックに1回使います。データセットの項目数（繰り返しを含む）がバッチサイズで割り切れない場合はエラーになります。`--alpha_masked_token_drop` を指定すると16×16に整列したalphaから残存トークン数を計算します。`--dry-bucketing` は重みを読み込まずにバッチ構成と画像トークンのpaddingを表示します。

</details>

### LoRA target layers / LoRAの対象レイヤー

By default, the Krea 2 LoRA targets **all Linear layers** in the DiT (264 layers: attention, MLP, the text-fusion transformer, and the projection MLPs). This matches the model authors' recommended default configuration (rank/alpha 32). The modulation and RMSNorm parameters are raw tensors (not Linear modules), so they are never wrapped — no exclusion is needed.

Because the default already targets everything, both `exclude_patterns` and `include_patterns` are free for you to narrow the target set, passed via `--network_args`:

- **Attention-only** (the authors' "long training run" config — increase rank and focus on the attention projections to preserve prompt adherence):

  ```bash
  --network_args "exclude_patterns=['.*\.mlp\..*','first','last\.linear','tmlp\..*','txtmlp\..*','tproj\.1','txtfusion\..*']"
  ```

  This keeps only the per-block attention projections (`wq`/`wk`/`wv`/`wo`/`gate`, 140 Linears).

- **Arbitrary subset**: drop everything with `exclude_patterns=['.*']`, then add back the layers you want with `include_patterns=[...]`.

<details>
<summary>日本語</summary>

デフォルトでは、Krea 2のLoRAはDiTの **すべてのLinear層**（264層：attention、MLP、text-fusion transformer、projection MLP）を対象とします。これはモデル作者が推奨するデフォルト設定（rank/alpha 32）に一致します。modulationとRMSNormのパラメータは生のテンソル（Linearモジュールではない）なので、対象に含まれません。除外指定は不要です。

デフォルトですべてを対象とするため、`exclude_patterns`と`include_patterns`の両方を対象の絞り込みに自由に使えます。`--network_args`で指定します。

- **Attentionのみ**（作者の「長時間学習」設定。rankを上げてattention projectionに集中し、プロンプト追従性を保つ）: コマンドは英語版を参照。per-blockのattention projection（`wq`/`wk`/`wv`/`wo`/`gate`、140 Linear）のみを残します。
- **任意のサブセット**: `exclude_patterns=['.*']`ですべてを除外し、`include_patterns=[...]`で必要な層を戻します。

</details>

### Memory Optimization / メモリ最適化

- `--fp8_base` and `--fp8_scaled` reduce DiT memory usage. **Both must be specified together** (plain fp8 without scaled is rejected, because it would cast the norms to fp8 and break the model). fp8 is applied to the 28 main blocks only; the text-fusion transformer stays bf16.
- `--blocks_to_swap N` offloads some of the main blocks to CPU. The maximum is **26** (28 blocks − 2).
- `--gradient_checkpointing` is available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

- `--fp8_base`と`--fp8_scaled`でDiTのメモリ使用量を削減します。**両方を同時に指定する必要があります**（scaledなしのplain fp8は、normをfp8にキャストしてモデルを壊すため拒否されます）。fp8は28個のメインブロックのみに適用され、text-fusion transformerはbf16のまま保持されます。
- `--blocks_to_swap N`で一部のメインブロックをCPUにオフロードします。最大値は **26**（28ブロック − 2）です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

</details>

### Attention / Attention

- `--sdpa` for PyTorch's scaled dot product attention (default, no extra dependencies).
- `--flash_attn` for FlashAttention.
- `--sage_attn` for SageAttention.
- `--xformers` for xformers.
- `--split_attn` processes attention in chunks, reducing VRAM usage slightly. Recommended when using any backend other than `--sdpa`.

Krea 2 uses Grouped-Query Attention (48 query heads / 12 key-value heads). SDPA, FlashAttention, and SageAttention support this natively; for xformers the key/value heads are expanded to match (numerically identical).

<details>
<summary>日本語</summary>

- `--sdpa`でPyTorchのscaled dot product attentionを使用（デフォルト、追加の依存ライブラリ不要）。
- `--flash_attn`でFlashAttentionを使用。
- `--sage_attn`でSageAttentionを使用。
- `--xformers`でxformersを使用。
- `--split_attn`を指定すると、attentionを分割して処理し、VRAM使用量をわずかに減らします。`--sdpa`以外のバックエンドを使う場合は指定を推奨します。

Krea 2はGrouped-Query Attention（48 query head / 12 key-value head）を使用します。SDPA、FlashAttention、SageAttentionはこれをネイティブにサポートしており、xformersではkey/value headを拡張して一致させます（数値的に同一）。

</details>

### torch.compile / torch.compile

`--compile` compiles the 28 main SingleStreamBlocks (the heavy, repeated compute and the fp8/LoRA target) for faster training. See [torch.compile documentation](./torch_compile.md). It composes with fp8 and block swap.

<details>
<summary>日本語</summary>

`--compile`で28個のメインSingleStreamBlock（重く繰り返される計算であり、fp8/LoRAの対象）をコンパイルし、学習を高速化します。[torch.compileのドキュメント](./torch_compile.md)を参照してください。fp8やblock swapと併用できます。

</details>

### Sample image generation during training / 学習中のサンプル画像生成

To generate sample images during training, specify `--text_encoder` (Qwen3-VL-4B-Instruct) and the usual `--sample_prompts` / `--sample_every_n_epochs` options. See the [sampling during training documentation](./sampling_during_training.md) for the prompt file format.

By default, samples are generated on the RAW model being trained, using CFG (specify a negative prompt and a CFG scale via `--l` in the sample prompt; CFG-off output is blurry, which is expected for the K2 RAW model).

**Recommended: generate samples on the Turbo model.** Since the recommended workflow is RAW-train → Turbo-infer, you can preview results closer to actual use by sampling on the distilled Turbo model. Pass `--turbo_dit path/to/turbo_dit`. The trained LoRA is applied on top of the Turbo weights automatically (no second network, no merge). When `--turbo_dit` is set, the Turbo schedule is used (fixed `mu = 1.15`); in your sample prompt set CFG off and a low step count, e.g. `--l 1 --s 8`.

```text
A fox in the snow.  --w 1024 --h 1024 --s 8 --l 1 --d 0
```

`--turbo_dit` has two memory modes:

- **Default (streaming)**: the Turbo weights are loaded from disk for each sampling step (re-quantized if fp8) — roughly **no extra steady CPU RAM**, at the cost of per-sample load time.
- **`--turbo_dit_cache` (resident)**: the Turbo weights are quantized once at startup and kept resident in CPU RAM, swapped in for each sample — **faster**. Without block swap, the Turbo copy plus the RAW restore snapshot use roughly **2× the DiT size in extra CPU RAM**. With H2D-only block swap, streamed RAW weights reuse the existing offloader master, so the extra cost is one Turbo copy plus only the resident RAW fraction (between roughly 1× and 2×, depending on `blocks_to_swap`).

> **Block-swap exception:** `--turbo_dit` can be combined with `--blocks_to_swap` only when both `--turbo_dit_cache` and `--block_swap_h2d_only` are enabled. The H2D-only offloader keeps separate RAW/Turbo CPU master banks and invalidates its GPU ring when switching. Classic block swap and the disk-streaming Turbo mode are not compatible with this weight-bank switch.

<details>
<summary>日本語</summary>

学習中にサンプル画像を生成するには、`--text_encoder`（Qwen3-VL-4B-Instruct）と通常の`--sample_prompts` / `--sample_every_n_epochs`オプションを指定します。プロンプトファイルの形式は[学習中のサンプル生成ドキュメント](./sampling_during_training.md)を参照してください。

デフォルトでは、学習中のRAWモデルでCFGを使用してサンプルが生成されます（サンプルプロンプトでネガティブプロンプトと`--l`によるCFGスケールを指定してください。CFGなしの出力はぼやけますが、これはK2 RAWモデルでは想定通りです）。

**推奨: Turboモデルでサンプルを生成する。** 推奨ワークフローがRAW学習→Turbo推論であるため、蒸留されたTurboモデルでサンプリングすると、実利用に近い結果をプレビューできます。`--turbo_dit path/to/turbo_dit`を指定します。学習中のLoRAは自動的にTurboの重みの上に適用されます（2つ目のネットワークもマージも不要）。`--turbo_dit`指定時はTurboのスケジュール（固定`mu = 1.15`）が使われます。サンプルプロンプトではCFGをオフにし、少ないステップ数を指定してください（例: `--l 1 --s 8`）。

`--turbo_dit`には2つのメモリモードがあります。

- **デフォルト（ストリーミング）**: Turboの重みをサンプリングステップごとにディスクから読み込みます（fp8の場合は再量子化）。**定常のCPU RAM増加はほぼゼロ**ですが、サンプルごとの読み込み時間がかかります。
- **`--turbo_dit_cache`（常駐）**: Turboの重みを起動時に一度量子化してCPU RAMに常駐させ、サンプルごとにスワップインするため**高速**です。block swapなしでは、TurboコピーとRAW復元用スナップショットを合わせて **DiTサイズの約2倍** のCPU RAMを追加で使用します。H2D-only block swapでは、streamed RAW weightsは既存のoffloader masterを再利用するため、追加コストはTurbo 1コピーとresident RAW部分だけです（`blocks_to_swap`に応じて概ね1〜2倍）。

> **block swapの例外:** `--turbo_dit`と`--blocks_to_swap`を併用できるのは、`--turbo_dit_cache`と`--block_swap_h2d_only`も両方有効な場合だけです。H2D-only offloaderはRAW/TurboそれぞれのCPUマスターバンクを保持し、切り替え時にGPUリングを無効化します。通常のblock swapおよびディスクから毎回読み込むTurboモードは、このウェイトバンク切り替えと互換性がありません。

</details>

## Inference / 推論

Inference uses a dedicated script `krea2_generate_image.py`. For the recommended workflow, run inference on the **Turbo** model.

**RAW model inference:**

```bash
python src/musubi_tuner/krea2_generate_image.py \
    "A fox in the snow." \
    --dit path/to/raw_dit_model \
    --vae path/to/qwen_image_vae \
    --text_encoder path/to/qwen3_vl_4b \
    --steps 28 --guidance_scale 5.5 \
    --width 1024 --height 1024 \
    --attn_mode torch \
    --seed 0 --save_path path/to/save/dir \
    --lora_weight path/to/lora.safetensors --lora_multiplier 1.0
```

**Turbo model inference (recommended):**

```bash
python src/musubi_tuner/krea2_generate_image.py \
    "A fox in the snow." \
    --dit path/to/turbo_dit_model \
    --vae path/to/qwen_image_vae \
    --text_encoder path/to/qwen3_vl_4b \
    --steps 8 --guidance_scale 1 --mu 1.15 \
    --width 1024 --height 1024 \
    --attn_mode torch \
    --seed 0 --save_path path/to/save/dir \
    --lora_weight path/to/lora.safetensors --lora_multiplier 1.0
```

- Uses `krea2_generate_image.py`. There are three input modes: a single positional prompt (above), `--from_file <path>` (one prompt per line), and `--interactive` (read prompts from the console). Exactly one must be given.
  - In `--from_file` / `--interactive`, each line may carry per-prompt overrides as `--<opt> <value>` after the prompt text: `--w` (width), `--h` (height), `--s` (steps), `--d` (seed), `--g` / `--l` (guidance_scale), `--n` (negative prompt), `--y1`, `--y2`, `--mu`, `--i` (num images). Example: `A fox in the snow. --w 1280 --h 768 --s 8 --l 1 --d 0`. Blank lines and lines starting with `#` are skipped. Anything not overridden falls back to the command-line value. `--bell` rings the terminal bell after each prompt (interactive) or at the end.
- **Requires** `--dit`, `--vae` (Qwen-Image VAE), and `--text_encoder` (Qwen3-VL-4B-Instruct).
- `--steps` defaults to 28 (use ~8 for Turbo).
- `--guidance_scale` is the classifier-free guidance scale, default 5.5. Values `<= 1` disable CFG (use `--guidance_scale 1` for the Turbo model; no negative prompt needed). `--negative_prompt` is used only when `--guidance_scale > 1`. Krea 2 uses the standard CFG form `uncond + scale * (cond - uncond)`, like the other architectures here. **Note:** the official Krea 2 reference uses a "guidance" value with a different baseline (`uncond` at `0`), related by `guidance_scale = guidance + 1` — so the official default `guidance 4.5` corresponds to `--guidance_scale 5.5`.
- Timestep-shift `mu`: by default it is resolution-aware (interpolated between `--y1` at min resolution and `--y2` at max). For the Turbo model, pin a constant with `--mu 1.15`.
- `--attn_mode` selects the attention backend: `torch` (SDPA, default), `flash`, `sageattn`, `xformers`. Add `--split_attn` for the non-sdpa backends (required for `xformers` with GQA).
- `--width` / `--height` default to 1024. `--num-images` generates multiple images (image *i* uses `--seed` + *i*).
- `--save_path` is a **directory** (required, created if missing); file names are auto-generated as `<timestamp>_<seed>.png`.
- `--seed` is the base seed; image *i* uses `--seed` + *i*. **If omitted, a random seed is used** (logged, and reflected in the file name). In `--from_file` / `--interactive`, prompts without a per-line `--d` each draw a fresh random seed.
- LoRA loading: `--lora_weight` (one or more) and `--lora_multiplier`. LoRA is merged into the base DiT weights at load time (the only correct route under fp8).
- **Memory-efficient inference (fits a 24GB card):**
  - `--fp8_scaled` quantizes the 28 main blocks to dynamic scaled fp8 at load time (K2 supports only scaled fp8; the text-fusion transformer stays bf16). Roughly halves DiT weight memory.
  - `--blocks_to_swap N` offloads `N` of the main blocks to CPU (max **26** = 28 − 2), trading speed for VRAM. Composes with `--fp8_scaled`.
  - `--use_pinned_memory_for_block_swap` uses pinned host memory for faster H2D copies (more host RAM).
  - `--block_swap_h2d_only` streams blocks Host→Device only (keeping a CPU master, no device→host copy), which is always safe at inference since the base weights are frozen. `--block_swap_ring_size N` sets the number of GPU ring buffers (2 = transfer/compute overlap, 1 = minimal memory).
  - **Memory model:** the DiT stays resident on the GPU (with block swap as needed) for the whole run, while the text encoder (Qwen3-VL-4B, ~8GB) and the VAE shuttle between CPU and GPU. The encoder is kept on CPU and moved to the GPU only to encode each prompt; the VAE is kept on CPU and moved to the GPU only to decode, then moved back. So the headroom for encoding/decoding comes from running the DiT under `--fp8_scaled` and/or `--blocks_to_swap` — not from evacuating the ~24GB DiT to host RAM. (Block swap and/or fp8 is therefore effectively required to fit a 24GB card; without it the DiT alone leaves no room for the decode.)
  - `--text_encoder_cpu` encodes prompts on CPU instead of moving the encoder to the GPU. Use it when the encoder (on the GPU, alongside the resident DiT) does not fit; it is slower but keeps the encode off the GPU.

<details>
<summary>日本語</summary>

推論は専用のスクリプト`krea2_generate_image.py`を使用します。推奨ワークフローでは **Turbo** モデルで推論します。コマンド例は英語版を参照してください。

- `krea2_generate_image.py`を使用します。入力モードは3種類あります: 単一の位置引数プロンプト（上記）、`--from_file <path>`（1行1プロンプト）、`--interactive`（コンソールから入力）。いずれか1つを指定します。
  - `--from_file` / `--interactive`では、各行のプロンプト本文の後ろに`--<opt> <value>`形式でプロンプトごとの上書きを記述できます: `--w`（幅）、`--h`（高さ）、`--s`（ステップ数）、`--d`（シード）、`--g` / `--l`（guidance_scale）、`--n`（ネガティブプロンプト）、`--y1`、`--y2`、`--mu`、`--i`（生成枚数）。例: `A fox in the snow. --w 1280 --h 768 --s 8 --l 1 --d 0`。空行と`#`で始まる行はスキップされます。上書きされなかった項目はコマンドライン引数の値が使われます。`--bell`で各プロンプト後（対話モード）または最後（その他のモード）に端末のベルを鳴らします。
- `--dit`、`--vae`（Qwen-Image VAE）、`--text_encoder`（Qwen3-VL-4B-Instruct）が必要です。
- `--steps`のデフォルトは28です（Turboでは約8）。
- `--guidance_scale`はclassifier-freeガイダンスのスケールで、デフォルトは5.5です。`<= 1` でCFGを無効化します（Turboモデルでは`--guidance_scale 1`。ネガティブプロンプト不要）。`--negative_prompt`は`--guidance_scale > 1`のときのみ使用されます。Krea 2は他のアーキテクチャと同じく標準のCFG式 `uncond + scale * (cond - uncond)` を使用します。**注意:** 公式のKrea 2 referenceは基準点の異なる「guidance」値（`0` で `uncond`）を使っており、関係式は `guidance_scale = guidance + 1` です。したがって公式デフォルトの `guidance 4.5` は `--guidance_scale 5.5` に対応します。
- Timestep-shiftの`mu`: デフォルトは解像度依存（最小解像度の`--y1`と最大解像度の`--y2`の間で補間）です。Turboモデルでは`--mu 1.15`で固定値を指定してください。
- `--attn_mode`でattentionバックエンドを選択します: `torch`（SDPA、デフォルト）、`flash`、`sageattn`、`xformers`。sdpa以外のバックエンドでは`--split_attn`を追加してください（`xformers` + GQAでは必須）。
- `--width` / `--height`のデフォルトは1024です。`--num-images`で複数画像を生成します（画像 *i* は`--seed` + *i* を使用）。
- `--save_path`は保存先の**ディレクトリ**です（必須、なければ作成）。ファイル名は`<タイムスタンプ>_<seed>.png`の形式で自動生成されます。
- `--seed`はベースシードで、画像 *i* は`--seed` + *i* を使用します。**省略した場合はランダムなシードが使われます**（ログに出力され、ファイル名にも反映されます）。`--from_file` / `--interactive`では、行ごとの`--d`がないプロンプトはそれぞれ新しいランダムシードを引きます。
- LoRAの読み込み: `--lora_weight`（1つ以上）と`--lora_multiplier`。LoRAは読み込み時にベースのDiT重みにマージされます（fp8でも正しく動作する唯一の方法）。
- **省メモリ推論（24GBに収まります）:**
  - `--fp8_scaled`で28個のメインブロックを読み込み時に動的スケールfp8に量子化します（K2はscaled fp8のみ対応。text-fusion transformerはbf16のまま）。DiTの重みメモリがおよそ半減します。
  - `--blocks_to_swap N`で`N`個のメインブロックをCPUにオフロードします（最大 **26** = 28 − 2）。速度と引き換えにVRAMを削減します。`--fp8_scaled`と併用できます。
  - `--use_pinned_memory_for_block_swap`でpinnedホストメモリを使い、H2Dコピーを高速化します（ホストRAMを多く使用）。
  - `--block_swap_h2d_only`はブロックをHost→Deviceのみでストリーミングします（CPUマスターを保持し、device→hostコピーを行わない）。推論ではベース重みが凍結されているため常に安全です。`--block_swap_ring_size N`でGPUリングバッファ数を設定します（2で転送と計算をオーバーラップ、1で最小メモリ）。
  - **メモリモデル:** DiTは実行中ずっとGPUに常駐させ（必要に応じてblock swap）、テキストエンコーダ（Qwen3-VL-4B、約8GB）とVAEがCPUとGPUを行き来します。エンコーダはCPUに置き、各プロンプトのエンコード時のみGPUへ移動します。VAEもCPUに置き、デコード時のみGPUへ移動してから戻します。したがってエンコード/デコードのためのVRAMの余裕は、`--fp8_scaled`や`--blocks_to_swap`でDiTを動かすことから生まれます（約24GBのDiTをホストRAMへ退避させるのではありません）。このため24GBに収めるにはblock swapおよび/またはfp8が実質必須です（使わない場合、DiTだけでデコードの余地が残りません）。
  - `--text_encoder_cpu`はエンコーダをGPUに移動せずCPUでエンコードします。常駐DiTと並んでエンコーダがGPUに載りきらない場合に使用します。低速ですがエンコードをGPUの外に出せます。

</details>
