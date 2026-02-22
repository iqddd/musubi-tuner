# Pipeline Artifact: FLUX.2 train launch via `accelerate` + `/workspace/config.toml`

## Команда запуска

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/flux_2_train_network.py \
  --config_file /workspace/config.toml
```

## Результат тестового запуска (факт)

- Дата/время запуска: `2026-02-22 09:18 UTC`
- Запуск выполнен этой же командой из активированного `.venv`
- Успешно начался train-loop на `accelerator device: cuda`
- Достигнут прогресс: `steps: 10/312`
- После достижения `10` шагов процесс остановлен вручную (`Ctrl+C`)
- Event-файл TensorBoard создан:
  - `/workspace/logs/20260222091811/network_train/events.out.tfevents.1771751898.225c70ef3e07.10786.0`
- Чекпоинты в `/workspace/output` не появились, т.к. прерывание выполнено до конца 1-й эпохи (сохранение по `save_every_n_epochs=1`)

## Фактическая проверка dtype (runtime)

Проверка сделана отдельным пробным запуском инициализации через `accelerate` с тем же конфигом:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  speedup_task/inspect_runtime_dtypes_flux2.py \
  --config_file /workspace/config.toml
```

Итог (факт из runtime-отчета):

- `transformer` (DiT), замороженные веса:
  - `201` тензор
  - `9,078,581,248` параметров
  - dtype: только `torch.bfloat16`
- `network` (LoRA), обучаемые веса:
  - `224` тензора
  - `82,837,504` параметров
  - dtype: только `torch.float32`
- `optimizer` param groups:
  - `224` тензора
  - `82,837,504` параметров
  - dtype: только `torch.float32`
- До и после `accelerator.prepare(...)` распределение dtype не изменилось.

Вывод: в вашем текущем запуске trainable LoRA-параметры и параметры оптимизатора реально идут в `float32`, а замороженные DiT-веса находятся в `bfloat16`.

## Что именно используется из окружения

- Python из активированного `.venv` проекта: `.venv/bin/python`
- `accelerate`: 1.6.0
- `torch`: 2.10.0+cu130
- `flash_attn`: установлен в `.venv`

Примечание: в sandbox-режиме (без эскалации) этот запуск падал на доступе к `/dev/shm`, но фактический тестовый прогон выше выполнен в unrestricted-режиме и дошел до `10/312` на `cuda`.

## Effective-параметры (после чтения `--config_file`)

`/workspace/config.toml` читается и "сплющивается": секции `[general]`, `[logging]`, `[saves]`, `[fp8]`, `[timestamps]` становятся единым набором CLI-аргументов.

Ключевые значения для этого запуска:

- `model_version=klein-base-9b`
- `dit=/workspace/models/flux-2-klein-base-9b.safetensors`
- `vae=/workspace/models/ae.safetensors`
- `text_encoder=/workspace/models/model-00001-of-00004.safetensors`
- `dataset_config=/workspace/dataset.toml`
- `flash_attn=true`
- `mixed_precision=bf16`
- `optimizer_type=prodigyplus.ProdigyPlusScheduleFree`
- `learning_rate=1.0`
- `gradient_checkpointing=true`
- `max_data_loader_n_workers=2`
- `persistent_data_loader_workers=true`
- `network_module=networks.lora_flux_2`
- `network_dim=32`, `network_alpha=32`
- `max_train_epochs=13`
- `seed=420`
- `output_dir=/workspace/output/`, `output_name=diamel`
- `sample_every_n_epochs=1`
- `sample_prompts=/workspace/prompts.txt`
- `max_grad_norm=0.0` (клиппинг отключен)
- `save_every_n_epochs=1`, `save_state=true`, `save_last_n_epochs_state=2`
- `weighting_scheme=none`
- `timestep_sampling=qinglong_flux`
- `max_timestep=900`, `preserve_distribution_shape=true`
- `fp8_base=false`, `fp8_scaled=false`

Параметры по умолчанию, которые остаются активными:

- `gradient_accumulation_steps=1`
- `discrete_flow_shift=1.0` (присутствует в args/metadata, но при `timestep_sampling=qinglong_flux` не является основным драйвером train-сэмплинга)
- `vae_dtype=float32`
- `sample_every_n_steps=None`

## Пайплайн выполнения

### 1) `accelerate launch` подготавливает процесс

`accelerate`:

- формирует команду дочернего процесса `python src/musubi_tuner/flux_2_train_network.py --config_file /workspace/config.toml`
- выставляет переменные окружения, включая:
  - `ACCELERATE_MIXED_PRECISION=bf16`
  - `OMP_NUM_THREADS=1` (из `--num_cpu_threads_per_process 1`)
- выбирает режим запуска (single-process или multi-process) по доступным устройствам и конфигурации `accelerate`

### 2) Вход в `flux_2_train_network.py`

- создается общий parser (`setup_parser_common`) + FLUX.2-специфичные аргументы (`flux2_setup_parser`)
- сначала читаются CLI-аргументы
- затем вызывается `read_config_from_file`, который загружает `/workspace/config.toml` и накладывает значения на parser
- инициализируется `Flux2NetworkTrainer`, далее `trainer.train(args)`

### 3) Валидация и подготовка тренера

В `train()`:

- проверяются обязательные аргументы (`dataset_config`, `dit`)
- FLUX.2-специфика:
  - `model_version=klein-base-9b` -> архитектура `f2k9b` / `flux_2_klein_9b`
  - `dit_dtype` для модели ставится в `torch.bfloat16` (из `mixed_precision=bf16`)
  - дефолтный guidance для sampling = 4.0

### 4) Загрузка датасета из кэшей

Берется `/workspace/dataset.toml`:

```toml
[general]
resolution = [832, 832]
caption_extension = ".txt"
batch_size = 5
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/workspace/dataset/images"
cache_directory = "/workspace/dataset/caches"
```

Что происходит:

- строится `Blueprint` датасета
- для FLUX.2-image training читаются только кэши:
  - latent: `*_f2k9b.safetensors`
  - text encoder: `*_f2k9b_te.safetensors`
- элементы группируются в bucket-ы по разрешениям
- реальный тензорный батч формируется внутри `BucketBatchManager` с `batch_size=5`
- внешний `DataLoader` использует `batch_size=1`, а collate-функция разворачивает подготовленный batch

Для текущих данных в `/workspace/dataset/caches`:

- найдено 120 валидных latent+text-cache пар
- при `batch_size=5` это 24 батча на эпоху

### 5) Инициализация `Accelerator` и precision

`prepare_accelerator(args)`:

- запускает `Accelerator(gradient_accumulation_steps=1, mixed_precision='bf16', log_with='tensorboard', project_dir=<logging_dir/timestamp>)`
- логи пишутся в `/workspace/logs/<timestamp>/`

DType-расклад:

- `weight_dtype = bfloat16` (mixed precision)
- `dit_dtype = bfloat16`
- `vae_dtype = float32`
- LoRA train-параметры тренируются в обычной схеме (не `full_bf16`)

### 6) Подготовка sample-промптов

Так как задан `sample_prompts=/workspace/prompts.txt`:

- файл промптов читается до начала тренировки (3 промпта)
- загружается FLUX.2 text encoder (`/workspace/models/model-00001-of-00004.safetensors`)
- для промптов кэшируются `ctx_vec` и `negative_ctx_vec`
- загружается VAE (`/workspace/models/ae.safetensors`) для последующего sample-инференса

### 7) Загрузка DiT и подключение LoRA

- выбирается attention mode `flash` (из `flash_attn=true`)
- грузится DiT: `/workspace/models/flux-2-klein-base-9b.safetensors`
- `blocks_to_swap=0` -> offload блоков не включается
- импортируется `network_module=networks.lora_flux_2`
- создается LoRA-сеть с rank/alpha = `32/32`
- LoRA применяется к transformer (`apply_to(... apply_unet=True)`)

### 8) Оптимизатор, scheduler, dataloader

- trainable params берутся из LoRA
- создается `prodigyplus.ProdigyPlusScheduleFree` с `learning_rate=1.0`
- так как optimizer schedule-free, используется dummy LR scheduler для совместимости логирования
- dataloader: `num_workers=2`, `persistent_workers=True`

### 9) Таймстепы и loss в train-loop

На каждом шаге:

- из батча берутся `latents` и `ctx_vec`
- генерируется шум
- семплируются timesteps через `timestep_sampling=qinglong_flux`
  - с ограничением диапазона до `max_timestep=900`
  - с `preserve_distribution_shape=true` (rejection sampling, чтобы не искажать форму распределения)
- формируется `noisy_model_input`
- DiT вызывается через FLUX.2 путь (`call_dit`), таргет: `noise - latents`
- loss: `mse(model_pred, target)`; при `weighting_scheme=none` без дополнительного весового множителя
- backward + optimizer step
- `max_grad_norm=0.0` -> gradient clipping не применяется

Уточнение: в этой конфигурации train-сэмплинг таймстепов задается веткой `qinglong_flux` (mid-shift/logsnr mix), а не прямым `args.discrete_flow_shift`.

### 10) Сэмплы, чекпоинты, state

В конце каждой эпохи (`sample_every_n_epochs=1`, `save_every_n_epochs=1`):

- генерируются sample-изображения по 3 промптам в:
  - `/workspace/output/sample/`
- сохраняется LoRA checkpoint:
  - `/workspace/output/diamel-000001.safetensors`, ..., `/workspace/output/diamel-000012.safetensors`
  - финальный: `/workspace/output/diamel.safetensors`
- сохраняется training state (`save_state=true`):
  - `/workspace/output/diamel-000001-state/` и т.д.
  - хранится последние 2 state-директории по `save_last_n_epochs_state=2`
  - в конце обучения дополнительно сохраняется финальный state: `/workspace/output/diamel-state/`

Отдельно про sample-инференс: для FLUX.2 в `Flux2NetworkTrainer` выставлен `default_discrete_flow_shift=None`, поэтому при отсутствии `--fs` в строке промпта schedule строится через `flux2_utils.get_schedule(..., flow_shift=None)`, то есть через `compute_empirical_mu(...)`. Явный `--fs` в промпте переопределит это поведение.

Важно: для checkpoint-файлов в этом конфиге ограничение по количеству эпоховых `.safetensors` не задано (`save_last_n_epochs` не указан), поэтому они не удаляются автоматически.

### 11) Расчет ожидаемого числа шагов

Формула из кода:

```text
max_train_steps = max_train_epochs * ceil(len(train_dataloader) / num_processes / gradient_accumulation_steps)
```

Для текущих данных и single-process (`num_processes=1`):

- `len(train_dataloader)=24`
- `max_train_epochs=13`
- `gradient_accumulation_steps=1`
- ожидаемо: `13 * 24 = 312` оптимизационных шагов

Если `accelerate` поднимет multi-process, число шагов на эпоху будет делиться на `num_processes` по формуле выше.

## Критические точки отказа

- Нет GPU / недоступна CUDA: запуск FLUX.2 практично неработоспособен в CPU-only режиме
- Нет/битые кэши `*_f2k9b.safetensors` и `*_f2k9b_te.safetensors` -> `No training items found`
- Нет `flash_attn` при `flash_attn=true` -> ошибка загрузки/выполнения attention
- Несовпадение `model_version` и подготовленных кэшей/весов
- Ограниченный sandbox без прав на `/dev/shm` -> падение на `multiprocessing.Value(...)` с `PermissionError`

## Краткий итог

Команда запускает FLUX.2 LoRA-training (klein-base-9b) из уже предкэшированного image-dataset, с bf16 mixed precision, schedule-free оптимизатором ProdigyPlus, qinglong_flux timestep sampling (ограниченным до 900), эпоховыми sample-рендерами и чекпоинтами в `/workspace/output`.

## Добавленный тестовый стенд профайлинга (N-й шаг + стоп на M-м)

В код тренировки добавлены новые CLI-аргументы в `setup_parser_common` (файл `src/musubi_tuner/hv_train_network.py`):

- `--profile_capture_step N` — записать ровно один оптимизационный шаг `N`
- `--profile_stop_step M` — принудительно завершить прогон на шаге `M` (внутри кода `max_train_steps` ограничивается до `M`)
- `--profile_with_torch` — запись `torch.profiler` только для шага `N`
- `--profile_with_cuda_profiler_api` — вызов `cudaProfilerStart/Stop` вокруг шага `N` (для `nsys`/`ncu` с capture-range API)
- `--profile_artifacts_dir` — базовая директория артефактов
- `--profile_disable_sampling` — принудительно отключает `sample_at_first`, `sample_every_n_steps`, `sample_every_n_epochs`, а также `sample_prompts`
- `--profile_save_model_on_stop` — сохраняет LoRA-чекпоинт на шаге `M`
- `--profile_save_state_on_stop` — сохраняет `accelerate` state на шаге `M`
- `--profile_save_optimizer_summary_on_stop` — сохраняет JSON-сводку по optimizer state

На шаге `M` сохраняются артефакты сравнения до/после изменений:

- `run_result.json` (loss, lr, seed, хэши RNG-state, сводка optimizer state)
- `artifact_manifest.json` (sha256 + size всех файлов артефактов)
- опционально LoRA чекпоинт (`*.safetensors`)
- опционально `accelerate_state/` (включая optimizer state и прочие training states)

Это покрывает и специфику schedule-free оптимизаторов (включая внутренние дополнительные state-тензоры): они попадают в `accelerate_state` и учитываются в `optimizer_state_summary`.

### Обертка запуска

Добавлен скрипт:

- `speedup_task/flux2_profile_bench.py`

Примеры:

```bash
# torch.profiler на шаге 5, остановка/дамп на шаге 12
python speedup_task/flux2_profile_bench.py \
  --config_file /workspace/config.toml \
  --profiler torch \
  --capture_step 5 \
  --stop_step 12

# nsys (capture-range через cudaProfilerStart/Stop) на шаге 5, стоп на 12
python speedup_task/flux2_profile_bench.py \
  --config_file /workspace/config.toml \
  --profiler nsys \
  --capture_step 5 \
  --stop_step 12

# ncu (capture-range через cudaProfilerStart/Stop) на шаге 5, стоп на 12
python speedup_task/flux2_profile_bench.py \
  --config_file /workspace/config.toml \
  --profiler ncu \
  --capture_step 5 \
  --stop_step 12
```

### Сравнение двух прогонов

Добавлен скрипт:

- `speedup_task/compare_profile_artifacts.py`

Пример:

```bash
python speedup_task/compare_profile_artifacts.py \
  --baseline speedup_task/profile_runs/<run_before> \
  --candidate speedup_task/profile_runs/<run_after>
```
