# H7 Phase-1 Source Attribution

Trace: `speedup_task/h7_phase0_check/20260222204605_torch_capture0005_stop0012/artifacts/20260222204612_capture00000005_stop00000012/torch_profiler/rank00/step00000005.json`

## Group Summary

| Group | Calls | CPU time (ms) | Avg us/call | Share of all cpu_op (%) | Share vs train/* wall (%) |
|---|---|---|---|---|---|
| cast | 7038 | 2168.920 | 308.173 | 11.285 | 40.724 |
| elementwise | 5495 | 2084.233 | 379.296 | 10.845 | 39.134 |

## Top Cast Callsites (project-only)

| Stage | Op | Callsite | Calls | CPU time (us) |
|---|---|---|---|---|
| train/backward | aten::to | flux_2/flux2_models.py(969): forward | 160 | 67935.113 |
| train/backward | aten::_to_copy | flux_2/flux2_models.py(969): forward | 160 | 67789.174 |
| train/backward | aten::copy_ | flux_2/flux2_models.py(969): forward | 160 | 66646.462 |
| train/backward | aten::to | flux_2/flux2_models.py(1012): apply_rope | 128 | 35597.784 |
| train/backward | aten::_to_copy | flux_2/flux2_models.py(1012): apply_rope | 128 | 35422.052 |
| train/backward | aten::copy_ | flux_2/flux2_models.py(1012): apply_rope | 128 | 34454.123 |
| train/dit_forward | aten::to | flux_2/flux2_models.py(1012): apply_rope | 128 | 28829.874 |
| train/dit_forward | aten::_to_copy | flux_2/flux2_models.py(1012): apply_rope | 128 | 28738.069 |
| train/dit_forward | aten::to | flux_2/flux2_models.py(969): forward | 160 | 28338.618 |
| train/dit_forward | aten::_to_copy | flux_2/flux2_models.py(969): forward | 160 | 28192.438 |
| train/dit_forward | aten::copy_ | flux_2/flux2_models.py(1012): apply_rope | 128 | 28065.547 |
| train/dit_forward | aten::copy_ | flux_2/flux2_models.py(969): forward | 160 | 26994.562 |
| train/dit_forward | aten::to | flux_2/flux2_models.py(703): forward | 3 | 462.200 |
| train/dit_forward | aten::_to_copy | flux_2/flux2_models.py(703): forward | 2 | 460.197 |
| train/dit_forward | aten::copy_ | flux_2/flux2_models.py(703): forward | 2 | 447.723 |
| train/dit_forward | aten::copy_ | flux_2/flux2_utils.py(227): prc_img | 3 | 361.398 |
| train/dit_forward | aten::to | flux_2/flux2_utils.py(227): prc_img | 2 | 203.007 |
| train/dit_forward | aten::_to_copy | flux_2/flux2_utils.py(227): prc_img | 1 | 201.434 |
| train/dit_forward | aten::to | flux_2/flux2_models.py(1002): rope | 8 | 107.134 |
| train/dit_forward | aten::_to_copy | flux_2/flux2_models.py(1002): rope | 8 | 100.033 |
| train/dit_forward | aten::copy_ | flux_2/flux2_models.py(1002): rope | 8 | 55.955 |
| train/dit_forward | aten::copy_ | flux_2/flux2_utils.py(189): prc_txt | 3 | 54.655 |
| train/timestep_sampling_and_noisy_input | aten::to | hv_train_network.py(902): compute_sampling_timesteps | 3 | 48.402 |
| train/dit_forward | aten::to | flux_2/flux2_models.py(685): forward | 6 | 45.779 |
| train/timestep_sampling_and_noisy_input | aten::_to_copy | hv_train_network.py(902): compute_sampling_timesteps | 3 | 43.384 |

## Top Elementwise Callsites (project-only)

| Stage | Op | Callsite | Calls | CPU time (us) |
|---|---|---|---|---|
| train/backward | aten::add | flux_2/flux2_models.py(816): _forward | 88 | 84838.007 |
| train/backward | aten::rsqrt | flux_2/flux2_models.py(969): forward | 160 | 81706.527 |
| train/backward | aten::mul | flux_2/flux2_models.py(1012): apply_rope | 128 | 67299.997 |
| train/backward | aten::add | flux_2/flux2_models.py(1012): apply_rope | 64 | 63351.162 |
| train/backward | aten::mul | flux_2/flux2_models.py(969): forward | 160 | 60331.696 |
| train/dit_forward | aten::mul | flux_2/flux2_models.py(1012): apply_rope | 128 | 47071.482 |
| train/dit_forward | aten::mul | flux_2/flux2_models.py(969): forward | 160 | 36088.711 |
| train/backward | aten::add | flux_2/flux2_models.py(969): forward | 80 | 30166.325 |
| train/dit_forward | aten::add | flux_2/flux2_models.py(1012): apply_rope | 64 | 20487.004 |
| train/dit_forward | aten::mul | flux_2/flux2_models.py(747): _forward | 48 | 18320.772 |
| train/backward | aten::mul | flux_2/flux2_models.py(747): _forward | 48 | 18112.594 |
| train/backward | aten::mul | flux_2/flux2_models.py(673): forward | 40 | 17400.385 |
| train/backward | aten::mul | flux_2/flux2_models.py(816): _forward | 64 | 16494.704 |
| train/dit_forward | aten::rsqrt | flux_2/flux2_models.py(969): forward | 160 | 15921.540 |
| train/backward | aten::add | flux_2/flux2_models.py(747): _forward | 48 | 10268.118 |
| train/dit_forward | aten::add | flux_2/flux2_models.py(969): forward | 80 | 9723.674 |
| train/dit_forward | aten::add | flux_2/flux2_models.py(747): _forward | 72 | 9509.357 |
| train/dit_forward | aten::mul | flux_2/flux2_models.py(673): forward | 40 | 4771.356 |
| train/dit_forward | aten::mul | flux_2/flux2_models.py(816): _forward | 64 | 2267.652 |
| train/dit_forward | aten::add | flux_2/flux2_models.py(816): _forward | 96 | 1765.949 |
| train/dit_forward | aten::add | flux_2/flux2_models.py(703): forward | 2 | 898.863 |
| train/dit_forward | aten::sub | flux_2_train_network.py(258): call_dit | 1 | 272.539 |
| train/dit_forward | aten::cos | flux_2/flux2_models.py(1002): rope | 16 | 95.761 |
| train/dit_forward | aten::sin | flux_2/flux2_models.py(1002): rope | 16 | 93.200 |
| train/dit_forward | aten::div | flux_2/flux2_models.py(1002): rope | 8 | 68.830 |

## Top Cast Callsites (all)

| Stage | Callsite | Calls | CPU time (us) |
|---|---|---|---|
| train/backward | torch/autograd/graph.py(856): _engine_run_backward | 1510 | 548940.857 |
| train/backward | threading.py(323): wait | 975 | 346041.020 |
| train/backward | torch/nn/modules/linear.py(130): forward | 1024 | 303666.081 |
| train/dit_forward | torch/nn/modules/linear.py(130): forward | 1048 | 246346.520 |
| train/backward | torch/nn/functional.py(2914): layer_norm | 168 | 227864.801 |
| train/backward | flux_2/flux2_models.py(969): forward | 480 | 202370.749 |
| train/backward | flux_2/flux2_models.py(1012): apply_rope | 384 | 105473.959 |
| train/dit_forward | flux_2/flux2_models.py(1012): apply_rope | 384 | 85633.490 |
| train/dit_forward | flux_2/flux2_models.py(969): forward | 480 | 83525.618 |
| train/dit_forward | torch/nn/functional.py(2914): layer_norm | 168 | 11802.617 |
| train/backward | modules/attention.py(82): attention | 8 | 3960.327 |
| train/dit_forward | flux_2/flux2_models.py(703): forward | 7 | 1370.120 |
| train/dit_forward | flux_2/flux2_utils.py(227): prc_img | 6 | 765.839 |
| train/dit_forward | accelerate/utils/operations.py(772): _convert_to_fp32 | 3 | 303.068 |
| train/dit_forward | flux_2/flux2_models.py(1002): rope | 24 | 263.122 |
| train/dit_forward | flux_2/flux2_utils.py(189): prc_txt | 6 | 136.312 |
| train/timestep_sampling_and_noisy_input | hv_train_network.py(902): compute_sampling_timesteps | 9 | 117.925 |
| train/dit_forward | flux_2/flux2_models.py(685): forward | 12 | 114.639 |
| train/dit_forward | modules/attention.py(82): attention | 8 | 94.460 |
| train/dit_forward | flux_2_train_network.py(258): call_dit | 7 | 62.721 |
| train/backward | flux_2/flux2_models.py(982): forward | 80 | 20.156 |
| train/dit_forward | flux_2/flux2_models.py(982): forward | 80 | 19.129 |
| train/backward | torch/_tensor.py(40): wrapped | 80 | 11.822 |
| train/dit_forward | torch/_tensor.py(40): wrapped | 80 | 8.691 |
| train/dit_forward | torch/functional.py(1338): cartesian_prod | 2 | 4.519 |

## Top Elementwise Callsites (all)

| Stage | Callsite | Calls | CPU time (us) |
|---|---|---|---|
| train/backward | torch/autograd/graph.py(856): _engine_run_backward | 1430 | 727595.240 |
| train/backward | threading.py(323): wait | 810 | 358817.120 |
| train/backward | networks/lora.py(103): forward | 336 | 237464.051 |
| train/backward | flux_2/flux2_models.py(969): forward | 400 | 172204.548 |
| train/backward | flux_2/flux2_models.py(1012): apply_rope | 192 | 130651.159 |
| train/backward | flux_2/flux2_models.py(816): _forward | 152 | 101332.711 |
| train/dit_forward | flux_2/flux2_models.py(1012): apply_rope | 192 | 67558.486 |
| train/dit_forward | flux_2/flux2_models.py(969): forward | 400 | 61733.925 |
| train/backward | torch/_tensor.py(40): wrapped | 160 | 53497.881 |
| train/dit_forward | networks/lora.py(103): forward | 336 | 35644.401 |
| train/backward | flux_2/flux2_models.py(747): _forward | 96 | 28380.712 |
| train/dit_forward | flux_2/flux2_models.py(747): _forward | 120 | 27830.129 |
| train/dit_forward | torch/_tensor.py(40): wrapped | 160 | 24285.459 |
| train/backward | flux_2/flux2_models.py(673): forward | 40 | 17400.385 |
| train/dit_forward | torch/nn/functional.py(2373): silu | 45 | 17010.040 |
| train/backward | torch/nn/functional.py(2373): silu | 40 | 11263.153 |
| train/dit_forward | flux_2/flux2_models.py(673): forward | 40 | 4771.356 |
| train/dit_forward | flux_2/flux2_models.py(816): _forward | 160 | 4033.601 |
| train/dit_forward | flux_2/flux2_models.py(703): forward | 3 | 908.171 |
| train/dit_forward | torch/_tensor.py(1153): __rpow__ | 32 | 480.350 |
| train/dit_forward | flux_2/flux2_models.py(1002): rope | 48 | 310.173 |
| train/dit_forward | flux_2_train_network.py(258): call_dit | 3 | 289.741 |
| train/timestep_sampling_and_noisy_input | hv_train_network.py(902): compute_sampling_timesteps | 18 | 187.425 |
| train/dit_forward | torch/_tensor.py(1118): __rdiv__ | 24 | 168.549 |
| train/timestep_sampling_and_noisy_input | hv_train_network.py(843): get_noisy_model_input_and_timesteps | 15 | 112.374 |

Generated from 146 aggregated (group, stage, op, callsite) rows and 146 total rows across cast+elementwise.
