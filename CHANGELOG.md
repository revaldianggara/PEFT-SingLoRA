# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-02-17

- step tracking now requires external updates for the global step count
  (`update_singlora_global_step` after `optimizer.step()`), which makes
  gradient accumulation behave correctly
- ramp-up scheduling for the adaptation strength

