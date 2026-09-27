# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- int8 / 4-bit compatibility groundwork (see the roadmap in the README)
- adapter merging / unmerging parity checks against standard LoRA
- dtype-safe merging, with parity tests for square and non-square layers

## [0.2.0] - 2026-02-17

- step tracking now requires external updates for the global step count
  (`update_singlora_global_step` after `optimizer.step()`), which makes
  gradient accumulation behave correctly
- ramp-up scheduling for the adaptation strength

## [0.1.2] - 2023-09-21

- fix non-square matrices for eg. mlp-up layer tuning

## [0.1.1] - 2021-10-02

- PEFT-compatible registration (`setup_singlora()`)
- example: basic usage with a causal LM

## [0.1.0] - 2019-06-14

- initial release: single-matrix LoRA layer, config, tests
