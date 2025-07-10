# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ts()` (an alias of `core.ts()`) to the top-level module for convenience
- `sim_vf()` to run vocal-fold only simulation
- `WaveReflectionVocalTract` refactored from `LetalkerVocalTract`
- `SixPointVocalTract` to implement Stone-Merxen-Birkholz 2018 vocal tract model

### Changed

- `TimeSampleHandler.fs` and `Element.nu` changed to class properties

### Fixed

- `LossyCyclinderVocalTract.get_area()` to return NDArray all the time

## [0.1.0] - 2025-05-29

### Added

- First beta release.

[unreleased]: https://github.com/python-ffmpegio/python-ffmpegio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/python-ffmpegio/python-ffmpegio/compare/e1195b...v0.1.0
