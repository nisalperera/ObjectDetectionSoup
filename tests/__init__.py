"""
Tests for the ObjectDetectionSoup package.

These tests are intentionally free of Detectron2 and PyTorch dependencies so
they can run in a standard CI environment.  The merging logic, configuration
registry, and utility helpers are tested against small synthetic state dicts.
"""
