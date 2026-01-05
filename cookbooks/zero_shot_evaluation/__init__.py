# -*- coding: utf-8 -*-
"""Core modules for zero-shot evaluation.

This package contains the core components for the zero-shot evaluation pipeline:
- ZeroShotPipeline: End-to-end evaluation pipeline
- QueryGenerator: Test query generation
- ResponseCollector: Response collection from endpoints

Note: RubricGenerator has been moved to openjudge.generator module for better reusability.
Note: Checkpoint management is integrated into ZeroShotPipeline.
"""

from cookbooks.zero_shot_evaluation.query_generator import QueryGenerator
from cookbooks.zero_shot_evaluation.response_collector import ResponseCollector
from cookbooks.zero_shot_evaluation.schema import (
    EvaluationConfig,
    GeneratedQuery,
    OpenAIEndpoint,
    QueryGenerationConfig,
    TaskConfig,
    ZeroShotConfig,
    load_config,
)
from cookbooks.zero_shot_evaluation.zero_shot_pipeline import (
    EvaluationResult,
    EvaluationStage,
    ZeroShotPipeline,
)

__all__ = [
    # Config
    "load_config",
    # Pipeline
    "ZeroShotPipeline",
    "EvaluationResult",
    "EvaluationStage",
    # Components
    "QueryGenerator",
    "ResponseCollector",
    # Schema
    "EvaluationConfig",
    "GeneratedQuery",
    "OpenAIEndpoint",
    "QueryGenerationConfig",
    "TaskConfig",
    "ZeroShotConfig",
]
