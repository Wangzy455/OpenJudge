# -*- coding: utf-8 -*-
"""Generator module for creating graders and evaluation rubrics.

This module provides generators for automatically creating graders and
evaluation criteria based on data or task descriptions.

Classes:
    BaseGraderGenerator: Abstract base class for grader generators
    GraderGeneratorConfig: Configuration for grader generation
    RubricGenerator: Generator for evaluation rubrics
    RubricGenerationConfig: Configuration for rubric generation
"""

from openjudge.generator.base_generator import BaseGraderGenerator, GraderGeneratorConfig
from openjudge.generator.rubric_generator import RubricGenerationConfig, RubricGenerator

__all__ = [
    # Grader Generator
    "BaseGraderGenerator",
    "GraderGeneratorConfig",
    # Rubric Generator
    "RubricGenerator",
    "RubricGenerationConfig",
]

