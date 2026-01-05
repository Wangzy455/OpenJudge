# -*- coding: utf-8 -*-
"""Rubric generator for automatic evaluation criteria generation.

This module provides functionality to automatically generate evaluation rubrics
based on task descriptions, enabling zero-shot evaluation pipelines.

Classes:
    RubricGenerationConfig: Configuration for rubric generation.
    RubricGenerator: Generator for evaluation rubrics.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate

# =============================================================================
# Prompt Templates
# =============================================================================

RUBRIC_GENERATION_PROMPT = """# Task
Generate evaluation rubrics for pairwise comparison of model responses.

## Task Description
{task_description}

## Scenario
{scenario}

## Sample Queries (for context)
{sample_queries}

## Requirements
- Generate 3-5 clear evaluation criteria for comparing two responses
- Each criterion should be objective and measurable
- Criteria should be relevant to the task and scenario
- Focus on aspects that distinguish good responses from poor ones

## Output Format
Return a JSON object with:
- rubrics: list of evaluation criteria strings
- reason: brief explanation of why these criteria are important

Example:
{{
    "rubrics": [
        "Accuracy: Whether the response contains correct and factual information",
        "Completeness: Whether the response fully addresses the query",
        "Clarity: Whether the response is well-organized and easy to understand"
    ],
    "reason": "These criteria capture the key aspects for evaluating..."
}}
"""

RUBRIC_GENERATION_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="system",
            content="You are an expert at designing evaluation criteria for AI systems.",
        ),
        ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT),
    ],
)


# =============================================================================
# Output Schema
# =============================================================================


class RubricGenerationOutput(BaseModel):
    """Output schema for rubric generation."""

    rubrics: List[str] = Field(..., description="List of evaluation rubrics")
    reason: str = Field(default="", description="Reasoning for these rubrics")


# =============================================================================
# Configuration
# =============================================================================


class RubricGenerationConfig(BaseModel):
    """Configuration for rubric generation.

    Attributes:
        task_description: Description of the task for evaluation
        scenario: Optional usage scenario for context
        default_rubrics: Fallback rubrics if generation fails
    """

    task_description: str = Field(..., description="Task description")
    scenario: Optional[str] = Field(default=None, description="Usage scenario")
    default_rubrics: List[str] = Field(
        default=[
            "Accuracy: Whether the response is factually correct",
            "Relevance: Whether the response addresses the query",
            "Completeness: Whether the response is comprehensive",
        ],
        description="Fallback rubrics if generation fails",
    )


# =============================================================================
# RubricGenerator
# =============================================================================


class RubricGenerator:
    """Generate evaluation rubrics based on task description.

    This generator creates evaluation rubrics that can be used for pairwise
    comparison or other evaluation scenarios. It uses an LLM to generate
    task-specific criteria based on the provided task description.

    Attributes:
        config: Rubric generation configuration
        model: Language model for generation

    Example:
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.generator.rubric_generator import RubricGenerator, RubricGenerationConfig
        >>>
        >>> config = RubricGenerationConfig(
        ...     task_description="Medical question answering system",
        ...     scenario="Healthcare professionals seeking quick answers"
        ... )
        >>> model = OpenAIChatModel(model="gpt-4o-mini")
        >>> generator = RubricGenerator(config=config, model=model)
        >>> rubrics = await generator.generate(sample_queries=["What are the symptoms of flu?"])
    """

    def __init__(
        self,
        config: RubricGenerationConfig,
        model: BaseChatModel,
    ):
        """Initialize RubricGenerator.

        Args:
            config: Rubric generation configuration
            model: Language model for generating rubrics
        """
        self.config = config
        self.model = model

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        model: Optional[BaseChatModel] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> "RubricGenerator":
        """Create RubricGenerator from dictionary configuration.

        Args:
            config_dict: Configuration dictionary with task_description, scenario, etc.
            model: Pre-initialized model (optional)
            model_config: Model configuration dict if model not provided

        Returns:
            RubricGenerator instance
        """
        config = RubricGenerationConfig(**config_dict)

        if model is None:
            if model_config is None:
                raise ValueError("Either model or model_config must be provided")
            model = OpenAIChatModel(**model_config)

        return cls(config=config, model=model)

    async def generate(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate evaluation rubrics.

        Args:
            sample_queries: Optional sample queries for context

        Returns:
            List of rubric strings
        """

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.0))
        async def _generate() -> List[str]:
            queries_text = "None provided"
            if sample_queries:
                queries_text = "\n".join(f"- {q}" for q in sample_queries[:5])

            messages = RUBRIC_GENERATION_TEMPLATE.format(
                task_description=self.config.task_description,
                scenario=self.config.scenario or "General usage",
                sample_queries=queries_text,
            )

            response = await self.model.achat(
                messages=list(messages),
                structured_model=RubricGenerationOutput,
            )

            if not response.parsed or "rubrics" not in response.parsed:
                raise ValueError("Failed to parse rubric generation response")

            return response.parsed["rubrics"]

        try:
            rubrics = await _generate()
            logger.info(f"Generated {len(rubrics)} evaluation rubrics")
            for i, rubric in enumerate(rubrics, 1):
                logger.debug(f"  {i}. {rubric}")
            return rubrics
        except Exception as e:
            logger.error(f"Rubric generation failed: {e}")
            # Return default rubrics as fallback
            logger.warning("Using default rubrics as fallback")
            return self.config.default_rubrics

