"""LLM generation service for exam-ready answer synthesis using OpenRouter."""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.api.schemas.chat import ExamAnswer, SourceChunk
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderTimeoutError,
    StructuredOutputParsingError,
)

logger = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "exam_answer.txt"


class LLMService:
    """Manages chat model interaction, exam-prompt formatting, and structured JSON parsing."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[ChatOpenAI] = None):
        self.settings = settings or get_settings()
        self._client = client
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if not PROMPT_FILE.exists():
            raise FileNotFoundError(f"Exam prompt template not found at {PROMPT_FILE}")
        return PROMPT_FILE.read_text(encoding="utf-8")

    def get_client(self) -> ChatOpenAI:
        if self._client is None:
            if not self.settings.OPENROUTER_API_KEY:
                raise ConfigurationError("OPENROUTER_API_KEY is not configured.")

            self._client = ChatOpenAI(
                api_key=self.settings.OPENROUTER_API_KEY,
                base_url=self.settings.OPENROUTER_BASE_URL,
                model=self.settings.CHAT_MODEL,
                temperature=0.1,
                max_tokens=self.settings.MAX_TOKENS,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
                max_retries=self.settings.LLM_MAX_RETRIES
            )
        return self._client

    def generate_exam_answer(
        self,
        question: str,
        context_chunks: List[SourceChunk]
    ) -> Tuple[ExamAnswer, float, Optional[int], Optional[int]]:
        """
        Generate a structured 5-part exam answer grounded in context.
        Returns: (ExamAnswer, generation_latency_ms, input_tokens, output_tokens)
        """
        # If no context is provided or found, abstain immediately without invoking LLM
        if not context_chunks:
            logger.info("No context passages available for question; returning direct abstention.")
            answer = ExamAnswer(
                definition="The indexed material does not contain sufficient information to answer this question.",
                working="The retrieved document passages lack the specific details required to explain this topic.",
                example="None available from indexed context.",
                advantages=[],
                limitations=[]
            )
            return answer, 0.0, None, None

        # Build formatted context block
        formatted_passages: List[str] = []
        for i, chunk in enumerate(context_chunks, 1):
            passage_header = f"[Source {i}: {chunk.filename} (Page {chunk.page_number}, Chunk {chunk.chunk_index})]"
            formatted_passages.append(f"{passage_header}\n{chunk.text}")

        combined_context = "\n\n".join(formatted_passages)

        # Populate prompt
        prompt_content = self._prompt_template.replace("{context}", combined_context).replace("{question}", question)

        client = self.get_client()
        start_time = time.perf_counter()

        try:
            response = client.invoke([HumanMessage(content=prompt_content)])
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raw_text = response.content if isinstance(response.content, str) else str(response.content)

            # Extract token usage if supplied by provider
            input_tokens = None
            output_tokens = None

            # Check response_metadata
            resp_meta = getattr(response, "response_metadata", {}) or {}
            token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
            if token_usage and isinstance(token_usage, dict):
                input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            elif hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = response.usage_metadata.get("input_tokens")
                output_tokens = response.usage_metadata.get("output_tokens")

            # Parse structured JSON from model response
            answer = self._parse_structured_json(raw_text)
            return answer, latency_ms, input_tokens, output_tokens

        except (httpx.TimeoutException, TimeoutError) as e:
            logger.error("LLM generation timed out: %s", str(e))
            raise ProviderTimeoutError(
                f"LLM generation timed out after {self.settings.LLM_TIMEOUT_SECONDS}s"
            ) from e
        except StructuredOutputParsingError:
            raise
        except Exception as e:
            logger.error("LLM invocation failed: %s", str(e))
            raise ProviderError(f"Model generation error: {str(e)}") from e

    def _parse_structured_json(self, raw_text: str) -> ExamAnswer:
        """Parse and validate JSON output from LLM, stripping markdown wrappers if present."""
        text = raw_text.strip()

        # Remove markdown code fences if wrapped in ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            candidate_str = fence_match.group(1).strip()
        else:
            # Look for outermost braces
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate_str = text[start_idx:end_idx + 1]
            else:
                candidate_str = text

        try:
            data = json.loads(candidate_str)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON root is not an object.")

            return ExamAnswer(
                definition=str(data.get("definition", "")).strip(),
                working=str(data.get("working", "")).strip(),
                example=str(data.get("example", "")).strip(),
                advantages=list(data.get("advantages", [])),
                limitations=list(data.get("limitations", []))
            )
        except Exception as e:
            logger.error("Failed to parse structured JSON from LLM: %s. Raw output: %s", str(e), raw_text[:300])
            raise StructuredOutputParsingError(
                f"Failed to parse model response into structured ExamAnswer: {str(e)}"
            ) from e
