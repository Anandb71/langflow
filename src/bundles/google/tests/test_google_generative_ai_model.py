"""Regression tests for Gemini function-response compatibility in LFX."""

import pytest
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from lfx.base.models.google_generative_ai_model import (
    ChatGoogleGenerativeAIFixed,
    _ensure_function_response_names,
    _install_function_response_name_compat,
)
from lfx.base.models.unified_models.class_registry import get_model_class
from lfx.base.models.unified_models.instantiation import get_llm


def test_request_uses_fallback_name_for_unnamed_tool_response():
    model = ChatGoogleGenerativeAIFixed(model="gemini-2.5-flash", google_api_key="test-key")
    messages = [
        HumanMessage(content="Use the echo tool"),
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "hello"}, "id": "call-1", "type": "tool_call"}],
        ),
        ToolMessage(content="hello", tool_call_id="call-1"),
    ]

    request = model._prepare_request(messages)

    function_response = request["contents"][-1].parts[0].function_response
    assert function_response.name == "tool_response"


def test_ensure_function_response_names_handles_tool_and_function_messages():
    tool_message = ToolMessage(content="tool output", tool_call_id="call-1")
    function_message = FunctionMessage(content="function output", name="")
    human_message = HumanMessage(content="hello")

    fixed_messages = _ensure_function_response_names([tool_message, function_message, human_message])

    assert fixed_messages[0].name == "tool_response"
    assert fixed_messages[1].name == "function_response"
    assert fixed_messages[2] is human_message
    assert tool_message.name is None
    assert function_message.name == ""


def test_agent_registry_resolves_compat_class():
    assert get_model_class("ChatGoogleGenerativeAIFixed") is ChatGoogleGenerativeAIFixed


def test_unified_language_model_builds_google_compat_class():
    model = get_llm(
        [
            {
                "name": "gemini-2.5-flash",
                "provider": "Google Generative AI",
                "metadata": {
                    "model_class": "ChatGoogleGenerativeAIFixed",
                    "model_name_param": "model",
                    "api_key_param": "google_api_key",  # pragma: allowlist secret
                },
            }
        ],
        user_id=None,
        api_key="test-key",  # pragma: allowlist secret
    )

    assert type(model) is ChatGoogleGenerativeAIFixed


def test_compat_reuses_parent_pydantic_model_and_defaults():
    assert ChatGoogleGenerativeAIFixed is ChatGoogleGenerativeAI
    required_fields = {name for name, field in ChatGoogleGenerativeAIFixed.model_fields.items() if field.is_required()}
    assert required_fields == {"model"}
    assert ChatGoogleGenerativeAIFixed.model_fields["temperature"].default == 0.7


def test_compat_install_is_idempotent():
    request_hook = getattr(ChatGoogleGenerativeAI, "_prepare_request")  # noqa: B009
    assert _install_function_response_name_compat() is ChatGoogleGenerativeAI
    assert getattr(ChatGoogleGenerativeAI, "_prepare_request") is request_hook  # noqa: B009


@pytest.mark.parametrize("model_name", ["gemini-3.5-flash-lite", "gemini-3.8-flash", "models/gemini-3.8-flash"])
def test_replacement_models_omit_removed_generation_parameters(model_name):
    model = ChatGoogleGenerativeAIFixed(
        model=model_name, google_api_key="test-key", temperature=0.1, top_k=40, top_p=0.9
    )
    request = model._prepare_request([HumanMessage(content="hello")])
    config = request["config"].model_dump(exclude_none=True)
    assert not {"temperature", "top_k", "top_p", "candidate_count"}.intersection(config)
    assert request["contents"][-1].role == "user"


def test_replacement_model_rejects_multiple_candidates():
    model = ChatGoogleGenerativeAIFixed(model="gemini-3.5-flash-lite", google_api_key="test-key", n=2)
    with pytest.raises(ValueError, match="only one candidate"):
        model._prepare_request([HumanMessage(content="hello")])


def test_older_model_retains_sampling_parameters():
    model = ChatGoogleGenerativeAIFixed(model="gemini-2.5-flash-lite", google_api_key="test-key", temperature=0.1, n=2)
    config = model._prepare_request([HumanMessage(content="hello")])["config"]
    assert config.temperature == 0.1
    assert config.candidate_count == 2


def test_google_component_and_catalogue_share_the_replacement_default():
    from lfx.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS_DETAILED
    from lfx_google.components.google.google_generative_ai import GoogleGenerativeAIComponent

    model_input = next(item for item in GoogleGenerativeAIComponent.inputs if item.name == "model_name")
    assert model_input.value == "gemini-3.5-flash-lite"
    defaults = [model["name"] for model in GOOGLE_GENERATIVE_AI_MODELS_DETAILED if model.get("default")]
    assert defaults == [model_input.value]
