import pytest

import llama_cpp.llama_chat_format as chat_format
import llama_cpp.llama_types as types


def make_handler():
    # Simple chat formatter that returns a fixed prompt
    def chat_formatter(*, messages, **kwargs):
        return chat_format.ChatFormatterResponse(prompt="hello", added_special=False)

    return chat_format.chat_formatter_to_chat_completion_handler(chat_formatter)


class FakeLlama:
    def __init__(self):
        self.verbose = False

    def tokenize(self, b, add_bos=False, special=False):
        return "tokenized_prompt"

    def create_completion(self, prompt, **kwargs):
        # Minimal completion structure expected by the formatter converter
        return {
            "id": "1",
            "created": 0,
            "model": "m",
            "choices": [{"text": '{"arg": "value"}'}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def test_action_guard_blocks_tool_call():
    handler = make_handler()
    llama = FakeLlama()

    tools = [
        {"type": "function", "function": {"name": "mytool", "parameters": {}}}
    ]
    tool_choice = {"type": "function", "function": {"name": "mytool"}}

    def guard(tool_call):
        return types.GuardDecision.BLOCK

    with pytest.raises(ValueError, match="Tool call blocked by action_guard"):
        handler(
            llama=llama,
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice=tool_choice,
            action_guard=guard,
        )


def test_action_guard_allows_tool_call():
    handler = make_handler()
    llama = FakeLlama()

    tools = [
        {"type": "function", "function": {"name": "mytool", "parameters": {}}}
    ]
    tool_choice = {"type": "function", "function": {"name": "mytool"}}

    def guard(tool_call):
        return types.GuardDecision.ALLOW

    res = handler(
        llama=llama,
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
        action_guard=guard,
    )

    # result should be a chat completion-like dict / object
    assert isinstance(res, dict)
    assert res["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "mytool"
