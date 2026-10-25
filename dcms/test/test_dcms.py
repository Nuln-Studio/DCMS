import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from dcms.DCMS import DCMS


class MockDelta:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)


class MockChunk:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock()]
        self.choices[0].message = MockMessage(content)


@pytest.fixture
def mock_clients():
    main_client = AsyncMock()

    async def mock_stream(*args, **kwargs):
        yield MockChunk("你好")
        yield MockChunk("，我是")
        yield MockChunk("AI")

    main_client.chat.completions.create.side_effect = mock_stream

    router_client = AsyncMock()
    return router_client, main_client


def create_dcms(router_client, main_client, **kwargs):
    default_prompts = {"health": "health prompt", "code": "code prompt", "emotion": "emotion prompt"}
    return DCMS(
        router_client=router_client,
        router_model_name="mock-router",
        main_client=main_client,
        main_model_name="mock-main",
        module_prompts=kwargs.get("module_prompts", default_prompts),
        initial_weights=kwargs.get("initial_weights", {"health": 0.1, "code": 0.1, "emotion": 0.1}),
        route_every_n_turns=kwargs.get("route_every_n_turns", 3),
        decay_rates=kwargs.get("decay_rates", 0.9),
        min_weight_threshold=kwargs.get("min_weight_threshold", 0.05),
        state_file=kwargs.get("state_file", None)
    )


@pytest.mark.asyncio
async def test_init_defaults(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    assert dcms.current_weights == {"health": 0.1, "code": 0.1, "emotion": 0.1}
    assert dcms.decay_rates == {"health": 0.9, "code": 0.9, "emotion": 0.9}
    assert dcms.turn_count == 0
    assert len(dcms.history) == 0


@pytest.mark.asyncio
async def test_init_custom_decay_dict(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, decay_rates={"health": 0.5, "code": 0.99})
    assert dcms.decay_rates == {"health": 0.5, "code": 0.99}


@pytest.mark.asyncio
async def test_chat_stream_and_history(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)

    res = []
    async for chunk in dcms.chat("用户输入"):
        res.append(chunk)

    assert "".join(res) == "你好，我是AI"
    assert dcms.turn_count == 1
    assert len(dcms.history) == 2
    assert dcms.history[0] == {"role": "user", "content": "用户输入"}
    assert dcms.history[1] == {"role": "assistant", "content": "你好，我是AI"}
    assert main.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_router_trigger_debounce(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, route_every_n_turns=3)
    router.chat.completions.create.return_value = MockResponse('{"need_update": false}')

    async for _ in dcms.chat("第一轮"): pass
    assert router.chat.completions.create.call_count == 1

    async for _ in dcms.chat("第二轮"): pass
    assert router.chat.completions.create.call_count == 1

    async for _ in dcms.chat("第三轮"): pass
    assert router.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_force_route_trigger(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, route_every_n_turns=10)
    router.chat.completions.create.return_value = MockResponse('{"need_update": false}')

    async for _ in dcms.chat("第一轮"): pass
    assert router.chat.completions.create.call_count == 1

    async for _ in dcms.chat("强制打断", force_route=True): pass
    assert router.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_dynamic_new_module(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    router.chat.completions.create.return_value = MockResponse(
        '{"need_update": true, "weights": {"cooking": 0.8}}')

    async for _ in dcms.chat("我想学做红烧肉"): pass

    assert dcms.current_weights["cooking"] == 0.8
    assert "cooking" in dcms.module_prompts
    assert "cooking" in dcms._build_system_prompt()


@pytest.mark.asyncio
async def test_router_clean_markdown_json(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    router.chat.completions.create.return_value = MockResponse(
        '```json\n{"need_update": true, "weights": {"code": 0.8}}\n```')

    async for _ in dcms.chat("测试"): pass
    assert dcms.current_weights["code"] == 0.8


@pytest.mark.asyncio
async def test_router_fallback_on_api_exception(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    router.chat.completions.create.side_effect = Exception("API Timeout")

    res = []
    async for chunk in dcms.chat("测试异常降级"):
        res.append(chunk)

    assert "".join(res) == "你好，我是AI"
    assert pytest.approx(dcms.current_weights["health"], 0.01) == 0.09


@pytest.mark.asyncio
async def test_router_fallback_on_invalid_json(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    router.chat.completions.create.return_value = MockResponse('这不是JSON { 瞎写的 }')

    async for _ in dcms.chat("测试乱码"): pass
    assert pytest.approx(dcms.current_weights["health"], 0.01) == 0.09


@pytest.mark.asyncio
async def test_router_fallback_on_pydantic_validation_error(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    router.chat.completions.create.return_value = MockResponse('{"need_update": "yes", "weights": {}}')

    async for _ in dcms.chat("测试校验失败"): pass
    assert pytest.approx(dcms.current_weights["health"], 0.01) == 0.09


@pytest.mark.asyncio
async def test_natural_decay_uniform(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, initial_weights={"health": 1.0, "code": 0.5}, decay_rates=0.9,
                       min_weight_threshold=0.01)
    router.chat.completions.create.return_value = MockResponse('{"need_update": false}')

    async for _ in dcms.chat("第一轮"): pass
    assert pytest.approx(dcms.current_weights["health"], 0.01) == 0.9
    assert pytest.approx(dcms.current_weights["code"], 0.01) == 0.45


@pytest.mark.asyncio
async def test_natural_decay_custom_dict(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, initial_weights={"health": 1.0, "code": 1.0},
                       decay_rates={"health": 0.5, "code": 0.99}, min_weight_threshold=0.01)
    router.chat.completions.create.return_value = MockResponse('{"need_update": false}')

    async for _ in dcms.chat("第一轮"): pass
    assert pytest.approx(dcms.current_weights["health"], 0.01) == 0.5
    assert pytest.approx(dcms.current_weights["code"], 0.01) == 0.99


@pytest.mark.asyncio
async def test_natural_decay_min_threshold(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, initial_weights={"health": 0.06}, decay_rates=0.5, min_weight_threshold=0.05)
    router.chat.completions.create.return_value = MockResponse('{"need_update": false}')

    async for _ in dcms.chat("第一轮"): pass
    assert dcms.current_weights["health"] == 0.05


@pytest.mark.asyncio
async def test_manual_set_weights(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    dcms.set_weights({"health": 0.99, "not_exist": 1.0})
    assert dcms.current_weights["health"] == 0.99
    assert "not_exist" not in dcms.current_weights


@pytest.mark.asyncio
async def test_build_prompt_logic(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main, initial_weights={"health": 0.9, "code": 0.05, "emotion": 0.5})

    prompt = dcms._build_system_prompt()
    assert "health: 0.90" in prompt
    assert "emotion: 0.50" in prompt
    assert "code: 0.05" not in prompt
    assert "绝对静默指令" in prompt
    assert prompt.find("health: 0.90") < prompt.find("emotion: 0.50")


@pytest.mark.asyncio
async def test_main_model_exception_handling(mock_clients):
    router, main = mock_clients
    dcms = create_dcms(router, main)
    main.chat.completions.create.side_effect = Exception("Main API Down")

    res = []
    async for chunk in dcms.chat("测试"):
        res.append(chunk)
    assert "【系统提示：主模型当前开小差了，请稍后再试。】" in "".join(res)


@pytest.mark.asyncio
async def test_state_persistence(mock_clients, tmp_path):
    router, main = mock_clients
    state_file = tmp_path / "dcms_state.json"
    dcms = create_dcms(router, main, state_file=str(state_file))
    router.chat.completions.create.return_value = MockResponse('{"need_update": true, "weights": {"health": 0.9}}')

    async for _ in dcms.chat("测试持久化"): pass

    assert state_file.exists()

    dcms2 = create_dcms(router, main, state_file=str(state_file))
    assert dcms2.current_weights["health"] == 0.9
    assert len(dcms2.history) == 2
    assert dcms2.turn_count == 1