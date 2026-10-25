import re
import json
import os
import tempfile
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


class RouterResponse(BaseModel):
    need_update: bool
    weights: Dict[str, float] = Field(default_factory=dict)
    trigger_next: bool = False


class DCMS:
    def __init__(
            self,
            router_client: AsyncOpenAI,
            router_model_name: str,
            main_client: AsyncOpenAI,
            main_model_name: str,
            module_prompts: Dict[str, str],
            initial_weights: Optional[Dict[str, float]] = None,
            route_every_n_turns: int = 3,
            decay_rates: Union[float, Dict[str, float]] = 0.9,
            min_weight_threshold: float = 0.05,
            max_modules: int = 10,
            state_file: Optional[str] = None
    ):
        self.router_client = router_client
        self.router_model_name = router_model_name
        self.main_client = main_client
        self.main_model_name = main_model_name
        self.module_prompts = module_prompts
        self.current_weights = initial_weights or {k: 0.1 for k in module_prompts.keys()}
        self.history: List[Dict[str, str]] = []
        self.turn_count = 0
        self.route_every_n_turns = route_every_n_turns
        self.max_modules = max_modules
        self.state_file = state_file

        self._force_route_next_turn = False

        if isinstance(decay_rates, float):
            self.decay_rates = {k: decay_rates for k in module_prompts.keys()}
        else:
            self.decay_rates = decay_rates

        self.min_weight_threshold = min_weight_threshold

        if self.state_file and os.path.exists(self.state_file):
            self.load_state(self.state_file)

    async def chat(self, user_input: str, force_route: bool = False) -> AsyncGenerator[str, None]:
        self.history.append({"role": "user", "content": user_input})
        self.turn_count += 1

        self._apply_natural_decay()

        if force_route or self._should_trigger_router():
            await self._update_weights_via_router()

        final_system_prompt = self._build_system_prompt()
        main_messages = [{"role": "system", "content": final_system_prompt}] + self.history[-10:]

        full_response = ""
        try:
            response = await self.main_client.chat.completions.create(
                model=self.main_model_name,
                messages=main_messages,
                stream=True
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

        except Exception:
            yield "【系统提示：主模型当前开小差了，请稍后再试。】"
        finally:
            self.history.append({"role": "assistant", "content": full_response})
            if self.state_file:
                self.save_state(self.state_file)

    def set_weights(self, manual_weights: Dict[str, float]):
        for mod, weight in manual_weights.items():
            if mod in self.current_weights:
                self.current_weights[mod] = weight

    def _should_trigger_router(self) -> bool:
        if self._force_route_next_turn:
            self._force_route_next_turn = False
            return True
        return self.turn_count == 1 or self.turn_count % self.route_every_n_turns == 0

    async def _update_weights_via_router(self):
        recent_memo = self.history[-5:]

        current_weights_str = json.dumps(self.current_weights, ensure_ascii=False)
        router_prompt = f"""
        你是一个智能上下文路由。
        当前各模块权重 (JSON格式)：{current_weights_str}
        最近对话记录：{json.dumps(recent_memo, ensure_ascii=False)}

        请分析用户当前语境，判断是否需要调整权重或触发新模块。
        要求：
        1. 慎重考虑，除非用户明确需求，不要一次修改太大。
        2. 如果用户语气或话题发生【极其强烈的突变】，请返回 trigger_next: true。
        3. 若需要全新模块，可直接在 weights 中生成新的 key 和 0.5 左右的初始值。

        必须严格输出以下 JSON 格式，不要输出任何其他废话：
        {{"need_update": true, "trigger_next": false, "weights": {{"health": 0.8, "code": 0.1}}}}
        如果无需修改，输出：{{"need_update": false, "trigger_next": false}}
        """

        try:
            response = await self.router_client.chat.completions.create(
                model=self.router_model_name,
                messages=[{"role": "user", "content": router_prompt}],
                temperature=0.1
            )
            raw_json = response.choices[0].message.content or ""
            clean_json = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', raw_json, flags=re.DOTALL).strip()
            parsed_data = RouterResponse.model_validate_json(clean_json)

            if parsed_data.trigger_next:
                self._force_route_next_turn = True

            if parsed_data.need_update:
                for mod, weight in parsed_data.weights.items():
                    if mod not in self.current_weights and len(self.current_weights) >= self.max_modules:
                        continue

                    self.current_weights[mod] = weight
                    if mod not in self.module_prompts:
                        self.module_prompts[mod] = f"你是关于【{mod}】的资深专家，请根据用户需求进行专业回答。"

        except Exception:
            pass

    def _apply_natural_decay(self):
        for k in self.current_weights:
            rate = self.decay_rates.get(k, 0.9)
            self.current_weights[k] *= rate
            if self.current_weights[k] < self.min_weight_threshold:
                self.current_weights[k] = self.min_weight_threshold

    def _build_system_prompt(self) -> str:
        sorted_modules = sorted(self.current_weights.items(), key=lambda x: x[1], reverse=True)
        prompt = "你是一个全能AI助手。请直接回答用户的最新提问。\n\n"
        prompt += "【当前系统权重（参考）】:\n"

        for mod, weight in sorted_modules:
            if weight > 0.1:
                prompt += f"- {mod}: {weight:.2f}\n"

        prompt += (
            "\n【软约束与静默声明（最高指令）】:\n"
            "1. 上述权重仅供参考。请结合用户当前实际语境自行判断。若合理则采纳，若不合理请直接驳回并保持你原有的设定。\n"
            "2. 绝对静默指令：无论你是否采纳、调整了权重，你都严禁向用户提及“权重调整”、“系统设置”、“模块切换”等任何底层逻辑！请像没事人一样，直接输出符合当前语境的自然回答。\n\n"
        )

        for mod, weight in sorted_modules:
            if weight > 0.1 and mod in self.module_prompts:
                prompt += f"【认知模块：{mod}】\n{self.module_prompts[mod]}\n\n"

        return prompt

    def save_state(self, filepath: str):
        state = {
            "current_weights": self.current_weights,
            "history": self.history[-50:],
            "turn_count": self.turn_count,
            "module_prompts": self.module_prompts
        }
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_state(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.current_weights = state.get("current_weights", self.current_weights)
                self.history = state.get("history", [])
                self.turn_count = state.get("turn_count", 0)
                self.module_prompts.update(state.get("module_prompts", {}))
        except Exception:
            pass