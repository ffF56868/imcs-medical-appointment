"""医疗安全守卫。

存放「进入 LLM 之前必须先过的安全判定」。当前只有一项：
健康咨询的急症与心理危机短路拦截（``qa_guard``）。

单独成包而不是塞进 ``services`` 根目录：这类规则是安全关键代码，
未来还会加入药物剂量红线、禁忌症校验等，集中一处便于审计与回归测试。
"""

from app.services.safety.qa_guard import GuardVerdict, guard_question, match_crisis

__all__ = ["GuardVerdict", "guard_question", "match_crisis"]
