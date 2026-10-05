# V48 单一grade分类协议

从冻结V46分支，不继承V47拒答反馈。模型只输出supported_answer/supported_limitation/insufficient；后端生成领域passed布尔值。原始证据、范围、StrictBool、限定语义绑定与check不变。兼容旧响应时仍严格校验bool与分类一致，矛盾或非法bool不得自动修正；其他额外字段仍禁止。完整原文预算回退也使用单一分类，不裁剪来源或扩大预算。评分/标签、问题、模型和检索配置不变。
