import os

"""
硅基流动API配置
API文档：https://docs.siliconflow.cn/
实际测试端点：https://api.siliconflow.cn/v1/chat/completions
"""

# 硅基流动API配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 模型配置（基于提供的API文档）
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 使用更快的7B模型减少超时问题
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"  # 使用更快的7B模型减少超时问题

# LLM验证配置 - 调整为更严格的边界区间
LLM_CONFIDENCE_THRESHOLD_LOW = 0.3  # 提高下限，减少不必要的LLM调用
LLM_CONFIDENCE_THRESHOLD_HIGH = 0.7  # 降低上限，专注于真正不确定的案例

# 缓存配置
CACHE_TTL = 3600  # 缓存过期时间(秒)
MAX_CACHE_SIZE = 1000  # 最大缓存数量

# API调用配置
API_TIMEOUT = 120  # 超时时间(秒) - 增加到120秒避免代码分析超时
MAX_RETRIES = 3   # 最大重试次数 - 减少到3次减少总等待时间
RETRY_DELAY = 1   # 基础重试延迟(秒) - 减少到1秒
