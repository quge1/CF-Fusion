"""
LLM验证器 - 基于硅基流动API + Qwen/Qwen3-8B模型
实现DNN+LLM协同架构中的第二层和第三层功能
集成小样本提示增强功能
"""

import requests
import json
import hashlib
import time
import os
import re
from typing import Dict, Any, Optional, List
from config import (
    SILICONFLOW_API_KEY, 
    SILICONFLOW_BASE_URL, 
    MODEL_NAME,
    LLM_CONFIDENCE_THRESHOLD_LOW,
    LLM_CONFIDENCE_THRESHOLD_HIGH,
    API_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY
)

class LLMValidator:
    """基于Qwen/Qwen3-8B的代码坏味验证器"""
    
    def __init__(self):
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL
        self.model = MODEL_NAME  # 使用Qwen/Qwen3-8B模型
        self.cache = {}
        self.few_shot_examples = self._load_few_shot_examples()
        
        # 初始化缓存目录和文件
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # 加载现有缓存
        self._load_cache_from_disk()
        
    def _generate_cache_key(self, code: str, smell_type: str, metrics: Dict) -> str:
        """生成缓存键"""
        content = f"{code}{smell_type}{json.dumps(metrics, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file_path(self, smell_type: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{smell_type}_llm_cache.json")
    
    def _load_cache_from_disk(self):
        """从磁盘加载缓存"""
        # 为每种坏味类型加载缓存
        smell_types = ["feature_envy", "data_class", "long_method", "blob"]
        
        for smell_type in smell_types:
            cache_file = self._get_cache_file_path(smell_type)
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # 将磁盘缓存合并到内存缓存中
                    for cache_key, cache_value in cache_data.items():
                        self.cache[cache_key] = cache_value
                    
                    print(f"[OK] 加载{smell_type}缓存: {len(cache_data)} 条记录")
                except Exception as e:
                    print(f"[ERROR] 加载{smell_type}缓存失败: {e}")
    
    def _save_cache_to_disk(self, smell_type: str):
        """将缓存保存到磁盘"""
        cache_file = self._get_cache_file_path(smell_type)
        
        try:
            # 筛选出该坏味类型的缓存
            smell_cache = {}
            for cache_key, cache_value in self.cache.items():
                # 检查缓存值是否包含坏味类型信息
                if isinstance(cache_value, dict) and 'smell_type' in cache_value:
                    if cache_value['smell_type'] == smell_type:
                        smell_cache[cache_key] = cache_value
                else:
                    # 如果没有明确的坏味类型信息，尝试从缓存键推断
                    if smell_type in cache_key:
                        smell_cache[cache_key] = cache_value
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(smell_cache, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 保存{smell_type}缓存: {len(smell_cache)} 条记录")
        except Exception as e:
            print(f"❌ 保存{smell_type}缓存失败: {e}")
    
    def get_cache_key_by_entity_key(self, entity_key: str, smell_type: str) -> str:
        """通过entity_key查找对应的缓存键"""
        for cache_key, cache_value in self.cache.items():
            if cache_value.get('entity_key') == entity_key and cache_value.get('smell_type') == smell_type:
                return cache_key
        return None
    
    def get_cache_by_entity_key(self, entity_key: str, smell_type: str) -> dict:
        """通过entity_key获取缓存结果"""
        cache_key = self.get_cache_key_by_entity_key(entity_key, smell_type)
        if cache_key:
            return self.cache.get(cache_key)
        return None
    
    def get_all_entity_keys(self, smell_type: str = None) -> list:
        """获取所有entity_key列表，可指定坏味类型过滤"""
        entity_keys = []
        for cache_value in self.cache.values():
            if 'entity_key' in cache_value:
                if smell_type is None or cache_value.get('smell_type') == smell_type:
                    entity_keys.append(cache_value['entity_key'])
        return list(set(entity_keys))  # 去重

    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载小样本学习案例
        
        从dataset/sourceCode目录中加载16个精确的标杆案例（每种坏味4个等级各1个）
        使用用户提供的准确文件路径映射
        """
        examples = {"blob": [], "data_class": [], "feature_envy": [], "long_method": []}
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 使用dataset/sourceCode目录
        source_dir = os.path.join(project_root, "dataset", "sourceCode")
        if not os.path.exists(source_dir):
            print(f"⚠️  目录不存在: {source_dir}")
            return examples
        
        # 根据用户提供的准确文件路径定义16个标杆案例
        benchmark_files = {
            "blob": {
                0: "class__blob__none__0__12957__d6555bf5b0b62aef92be79f5f2fbe00426ebee36__C__3__20.java",
                1: "class__blob__minor__1__12883__4deb681aaaa79c248115037fc8e399c9876619fd__DefaultHotSpotLoweringProvider__184__809.java",
                2: "class__blob__major__1__4028__ac1e6e4035f9307b871478ed47246cf92cfd5f7f__ApplicationResource__63__563.java",
                3: "class__blob__critical__1__1096__a35e3a450b4c0134cb097b9e7de76dca08eb6654__FormatParser__22__1683.java"
            },
            "data_class": {
                0: "class__data_class__none__0__10641__7ba7f3c2e16df6c8db0d8114e124957199cea1ff__MetricsFactory__43__290.java",
                1: "class__data_class__minor__1__13615__1fb059d7e32b9b3514617d54e4dda41ab68e71ea__CallBuilder__179__191.java",
                2: "class__data_class__major__0__11078__f0d9ce06a1a98569a5a4eed76a2ec0aa87c1a1df__Device__3__45.java",
                3: "class__data_class__critical__1__6173__8a85a70643c4d6eec2d3abddeea44ecb06c2f486__RestConfiguration__25__585.java"
            },
            "feature_envy": {
                0: "function__feature_envy__none__0__1011__c6202a55f5f29afb37ffcf876674dca372f3fb4c__Field__101__103.java",
                1: "function__feature_envy__minor__1__8280__52293d20268de7c98833846ded2b70d6476773de__ResolveContext__158__179.java",
                2: "function__feature_envy__major__1__5564__6bf89e9c8804c8845ec4d38583dd33eea8256439__JwkUtils__448__469.java",
                3: "function__feature_envy__critical__1__14667__210e380df3ca5c74c8c2fa09e7fe1cffdb87e20a__DepositProductDataValidator__413__547.java"
            },
            "long_method": {
                0: "function__long_method__none__0__1365__0210210ce436eb83bf200f5d5f9a63a440c5b27a__RestTraversal__53__57.java",
                1: "function__long_method__minor__1__565__a9c1a0661198d9ba37c1facd8385fe05d538c4ad__ELParser__140__168.java",
                2: "function__long_method__major__1__12318__471504a735b48d5d4ace51afa1542cc4790a921a__SAX2DTM2__1658__1719.java",
                3: "function__long_method__critical__1__5777__f344a3c565b6a67233de1d1169104a728136e7a3__DataCreator__158__221.java"
            }
        }
        
        # 精确加载标杆案例
        loaded_count = 0
        for smell_type, severity_files in benchmark_files.items():
            for severity, filename in severity_files.items():
                filepath = os.path.join(source_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code = f.read()
                        
                        # 智能截断：保持方法/类结构完整
                        lines = code.split('\n')
                        if len(lines) > 200:  # 从40行提高到200行
                            # 保留前200行，通常能展示完整结构
                            code = '\n'.join(lines[:200]) + "\n// ... 代码截断，保留核心结构 ..."
                        elif len(code) > 8000:  # 从2000字符提高到8000字符
                            code = code[:8000] + "\n// ... 代码截断 ..."
                        
                        examples[smell_type].append({
                            "code": code,
                            "severity": severity,
                            "severity_name": self._get_severity_name(severity),
                            "filename": filename,
                            "entity_name": self._extract_entity_name(filename)
                        })
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  读取文件失败: {filepath} - {e}")
                else:
                    print(f"❌  文件不存在: {filepath}")
        
        # 打印详细加载统计
        print(f"[OK] 成功加载小样本案例: {loaded_count}/16 个")
        for smell_type, case_list in examples.items():
            severity_counts = {}
            for case in case_list:
                severity_name = case["severity_name"]
                severity_counts[severity_name] = severity_counts.get(severity_name, 0) + 1
            
            print(f"[INFO] {smell_type}: {len(case_list)} 个案例")
            if severity_counts:
                counts_str = ", ".join([f"{k}:{v}" for k, v in severity_counts.items()])
                print(f"     等级分布: {counts_str}")
            
        return examples

    def _get_severity_name(self, severity: int) -> str:
        """获取严重程度名称"""
        mapping = {0: "none", 1: "minor", 2: "major", 3: "critical"}
        return mapping.get(severity, "unknown")
    
    def _extract_entity_name(self, filename: str) -> str:
        """从文件名中提取实体名称"""
        try:
            # 文件名格式: class__blob__none__0__602__...__EntityName__...
            # 或 function__feature_envy__none__0__605__...__MethodName__...
            parts = filename.split("__")
            if len(parts) >= 6:
                return parts[-3]  # 实体名称在倒数第三个位置
            return "Unknown"
        except Exception:
            return "Unknown"

    def _extract_method_name_from_code(self, code_snippet: str, smell_type: str) -> str:
        """从代码片段中提取方法名或类名"""
        try:
            lines = code_snippet.strip().split('\n')
            
            if smell_type in ["long_method", "feature_envy"]:
                # 提取方法名
                for line in lines:
                    line = line.strip()
                    if line.startswith("public") or line.startswith("private") or line.startswith("protected"):
                        # 匹配方法定义
                        method_match = re.search(r'\b(\w+)\s*\([^)]*\)\s*\{', line)
                        if method_match:
                            return method_match.group(1)
                        # 匹配方法名
                        method_name = re.search(r'\b(\w+)\s*\(', line)
                        if method_name:
                            return method_name.group(1)
            else:  # blob, data_class
                # 提取类名
                for line in lines:
                    line = line.strip()
                    if line.startswith("public") or line.startswith("class") or line.startswith("abstract"):
                        class_match = re.search(r'class\s+(\w+)', line)
                        if class_match:
                            return class_match.group(1)
            
            return "未知方法/类"
        except Exception:
            return "未知方法/类"

    def _build_enhanced_prompt(self, smell_type: str, severity_mode: bool, method_name: str) -> str:
        """构建增强提示，强制要求包含方法名和新置信度"""
        if smell_type not in self.few_shot_examples:
            return ""
        
        cases = self.few_shot_examples[smell_type]
        if not cases:
            return ""
        
        prompt_parts = []
        
        if severity_mode:
            # 四级分类模式
            prompt_parts.append("\n【小样本学习案例 - 严重程度判断】")
            prompt_parts.append("请根据以下案例学习如何准确判断代码坏味的严重程度：")
            
            sorted_cases = sorted(cases, key=lambda x: x["severity"])
            
            for case in sorted_cases:
                prompt_parts.append(f"\n--- 严重程度 {case['severity']} ({case['severity_name'].upper()}) ---")
                prompt_parts.append(f"实体: {case['entity_name']}")
                prompt_parts.append("```java")
                prompt_parts.append(case["code"])
                prompt_parts.append("```")
        else:
            # 二分类模式
            prompt_parts.append("\n【小样本学习案例 - 坏味程度谱系】")
            prompt_parts.append("请根据以下四个等级的案例，理解坏味的渐进过程，并准确判断目标代码是否存在坏味：")
            prompt_parts.append("\n【判断指导】")
            prompt_parts.append("- 观察从无坏味到严重坏味的特征渐变")
            prompt_parts.append("- 注意关键指标（如方法长度、数据访问模式等）的变化")
            prompt_parts.append("- 如果目标代码特征接近0级→判断为'否'，接近1-3级→判断为'是'")
            prompt_parts.append("- 置信度根据与四个等级的接近程度综合评估")
            
            cases_by_severity = {0: [], 1: [], 2: [], 3: []}
            for case in cases:
                severity = case["severity"]
                if severity in cases_by_severity:
                    cases_by_severity[severity].append(case)
            
            severity_names = {0: "无坏味", 1: "轻微", 2: "中等", 3: "严重"}
            for severity in [0, 1, 2, 3]:
                if cases_by_severity[severity]:
                    case = cases_by_severity[severity][0]
                    prompt_parts.append(f"\n--- 等级 {severity} ({severity_names[severity]}) ---")
                    prompt_parts.append(f"实体: {case['entity_name']}")
                    prompt_parts.append("```java")
                    prompt_parts.append(case["code"])
                    prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def _call_deepseek_api(self, messages: list) -> Optional[str]:
        """调用硅基流动API - 使用Qwen/Qwen3-8B模型"""
        # 检查API配置是否有效
        if not self.api_key:
            print("⚠️  API密钥未配置，跳过LLM验证")
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,  # Qwen/Qwen3-8B
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 400,  # 减少token数量加快响应
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False  # 禁用流式响应
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # 速率限制
                    wait_time = min(RETRY_DELAY * (2 ** attempt) + 5, 60)  # 最大等待60秒
                    print(f"⚠️  API速率限制(尝试{attempt+1}/{MAX_RETRIES})，等待{wait_time}秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"⚠️  API调用失败({response.status_code}): {response.text}")
                    if attempt < MAX_RETRIES - 1:
                        wait_time = min(RETRY_DELAY * (2 ** attempt), 30)
                        print(f"⏳ 等待{wait_time}秒后重试...")
                        time.sleep(wait_time)
                    continue
                        
            except requests.exceptions.Timeout:
                print(f"⏰ 请求超时(尝试{attempt+1}/{MAX_RETRIES}) - {API_TIMEOUT}秒超时")
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(RETRY_DELAY * (2 ** attempt), 30)
                    print(f"⏳ 等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.ConnectionError:
                print(f"🌐 网络连接错误(尝试{attempt+1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(RETRY_DELAY * (2 ** attempt + 1), 30)
                    print(f"⏳ 等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                print(f"⚠️  API调用异常(尝试{attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(RETRY_DELAY * (2 ** attempt), 30)
                    print(f"⏳ 等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
        
        print("❌ LLM API调用失败，将使用DNN结果")
        return None
    
    def validate_boundary_case(self, 
                             code: str, 
                             smell_type: str, 
                             metrics: dict, 
                             dnn_confidence: float,
                             severity_mode: str = "normal",
                             entity_key: str = None) -> Dict[str, Any]:
        """
        第二层：边界案例验证
        
        Args:
            code: 代码片段
            smell_type: 坏味类型(feature_envy/data_class/long_method/blob)
            dnn_confidence: DNN置信度
            metrics: 关键指标
            
        Returns:
            验证结果字典
        """
        
        # 检查是否触发LLM验证
        if not (LLM_CONFIDENCE_THRESHOLD_LOW <= dnn_confidence <= LLM_CONFIDENCE_THRESHOLD_HIGH):
            return {"use_llm": False, "original_confidence": dnn_confidence}
        
        # 生成缓存键
        cache_key = self._generate_cache_key(code, smell_type, metrics)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 提取方法名/类名
        method_name = self._extract_method_name_from_code(code, smell_type)
        
        # 加载增强的小样本学习案例
        few_shot_prompt = self._build_enhanced_prompt(smell_type, severity_mode, method_name)
        
        # 根据坏味类型定义专门的提示词
        smell_definitions = {
            "feature_envy": {
                "name": "Feature Envy (特性依恋)",
                "definition": """Feature Envy是指一个类的方法过于关注其他类的数据，而忽视了自己类的数据。
                "cognitive_pattern": "该方法表现出对'外部数据'的过度关注，数据耦合度异常高",  
                "intuition": "该方法数据访问模式呈现异常的外部依赖倾向",
                认知特征：
                1. 数据访问模式呈现明显的跨类边界特征
                2. 方法与其所属类的内聚性显著降低,对内部数据的相对忽视
                3. 方法频繁使用其他类的getter方法访问数据
                4. 违反了"数据和行为应该在一起"的原则

                识别要点：
                1. 方法逻辑被外部类数据结构主导（非参数或接口）
                2. 外部类数据访问频率高于自身类属性访问
                3. 方法频繁使用其他类的getter方法访问数据
                4. 方法对其他类的内部结构了解过多
                5. 方法应该属于它所频繁访问的那个类
                6. 将方法迁移到其频繁访问的类中，逻辑会更加自然和清晰""",
                "severity_criteria": {
                    0: "无坏味：方法主要操作本类数据，或对其他类的访问合理",
                    1: "轻微：少量访问其他类数据，但仍在可接受范围内",
                    2: "中等：明显更关注其他类数据，建议考虑迁移方法",
                    3: "严重：方法几乎完全依赖其他类数据，必须重构"
                }
            },
            "data_class": {
                "name": "Data Class (数据类)",
                "definition": """Data Class是指一个类只包含数据字段和简单的getter/setter方法，缺乏有意义的行为。
                 "cognitive_pattern": "这个类呈现数据容器特征，行为封装严重不足",
                  "intuition": "该类的职责划分呈现明显的数据存储偏向",
                认识特征：
                1. 类主要包含公共的实例变量或私有变量+简单的访问方法
                2. 缺乏复杂的行为方法，仅有简单的数据验证
                3. 其他类频繁访问和修改这些数据
                4. 类更像是数据结构而非具有行为的实体
                
                识别要点：
                1. 单纯的DTO/VO类不算Data Class坏味
                2. 如果类有合理的业务逻辑，即使数据多也不算坏味
                3. 关键在于是否过度暴露内部数据且行为贫乏""",
                "severity_criteria": {
                    0: "无坏味：类有合理的行为方法，数据封装良好",
                    1: "轻微：数据为主但有一些简单行为，可接受",
                    2: "中等：明显数据过多行为过少，建议添加相关行为",
                    3: "严重：纯数据结构，无有意义行为，必须重构"
                }
            },
            "long_method": {
                "name": "Long Method (过长方法)",
                "cognitive_pattern": "控制流程复杂，嵌套层次深，职责不单一",
                "definition": """Long Method是指一个方法代码行数过多，逻辑复杂，难以理解和维护,
                "intuition": "该方法的复杂度指标显示明显的认知超载特征",
                           
                认知特征：
                1. 控制流程复杂，嵌套层次深，职责边界模糊
                2. 代码行数超出认知负荷阈值，理解成本显著增加
                3. 方法违反单一职责原则，承担多个不相关任务,包含多个不相关的逻辑片段
                4. 包含多层嵌套条件或循环

                识别要点：
                1. 方法逻辑被多个不相关的任务主导（非参数或接口）
                2. 包含多个不相关的逻辑片段，需要大量注释辅助理解
                3. 可通过提取方法等技术手段有效降低复杂度
                4. 方法体代码行数超过合理阈值（通常>20-30行）""",
                "severity_criteria": {
                    0: "无坏味：方法简洁，职责单一，易于理解",
                    1: "轻微：略长但逻辑清晰，仍可维护",
                    2: "中等：明显过长，建议拆分为多个方法",
                    3: "严重：极长且复杂，急需重构"
                }
            },
            "blob": {
                "name": "Blob (上帝类)",
                "definition": """Blob（上帝类）是指一个类承担了过多的职责，变得过于庞大和复杂。
                "cognitive_pattern": "职责泛化，封装边界模糊，内聚性低",
                 "intuition": "该类的职责划分呈现明显的职责泛化倾向",
                认识特征：
                1. 类包含大量属性和方法（通常>10个方法或>50个属性）
                2. 类涉及多个不相关的功能领域
                3. 其他类对该类有高度依赖
                4. 类违反了单一职责原则和高内聚原则
                
                识别要点：
                - 类是否涉及多个不同的业务领域
                - 方法之间是否缺乏内聚性
                - 是否可以通过提取类的方式分解功能""",
                "severity_criteria": {
                    0: "无坏味：类职责单一，方法内聚性好",
                    1: "轻微：略大但职责相对集中",
                    2: "中等：明显过大，涉及多个领域，建议分解",
                    3: "严重：典型的上帝类，必须分解为多个小类"
                }
            }
        }
        
        smell_info = smell_definitions.get(smell_type, smell_definitions["feature_envy"])
        
        # 根据模式构建验证提示
        if severity_mode:
            messages = [
                {
                    "role": "system",
                    "content": f"""你是一个专业的代码审查专家。请严格根据以下定义评估代码的{smell_info['name']}问题严重程度。
                    
                    {smell_info['definition']}
                    
                    【严重程度分级标准】
                    {chr(10).join([f"{k}级 - {v}" for k, v in smell_info['severity_criteria'].items()])}
                    
                    【重要规则】
                    1. 必须严格按照上述定义判断，不能基于个人偏好
                    2. 重点关注代码的本质特征，而非表面现象
                    3. 给出具体代码示例支持你的判断
                    4. 参考小样本学习案例中的判断标准
                    
                    {few_shot_prompt}
                    
                    【置信度设定指导】
                    新置信度必须精确对应你判断的严重程度等级：
                    - 0级（无坏味）：新置信度 = 0.1-0.3
                    - 1级（轻微）：新置信度 = 0.3-0.5  
                    - 2级（中等）：新置信度 = 0.5-0.7
                    - 3级（严重）：新置信度 = 0.7-0.9
                    
                    【输出格式 - 必须严格遵守】
                    等级：[0/1/2/3]
                    理由：这个代码片段包含{method_name}方法。然后[引用具体代码片段，详细说明符合哪个级别的标准]
                    重构建议：[具体的重构步骤和方法]
                    新置信度：[根据等级给出具体数值]"""
                },
                {
                    "role": "user",
                    "content": f"""
                    待分析代码：
                    ```java
                    {code}
                    ```
                    
                    DNN置信度：{dnn_confidence}
                    关键指标：{json.dumps(metrics, indent=2)}
                    
                    请严格按照以下格式回答：
                    
                    判断：[是/否]  - 基于{smell_info['name']}的精确定义判断是否存在该坏味
                    理由：[详细说明判断依据，包括违反的设计原则和具体表现]
                    新置信度：[0.0-1.0之间的数值] - 基于分析结果给出的新置信度值
                    
                    要求：
                    1. 必须返回"新置信度"字段
                    2. 新置信度应该基于代码实际情况和DNN置信度综合评估
                    3. 如果判断为"是"，新置信度应该高于原始置信度；如果判断为"否"，新置信度应该低于原始置信度
                    4. 请结合小样本案例的学习经验，基于{smell_info['name']}的精确定义进行判断，避免偏离坏味本质。
                    """
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": f"""你是一个专业的代码审查专家。请根据以下精确定义判断代码是否存在{smell_info['name']}问题。
                    
                    {few_shot_prompt}
                    
                    【置信度调整指导】
                    新置信度必须根据你的判断确定性和与案例的相似度来设定，不要总是输出相同的置信度，而是根据确定性调整，判断不要太保守
                    
                    当判断为"是"（存在坏味）时：
                    - 如果代码特征与案例中的1-3级非常相似：新置信度 = 0.65-0.75
                    - 如果代码特征与案例中的1-3级部分相似：新置信度 = 0.55-0.65
                    - 如果判断较模糊但倾向于"是"：新置信度 = 0.5-0.55
                    
                    当判断为"否"（无坏味）时：
                    - 如果代码特征与案例中的0级非常相似：新置信度 = 0.2-0.3
                    - 如果代码特征接近0级但略有疑问：新置信度 = 0.3-0.45
                    - 如果判断较模糊但倾向于"否"：新置信度 = 0.45-0.5
                    
                    【重要】新置信度必须是一个具体的数值，不能简单设为固定值，不要总是输出同一个置信度（特别注意，最少也要有细微的调整，具体到小数点后两位）。请根据实际判断确定性和代码特征相似度给出精确值，区间范围内的具体值根据确定性和相似度的情况具体考虑。特别注意不要过于保守，总是将可能存在坏味的情况判断为无坏味。
                    
                    【输出格式 - 必须严格遵守】
                    判断：[是/否]
                    理由：这个代码片段包含{method_name}方法。然后[引用具体代码片段，详细说明符合/不符合定义的地方]
                    违反的原则：[如果为"是"，详细说明违反的设计原则]。如果参考了小样本学习案例，请说明参考的案例名称（如等级2的案例SentryHDFSService、等级0案例ErrorMessages，具体案例根据参考案例输出）和判断原因。
                    新置信度：[根据上述指导给出具体数值]，请一定要返回新置信度"""
                },
                {
                    "role": "user",
                    "content": f"""
                    待分析代码：
                    ```java
                    {code}
                    ```
                    
                    DNN置信度：{dnn_confidence}
                    关键指标：{json.dumps(metrics, indent=2)}
                    
                    请结合小样本案例的学习经验，基于{smell_info['name']}的精确定义进行判断，避免偏离坏味本质。"""
                }
            ]
        
        llm_response = self._call_deepseek_api(messages)
        
        if llm_response:
            result = self._parse_llm_response(llm_response, dnn_confidence, severity_mode)
            # 添加坏味类型信息到结果中
            result['smell_type'] = smell_type
            
            # 只有当LLM成功调用且返回有效结果时才保存到缓存
            # 检查条件：1. use_llm为True 2. llm_judgment为'是'或'否' 3. 不是fallback到原始值的情况
            if (result.get('use_llm', False) and 
                result.get('llm_judgment') in ['是', '否'] and
                result.get('confidence_source') == 'llm_returned' and
                not result.get('parsing_warning', '')):
                # 添加entity_key到缓存结果中（如果传入）
                if entity_key is not None:
                    result['entity_key'] = entity_key
                
                self.cache[cache_key] = result
                # 保存缓存到磁盘
                self._save_cache_to_disk(smell_type)
                print(f"✅ LLM验证成功，保存到缓存: {cache_key}")
            else:
                # 检查具体的失败原因
                failure_reason = "未知原因"
                if not result.get('use_llm', False):
                    failure_reason = "use_llm为False"
                elif result.get('confidence_source') == 'fallback_to_original':
                    failure_reason = "fallback到原始值"
                elif result.get('parsing_warning', ''):
                    failure_reason = f"解析警告: {result.get('parsing_warning', '')}"
                else:
                    failure_reason = "判断字段无效"
                
                print(f"⚠️ LLM验证失败，不保存到缓存: {cache_key} - 原因: {failure_reason}")
            
            return result
        else:
            # 区分不同类型的失败原因
            api_failure_reason = "未知原因"
            if not self.api_key:
                api_failure_reason = "API密钥未配置"
            else:
                api_failure_reason = "API调用失败"
            
            return {
                "use_llm": False, 
                "error": f"解析失败: {str(e)}",
                "original_confidence": original_confidence,
                "llm_judgment": "否",  # 异常情况下使用默认值
                "llm_suggested_confidence": original_confidence  # 异常时也使用原始置信度
            }
    
    def _parse_llm_response(self, response: str, original_confidence: float, severity_mode: bool = False) -> Dict[str, Any]:
        """解析LLM响应并修复置信度调整逻辑，支持多种响应格式"""
        try:
            lines = response.strip().split('\n')
            result = {
                "use_llm": True,
                "original_confidence": original_confidence,
                "llm_raw_response": response,
                "severity_mode": severity_mode,
                "few_shot_used": len(self.few_shot_examples.get("blob", [])) > 0  # 检查是否使用了小样本提示
            }
            
            llm_judgment = None
            llm_suggested_confidence = None
            
            for line in lines:
                if severity_mode:
                    if line.startswith("等级：") or line.startswith("### 等级："):
                        try:
                            level_text = line.split("：")[1].strip()
                            result["severity_level"] = int(level_text)
                        except ValueError:
                            result["severity_level"] = 0
                else:
                    # 支持多种判断字段格式
                    judgment_patterns = ["判断：", "### 判断：", "判断:", "### 判断:"]
                    for pattern in judgment_patterns:
                        if line.startswith(pattern):
                            # 严格清理判断值，确保只有"是"或"否"
                            judgment_value = line.split(pattern)[1].strip()
                            # 移除中括号、引号、空格等
                            judgment_value = judgment_value.strip('[]"\'')
                            # 确保只有"是"或"否"
                            if judgment_value in ["是", "否"]:
                                llm_judgment = judgment_value
                            elif judgment_value == "0" or judgment_value.lower() in ["no", "false"]:
                                llm_judgment = "否"
                            elif judgment_value == "1" or judgment_value.lower() in ["yes", "true"]:
                                llm_judgment = "是"
                            else:
                                # 如果格式不正确，使用默认值
                                llm_judgment = "否"
                                result["parsing_warning"] = f"判断格式异常，使用默认值: {judgment_value}"
                            
                            result["llm_judgment"] = llm_judgment
                            break
                        
                # 支持多种理由字段格式
                reason_patterns = ["理由：", "### 理由：", "理由:", "### 理由:"]
                for pattern in reason_patterns:
                    if line.startswith(pattern):
                        result["reason"] = line.split(pattern, 1)[1].strip()
                        break
                
                # 支持多种建议字段格式
                suggestion_patterns = ["建议：", "### 建议：", "建议:", "### 建议:"]
                for pattern in suggestion_patterns:
                    if line.startswith(pattern):
                        result["suggestion"] = line.split(pattern, 1)[1].strip()
                        break
                
                # 支持多种新置信度字段格式
                confidence_patterns = ["新置信度：", "### 新置信度：", "新置信度:", "### 新置信度:"]
                for pattern in confidence_patterns:
                    if line.startswith(pattern):
                        try:
                            llm_suggested_confidence = float(line.split(pattern)[1].strip())
                        except ValueError:
                            llm_suggested_confidence = None
                        break
            
            # 确保llm_judgment有值
            if llm_judgment is None:
                # 尝试从响应中提取判断 - 更宽松的匹配
                for line in lines:
                    # 检查各种可能的判断格式
                    if any(pattern in line for pattern in ["判断：是", "判断:是", "判断： 是", "判断: 是"]):
                        llm_judgment = "是"
                        break
                    elif any(pattern in line for pattern in ["判断：否", "判断:否", "判断： 否", "判断: 否"]):
                        llm_judgment = "否"
                        break
                    # 检查英文判断
                    elif any(pattern in line for pattern in ["判断：yes", "判断:yes", "判断： yes", "判断: yes"]):
                        llm_judgment = "是"
                        break
                    elif any(pattern in line for pattern in ["判断：no", "判断:no", "判断： no", "判断: no"]):
                        llm_judgment = "否"
                        break
                    # 检查数字判断
                    elif any(pattern in line for pattern in ["判断：1", "判断:1", "判断： 1", "判断: 1"]):
                        llm_judgment = "是"
                        break
                    elif any(pattern in line for pattern in ["判断：0", "判断:0", "判断： 0", "判断: 0"]):
                        llm_judgment = "否"
                        break
                    # 检查Markdown格式
                    elif any(pattern in line for pattern in ["### 判断：是", "### 判断:是", "### 判断： 是", "### 判断: 是"]):
                        llm_judgment = "是"
                        break
                    elif any(pattern in line for pattern in ["### 判断：否", "### 判断:否", "### 判断： 否", "### 判断: 否"]):
                        llm_judgment = "否"
                        break
                
                if llm_judgment is None:
                    # 最后尝试简单的关键词匹配
                    response_lower = response.lower()
                    if "是" in response and "否" not in response:
                        llm_judgment = "是"
                    elif "否" in response and "是" not in response:
                        llm_judgment = "否"
                    elif "yes" in response_lower and "no" not in response_lower:
                        llm_judgment = "是"
                    elif "no" in response_lower and "yes" not in response_lower:
                        llm_judgment = "否"
                    else:
                        llm_judgment = "否"
                        result["parsing_warning"] = "未找到判断字段，使用默认值"
            
            result["llm_judgment"] = llm_judgment

            # 修复：确保llm_suggested_confidence始终有值，未返回时使用原始置信度
            if llm_suggested_confidence is None:
                llm_suggested_confidence = original_confidence
                result["llm_suggested_confidence"] = llm_suggested_confidence
                result["confidence_source"] = "fallback_to_original"  # 标记为回退到原始值
            else:
                result["llm_suggested_confidence"] = llm_suggested_confidence
                result["confidence_source"] = "llm_returned"  # 标记为LLM返回的值

            # 注意：置信度调整逻辑已移至调用方（enhanced_detection.py）
            # 这里只返回LLM的原始判断和建议置信度，由调用方决定如何融合

            return result
            
        except Exception as e:
            return {
                "use_llm": False, 
                "error": f"解析失败: {str(e)}",
                "original_confidence": original_confidence,
                "llm_judgment": "否",  # 异常情况下使用默认值
                "llm_suggested_confidence": original_confidence  # 异常时也使用原始置信度
            }

    def explain_high_risk_case(self, 
                             code_snippet: str, 
                             smell_type: str,
                             dnn_confidence: float) -> Optional[str]:
        """
        第三层：高风险案例解释
        仅当dnn_confidence > 0.9时触发
        """
        
        if dnn_confidence <= 0.9:
            return None
        
        messages = [
            {
                "role": "system",
                "content": f"""你是一个资深架构师。请为以下高置信度{smell_type}案例提供详细解释和改进建议。
                
                要求：
                1. 解释为什么这个代码存在坏味
                2. 分析可能的设计问题
                3. 用中文回答"""
            },
            {
                "role": "user",
                "content": f"""
                代码：
                ```java
                {code_snippet}
                ```
                
                DNN置信度：{dnn_confidence}
                """
            }
        ]
        
        return self._call_deepseek_api(messages)

# 全局验证器实例
llm_validator = LLMValidator()
