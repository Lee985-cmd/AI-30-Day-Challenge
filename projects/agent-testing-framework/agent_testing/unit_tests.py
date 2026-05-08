"""
单元测试工具

用于测试Agent系统的各个组件：
- Prompt模板
- 工具函数
- 输出解析器
"""

from typing import List, Dict, Any, Callable, Optional
import time


class TestPrompt:
    """Prompt模板测试类"""
    
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.test_cases = []
        
    def add_test_case(self, input_vars: Dict[str, str], expected_keywords: List[str] = None):
        """添加测试用例
        
        Args:
            input_vars: 输入变量
            expected_keywords: 期望出现的关键字（可选）
        """
        self.test_cases.append({
            "input": input_vars,
            "expected_keywords": expected_keywords or []
        })
        
    def run_tests(self) -> Dict[str, Any]:
        """运行所有测试用例
        
        Returns:
            测试结果字典
        """
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(self.test_cases):
            try:
                # 格式化Prompt
                formatted_prompt = self.prompt_template.format(**test_case["input"])
                
                # 检查关键字
                passed = True
                missing_keywords = []
                
                for keyword in test_case["expected_keywords"]:
                    if keyword.lower() not in formatted_prompt.lower():
                        passed = False
                        missing_keywords.append(keyword)
                
                result = {
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "formatted_prompt": formatted_prompt[:200] + "..." if len(formatted_prompt) > 200 else formatted_prompt,
                    "passed": passed,
                    "missing_keywords": missing_keywords
                }
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append(result)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "error": str(e),
                    "passed": False
                })
        
        return results


class TestToolFunction:
    """工具函数测试类"""
    
    def __init__(self, tool_func: Callable):
        self.tool_func = tool_func
        self.test_cases = []
        
    def add_test_case(self, input_args: Dict[str, Any], expected_type: type = None, 
                     expected_value: Any = None, should_raise: bool = False):
        """添加测试用例
        
        Args:
            input_args: 输入参数
            expected_type: 期望返回类型
            expected_value: 期望返回值
            should_raise: 是否应该抛出异常
        """
        self.test_cases.append({
            "input": input_args,
            "expected_type": expected_type,
            "expected_value": expected_value,
            "should_raise": should_raise
        })
        
    def run_tests(self) -> Dict[str, Any]:
        """运行所有测试用例
        
        Returns:
            测试结果字典
        """
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(self.test_cases):
            try:
                start_time = time.time()
                
                # 执行工具函数
                result = self.tool_func(**test_case["input"])
                execution_time = time.time() - start_time
                
                # 检查是否应该抛出异常
                if test_case["should_raise"]:
                    passed = False
                    error = "Expected exception but none was raised"
                else:
                    passed = True
                    error = None
                    
                    # 检查返回类型
                    if test_case["expected_type"] and not isinstance(result, test_case["expected_type"]):
                        # 允许int和float互相兼容
                        if not (isinstance(result, (int, float)) and test_case["expected_type"] in (int, float)):
                            passed = False
                            error = f"Expected type {test_case['expected_type']}, got {type(result)}"
                    
                    # 检查返回值
                    if test_case["expected_value"] is not None and result != test_case["expected_value"]:
                        passed = False
                        error = f"Expected value {test_case['expected_value']}, got {result}"
                
                result_detail = {
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "output": str(result)[:200],
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "passed": passed,
                    "error": error
                }
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append(result_detail)
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 如果期望抛出异常，则测试通过
                if test_case["should_raise"]:
                    results["passed"] += 1
                    results["details"].append({
                        "case_id": i + 1,
                        "input": test_case["input"],
                        "exception": str(e),
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "passed": True
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "case_id": i + 1,
                        "input": test_case["input"],
                        "error": str(e),
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "passed": False
                    })
        
        return results


class TestParser:
    """输出解析器测试类"""
    
    def __init__(self, parser_func: Callable):
        self.parser_func = parser_func
        self.test_cases = []
        
    def add_test_case(self, raw_output: str, expected_result: Any = None,
                     should_fail: bool = False):
        """添加测试用例
        
        Args:
            raw_output: 原始输出
            expected_result: 期望解析结果
            should_fail: 是否应该解析失败
        """
        self.test_cases.append({
            "raw_output": raw_output,
            "expected_result": expected_result,
            "should_fail": should_fail
        })
        
    def run_tests(self) -> Dict[str, Any]:
        """运行所有测试用例
        
        Returns:
            测试结果字典
        """
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(self.test_cases):
            try:
                start_time = time.time()
                
                # 执行解析
                parsed_result = self.parser_func(test_case["raw_output"])
                execution_time = time.time() - start_time
                
                # 检查是否应该失败
                if test_case["should_fail"]:
                    passed = False
                    error = "Expected parsing to fail but it succeeded"
                else:
                    passed = True
                    error = None
                    
                    # 检查解析结果
                    if test_case["expected_result"] is not None:
                        if parsed_result != test_case["expected_result"]:
                            passed = False
                            error = f"Expected {test_case['expected_result']}, got {parsed_result}"
                
                result_detail = {
                    "case_id": i + 1,
                    "raw_output": test_case["raw_output"][:100] + "...",
                    "parsed_result": str(parsed_result)[:200],
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "passed": passed,
                    "error": error
                }
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append(result_detail)
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 如果期望失败，则测试通过
                if test_case["should_fail"]:
                    results["passed"] += 1
                    results["details"].append({
                        "case_id": i + 1,
                        "raw_output": test_case["raw_output"][:100] + "...",
                        "exception": str(e),
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "passed": True
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "case_id": i + 1,
                        "raw_output": test_case["raw_output"][:100] + "...",
                        "error": str(e),
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "passed": False
                    })
        
        return results
