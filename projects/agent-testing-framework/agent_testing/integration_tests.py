"""
集成测试工具

用于测试Agent系统的整体流程：
- RAG系统端到端测试
- Agent工作流测试
"""

from typing import List, Dict, Any, Optional
import time


class TestRAGSystem:
    """RAG系统集成测试类"""
    
    def __init__(self, rag_system):
        """
        Args:
            rag_system: RAG系统实例，需要实现 query() 方法
        """
        self.rag_system = rag_system
        self.test_cases = []
        
    def add_test_case(self, query: str, expected_keywords: List[str] = None,
                     max_response_time_ms: float = 3000):
        """添加测试用例
        
        Args:
            query: 查询问题
            expected_keywords: 期望在回答中出现的关键字
            max_response_time_ms: 最大响应时间（毫秒）
        """
        self.test_cases.append({
            "query": query,
            "expected_keywords": expected_keywords or [],
            "max_response_time_ms": max_response_time_ms
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
            "avg_response_time_ms": 0,
            "details": []
        }
        
        total_response_time = 0
        
        for i, test_case in enumerate(self.test_cases):
            try:
                start_time = time.time()
                
                # 执行查询
                response = self.rag_system.query(test_case["query"])
                response_time_ms = (time.time() - start_time) * 1000
                total_response_time += response_time_ms
                
                # 检查响应时间
                time_ok = response_time_ms <= test_case["max_response_time_ms"]
                
                # 检查关键字
                keyword_matches = []
                missing_keywords = []
                
                for keyword in test_case["expected_keywords"]:
                    if keyword.lower() in response.lower():
                        keyword_matches.append(keyword)
                    else:
                        missing_keywords.append(keyword)
                
                keywords_ok = len(missing_keywords) == 0
                
                # 综合判断
                passed = time_ok and keywords_ok
                
                result_detail = {
                    "case_id": i + 1,
                    "query": test_case["query"],
                    "response": response[:300] + "..." if len(response) > 300 else response,
                    "response_time_ms": round(response_time_ms, 2),
                    "time_ok": time_ok,
                    "keyword_matches": keyword_matches,
                    "missing_keywords": missing_keywords,
                    "keywords_ok": keywords_ok,
                    "passed": passed
                }
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append(result_detail)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "case_id": i + 1,
                    "query": test_case["query"],
                    "error": str(e),
                    "passed": False
                })
        
        # 计算平均响应时间
        if results["total"] > 0:
            results["avg_response_time_ms"] = round(total_response_time / results["total"], 2)
        
        return results


class TestAgentWorkflow:
    """Agent工作流测试类"""
    
    def __init__(self, agent_workflow):
        """
        Args:
            agent_workflow: Agent工作流实例，需要实现 run() 方法
        """
        self.agent_workflow = agent_workflow
        self.test_cases = []
        
    def add_test_case(self, input_data: Dict[str, Any], 
                     expected_steps: List[str] = None,
                     max_execution_time_ms: float = 10000):
        """添加测试用例
        
        Args:
            input_data: 输入数据
            expected_steps: 期望执行的工作流步骤
            max_execution_time_ms: 最大执行时间（毫秒）
        """
        self.test_cases.append({
            "input": input_data,
            "expected_steps": expected_steps or [],
            "max_execution_time_ms": max_execution_time_ms
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
            "avg_execution_time_ms": 0,
            "details": []
        }
        
        total_execution_time = 0
        
        for i, test_case in enumerate(self.test_cases):
            try:
                start_time = time.time()
                
                # 执行工作流
                workflow_result = self.agent_workflow.run(test_case["input"])
                execution_time_ms = (time.time() - start_time) * 1000
                total_execution_time += execution_time_ms
                
                # 检查执行时间
                time_ok = execution_time_ms <= test_case["max_execution_time_ms"]
                
                # 检查工作流步骤（如果有追踪）
                steps_ok = True
                executed_steps = workflow_result.get("steps", [])
                missing_steps = []
                
                if test_case["expected_steps"]:
                    for step in test_case["expected_steps"]:
                        if step not in executed_steps:
                            steps_ok = False
                            missing_steps.append(step)
                
                # 综合判断
                passed = time_ok and steps_ok
                
                result_detail = {
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "output": str(workflow_result.get("result", ""))[:300],
                    "execution_time_ms": round(execution_time_ms, 2),
                    "time_ok": time_ok,
                    "executed_steps": executed_steps,
                    "missing_steps": missing_steps,
                    "steps_ok": steps_ok,
                    "passed": passed
                }
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append(result_detail)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "error": str(e),
                    "passed": False
                })
        
        # 计算平均执行时间
        if results["total"] > 0:
            results["avg_execution_time_ms"] = round(total_execution_time / results["total"], 2)
        
        return results
