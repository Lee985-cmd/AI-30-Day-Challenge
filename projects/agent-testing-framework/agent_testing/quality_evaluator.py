"""
质量评估工具

用于评估Agent系统的输出质量：
- 基础质量指标（准确性、相关性、完整性）
- RAGAS指标（faithfulness, answer_relevance, context_relevance等）
"""

from typing import List, Dict, Any, Optional
import numpy as np


class QualityEvaluator:
    """基础质量评估器"""
    
    def __init__(self):
        self.evaluations = []
        
    def evaluate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """评估准确性
        
        Args:
            predicted: 预测答案
            ground_truth: 标准答案
            
        Returns:
            准确性得分 (0-1)
        """
        # 使用更宽松的关键词匹配
        pred_lower = predicted.lower()
        truth_lower = ground_truth.lower()
        
        # 提取ground_truth中的关键词（长度>=2的词语）
        truth_words = [word for word in truth_lower.split() if len(word) >= 2]
        
        if not truth_words:
            return 0.0
        
        # 计算匹配度
        matched = sum(1 for word in truth_words if word in pred_lower)
        accuracy = matched / len(truth_words)
        
        return min(accuracy, 1.0)
    
    def evaluate_relevance(self, query: str, response: str) -> float:
        """评估相关性
        
        Args:
            query: 用户问题
            response: Agent回答
            
        Returns:
            相关性得分 (0-1)
        """
        # 检查回答是否包含问题的关键内容
        query_lower = query.lower()
        response_lower = response.lower()
        
        # 提取query中的关键词（长度>=2）
        query_words = [word for word in query_lower.split() if len(word) >= 2]
        
        if not query_words:
            return 0.0
        
        # 计算匹配度
        matched = sum(1 for word in query_words if word in response_lower)
        relevance = matched / len(query_words)
        
        return min(relevance, 1.0)
    
    def evaluate_completeness(self, response: str, expected_points: List[str]) -> float:
        """评估完整性
        
        Args:
            response: Agent回答
            expected_points: 期望覆盖的要点列表
            
        Returns:
            完整性得分 (0-1)
        """
        if not expected_points:
            return 1.0
        
        covered_points = 0
        for point in expected_points:
            if point.lower() in response.lower():
                covered_points += 1
        
        completeness = covered_points / len(expected_points)
        
        return min(completeness, 1.0)
    
    def evaluate_response_time(self, response_time_ms: float, max_time_ms: float = 3000) -> float:
        """评估响应时间
        
        Args:
            response_time_ms: 实际响应时间（毫秒）
            max_time_ms: 最大可接受时间（毫秒）
            
        Returns:
            时间得分 (0-1)，越快得分越高
        """
        if response_time_ms <= 0:
            return 1.0
        
        if response_time_ms > max_time_ms:
            return 0.0
        
        # 线性评分：越接近0越好
        score = 1 - (response_time_ms / max_time_ms)
        
        return max(score, 0.0)
    
    def comprehensive_evaluation(self, query: str, response: str, ground_truth: str,
                                expected_points: List[str], response_time_ms: float,
                                weights: Dict[str, float] = None) -> Dict[str, Any]:
        """综合评估
        
        Args:
            query: 用户问题
            response: Agent回答
            ground_truth: 标准答案
            expected_points: 期望覆盖的要点
            response_time_ms: 响应时间
            weights: 各指标权重
            
        Returns:
            评估结果字典
        """
        # 默认权重
        if weights is None:
            weights = {
                "accuracy": 0.3,
                "relevance": 0.25,
                "completeness": 0.25,
                "response_time": 0.2
            }
        
        # 计算各项指标
        accuracy = self.evaluate_accuracy(response, ground_truth)
        relevance = self.evaluate_relevance(query, response)
        completeness = self.evaluate_completeness(response, expected_points)
        time_score = self.evaluate_response_time(response_time_ms)
        
        # 加权总分
        total_score = (
            accuracy * weights["accuracy"] +
            relevance * weights["relevance"] +
            completeness * weights["completeness"] +
            time_score * weights["response_time"]
        )
        
        result = {
            "accuracy": round(accuracy, 4),
            "relevance": round(relevance, 4),
            "completeness": round(completeness, 4),
            "response_time_score": round(time_score, 4),
            "total_score": round(total_score, 4),
            "grade": self._score_to_grade(total_score)
        }
        
        self.evaluations.append(result)
        
        return result
    
    def get_average_scores(self) -> Dict[str, float]:
        """获取平均分数
        
        Returns:
            各项指标的平均分
        """
        if not self.evaluations:
            return {}
        
        avg_scores = {
            "accuracy": np.mean([e["accuracy"] for e in self.evaluations]),
            "relevance": np.mean([e["relevance"] for e in self.evaluations]),
            "completeness": np.mean([e["completeness"] for e in self.evaluations]),
            "response_time_score": np.mean([e["response_time_score"] for e in self.evaluations]),
            "total_score": np.mean([e["total_score"] for e in self.evaluations])
        }
        
        return {k: round(v, 4) for k, v in avg_scores.items()}
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级
        
        Args:
            score: 总分 (0-1)
            
        Returns:
            等级字符串
        """
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "D"


class RAGASEvaluator:
    """RAGAS指标评估器（需要安装ragas库）"""
    
    def __init__(self):
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevance, context_relevance
            self.ragas_available = True
            self.evaluate = evaluate
            self.metrics = [faithfulness, answer_relevance, context_relevance]
        except ImportError:
            self.ragas_available = False
            print("⚠️  RAGAS库未安装，请运行: pip install ragas")
    
    def evaluate_rag_quality(self, questions: List[str], answers: List[str], 
                           contexts: List[List[str]], 
                           ground_truths: List[str] = None) -> Dict[str, Any]:
        """评估RAG系统质量
        
        Args:
            questions: 问题列表
            answers: 答案列表
            contexts: 上下文列表（每个问题对应的检索文档）
            ground_truths: 标准答案列表（可选）
            
        Returns:
            RAGAS评估结果
        """
        if not self.ragas_available:
            return {"error": "RAGAS库未安装"}
        
        try:
            from datasets import Dataset
            
            # 构建数据集
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts
            }
            
            if ground_truths:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # 执行评估
            result = self.evaluate(dataset, metrics=self.metrics)
            
            # 转换为字典
            scores = result.scores.to_pandas().mean().to_dict()
            
            return {
                "faithfulness": round(scores.get("faithfulness", 0), 4),
                "answer_relevance": round(scores.get("answer_relevance", 0), 4),
                "context_relevance": round(scores.get("context_relevance", 0), 4),
                "overall_quality": round(np.mean(list(scores.values())), 4)
            }
            
        except Exception as e:
            return {"error": str(e)}
