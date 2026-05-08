"""
A/B测试框架

用于对比不同Agent配置或模型的效果
"""

from typing import List, Dict, Any, Callable, Optional
import numpy as np
from scipy import stats


class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self):
        self.experiments = []
        
    def create_experiment(self, name: str, variant_a: Callable, variant_b: Callable,
                         test_cases: List[Dict[str, Any]]):
        """创建A/B测试实验
        
        Args:
            name: 实验名称
            variant_a: 变体A（对照组）
            variant_b: 变体B（实验组）
            test_cases: 测试用例列表
        """
        self.experiments.append({
            "name": name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "test_cases": test_cases,
            "results_a": [],
            "results_b": []
        })
        
    def run_experiment(self, experiment_index: int = 0) -> Dict[str, Any]:
        """运行A/B测试实验
        
        Args:
            experiment_index: 实验索引
            
        Returns:
            实验结果
        """
        if experiment_index >= len(self.experiments):
            return {"error": "实验索引超出范围"}
        
        experiment = self.experiments[experiment_index]
        
        # 运行变体A
        for test_case in experiment["test_cases"]:
            try:
                result = experiment["variant_a"](test_case)
                experiment["results_a"].append(result)
            except Exception as e:
                experiment["results_a"].append({"error": str(e)})
        
        # 运行变体B
        for test_case in experiment["test_cases"]:
            try:
                result = experiment["variant_b"](test_case)
                experiment["results_b"].append(result)
            except Exception as e:
                experiment["results_b"].append({"error": str(e)})
        
        # 统计分析
        return self._analyze_results(experiment)
    
    def _analyze_results(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验结果
        
        Args:
            experiment: 实验数据
            
        Returns:
            分析结果
        """
        results_a = experiment["results_a"]
        results_b = experiment["results_b"]
        
        # 提取指标（假设结果是数值型）
        metrics_a = self._extract_metrics(results_a)
        metrics_b = self._extract_metrics(results_b)
        
        if not metrics_a or not metrics_b:
            return {
                "experiment_name": experiment["name"],
                "error": "无法提取有效指标"
            }
        
        # 计算统计量
        mean_a = np.mean(metrics_a)
        mean_b = np.mean(metrics_b)
        std_a = np.std(metrics_a)
        std_b = np.std(metrics_b)
        
        # T检验
        if len(metrics_a) >= 2 and len(metrics_b) >= 2:
            t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)
        else:
            t_stat, p_value = None, None
        
        # 提升百分比
        improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
        
        # 判断显著性
        significant = p_value < 0.05 if p_value is not None else False
        winner = "B" if (mean_b > mean_a and significant) else ("A" if significant else "Tie")
        
        return {
            "experiment_name": experiment["name"],
            "sample_size": len(metrics_a),
            "variant_a": {
                "mean": round(mean_a, 4),
                "std": round(std_a, 4),
                "min": round(min(metrics_a), 4),
                "max": round(max(metrics_a), 4)
            },
            "variant_b": {
                "mean": round(mean_b, 4),
                "std": round(std_b, 4),
                "min": round(min(metrics_b), 4),
                "max": round(max(metrics_b), 4)
            },
            "improvement_percent": round(improvement, 2),
            "t_statistic": round(t_stat, 4) if t_stat is not None else None,
            "p_value": round(p_value, 6) if p_value is not None else None,
            "statistically_significant": significant,
            "winner": winner,
            "recommendation": self._get_recommendation(winner, improvement, significant)
        }
    
    def _extract_metrics(self, results: List[Any]) -> List[float]:
        """从结果中提取数值指标
        
        Args:
            results: 结果列表
            
        Returns:
            数值指标列表
        """
        metrics = []
        
        for result in results:
            if isinstance(result, dict):
                # 优先提取quality_score，其次是score、accuracy等
                for key in ["quality_score", "score", "accuracy", "response_time"]:
                    if key in result and isinstance(result[key], (int, float)):
                        metrics.append(float(result[key]))
                        break
            elif isinstance(result, (int, float)):
                metrics.append(float(result))
        
        return metrics
    
    def _get_recommendation(self, winner: str, improvement: float, 
                          significant: bool) -> str:
        """获取建议
        
        Args:
            winner: 获胜者
            improvement: 提升百分比
            significant: 是否显著
            
        Returns:
            建议字符串
        """
        if not significant:
            return "差异不显著，建议继续收集数据或使用当前方案"
        
        if winner == "A":
            return f"变体A表现更好，建议使用变体A"
        elif winner == "B":
            return f"变体B表现更好（提升{improvement:.2f}%），建议使用变体B"
        else:
            return "两个变体表现相当，可任选其一"
    
    def get_all_experiments_summary(self) -> List[Dict[str, Any]]:
        """获取所有实验的摘要
        
        Returns:
            实验摘要列表
        """
        summaries = []
        
        for i, experiment in enumerate(self.experiments):
            if experiment["results_a"] and experiment["results_b"]:
                summary = self._analyze_results(experiment)
                summaries.append(summary)
            else:
                summaries.append({
                    "experiment_name": experiment["name"],
                    "status": "未运行"
                })
        
        return summaries
