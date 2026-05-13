"""
框架性能基准测试
对比LangChain、LlamaIndex和Haystack的性能指标
"""

import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# 加载环境变量
load_dotenv()


def benchmark_langchain():
    """基准测试LangChain"""
    print("\n测试 LangChain...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template("请用一句话回答：{question}")
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # 测试10次查询
        questions = [f"问题{i}" for i in range(10)]
        
        start_time = time.time()
        for question in questions:
            try:
                chain.run(question=question)
            except:
                pass
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(questions)
        
        return {
            "framework": "LangChain",
            "avg_response_time": avg_time,
            "total_time": end_time - start_time,
            "queries_per_second": len(questions) / (end_time - start_time),
            "success": True
        }
    except Exception as e:
        print(f"LangChain测试失败: {str(e)}")
        return {
            "framework": "LangChain",
            "avg_response_time": 0,
            "total_time": 0,
            "queries_per_second": 0,
            "success": False
        }


def benchmark_llamaindex():
    """基准测试LlamaIndex"""
    print("\n测试 LlamaIndex...")
    
    try:
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        llm = OpenAI(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 创建简单索引
        documents = [Document(text=f"文档{i}的内容") for i in range(10)]
        index = VectorStoreIndex.from_documents(documents)
        
        query_engine = index.as_query_engine(llm=llm)
        
        # 测试10次查询
        questions = [f"问题{i}" for i in range(10)]
        
        start_time = time.time()
        for question in questions:
            try:
                query_engine.query(question)
            except:
                pass
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(questions)
        
        return {
            "framework": "LlamaIndex",
            "avg_response_time": avg_time,
            "total_time": end_time - start_time,
            "queries_per_second": len(questions) / (end_time - start_time),
            "success": True
        }
    except Exception as e:
        print(f"LlamaIndex测试失败: {str(e)}")
        return {
            "framework": "LlamaIndex",
            "avg_response_time": 0,
            "total_time": 0,
            "queries_per_second": 0,
            "success": False
        }


def benchmark_haystack():
    """基准测试Haystack"""
    print("\n测试 Haystack...")
    
    try:
        from haystack import Pipeline, Document
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.components.builders.prompt_builder import PromptBuilder
        from haystack.components.generators import OpenAIGenerator
        from haystack.utils import Secret
        
        document_store = InMemoryDocumentStore()
        documents = [Document(content=f"文档{i}的内容") for i in range(10)]
        document_store.write_documents(documents)
        
        retriever = InMemoryBM25Retriever(document_store=document_store)
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            generator = OpenAIGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo")
            
            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)
            pipeline.add_component("prompt_builder", PromptBuilder(template="问题：{{query}}\n回答："))
            pipeline.add_component("generator", generator)
            pipeline.connect("retriever", "prompt_builder.documents")
            pipeline.connect("prompt_builder", "generator")
            
            # 测试10次查询
            questions = [f"问题{i}" for i in range(10)]
            
            start_time = time.time()
            for question in questions:
                try:
                    pipeline.run({"retriever": {"query": question}, "prompt_builder": {"query": question}})
                except:
                    pass
            end_time = time.time()
            
            avg_time = (end_time - start_time) / len(questions)
        else:
            # 无API Key时只测试检索
            start_time = time.time()
            for i in range(10):
                retriever.run(query=f"问题{i}", top_k=2)
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
        
        return {
            "framework": "Haystack",
            "avg_response_time": avg_time,
            "total_time": end_time - start_time,
            "queries_per_second": 10 / (end_time - start_time),
            "success": True
        }
    except Exception as e:
        print(f"Haystack测试失败: {str(e)}")
        return {
            "framework": "Haystack",
            "avg_response_time": 0,
            "total_time": 0,
            "queries_per_second": 0,
            "success": False
        }


def run_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("开始框架性能基准测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(benchmark_langchain())
    results.append(benchmark_llamaindex())
    results.append(benchmark_haystack())
    
    # 打印结果
    print("\n" + "=" * 60)
    print("基准测试结果")
    print("=" * 60)
    
    print(f"\n{'框架':<15} {'平均响应时间(s)':<15} {'QPS':<10} {'状态':<10}")
    print("-" * 60)
    
    for result in results:
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{result['framework']:<15} {result['avg_response_time']:<15.4f} {result['queries_per_second']:<10.2f} {status:<10}")
    
    return results


def visualize_results(results):
    """可视化测试结果"""
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        print("\n没有成功的测试结果，无法生成图表")
        return
    
    frameworks = [r["framework"] for r in successful_results]
    response_times = [r["avg_response_time"] for r in successful_results]
    qps = [r["queries_per_second"] for r in successful_results]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 平均响应时间
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(frameworks, response_times, color=colors[:len(frameworks)])
    ax1.set_ylabel('平均响应时间 (秒)', fontsize=12)
    ax1.set_title('框架响应时间对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, time in zip(bars1, response_times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # QPS对比
    bars2 = ax2.bar(frameworks, qps, color=colors[:len(frameworks)])
    ax2.set_ylabel('每秒查询数 (QPS)', fontsize=12)
    ax2.set_title('框架吞吐量对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, qps_value in zip(bars2, qps):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{qps_value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 图表已保存到 benchmark_results.png")
    plt.show()


def generate_report(results):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("性能分析报告")
    print("=" * 60)
    
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        print("\n没有足够的成功测试结果来生成报告")
        return
    
    # 找出最佳性能
    fastest = min(successful_results, key=lambda x: x["avg_response_time"])
    highest_qps = max(successful_results, key=lambda x: x["queries_per_second"])
    
    print(f"\n🏆 最快响应时间: {fastest['framework']} ({fastest['avg_response_time']:.4f}s)")
    print(f"🏆 最高吞吐量: {highest_qps['framework']} ({highest_qps['queries_per_second']:.2f} QPS)")
    
    print("\n💡 建议:")
    print("  - 如果需要快速响应，选择响应时间最短的框架")
    print("  - 如果需要高并发，选择QPS最高的框架")
    print("  - 考虑功能需求、学习曲线和社区支持")
    print("  - 实际性能可能因硬件、网络和具体用例而异")


if __name__ == "__main__":
    # 检查依赖
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    
    # 运行基准测试
    results = run_benchmarks()
    
    # 生成报告
    generate_report(results)
    
    # 可视化（如果可能）
    try:
        visualize_results(results)
    except:
        print("\n⚠️ 无法生成可视化图表")
    
    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)
