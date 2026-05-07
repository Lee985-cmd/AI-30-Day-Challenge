"""
Agent监控系统 - 示例应用
演示：Prometheus指标、结构化日志、链路追踪
"""

import time
import random
import logging
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ==================== 配置日志 ====================

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s"}'
)
logger = logging.getLogger(__name__)

# ==================== Prometheus指标 ====================

# 请求计数器
REQUEST_COUNT = Counter(
    'agent_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

# 请求耗时
REQUEST_DURATION = Histogram(
    'agent_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# 错误计数器
ERROR_COUNT = Counter(
    'agent_errors_total',
    'Total number of errors',
    ['error_type']
)

# 缓存命中率
CACHE_HIT_RATE = Gauge(
    'agent_cache_hit_rate',
    'Cache hit rate percentage',
    ['cache_type']
)

# API费用
API_COST = Counter(
    'agent_api_cost_usd',
    'API cost in USD',
    ['model']
)

# 活跃连接数
ACTIVE_CONNECTIONS = Gauge(
    'agent_active_connections',
    'Number of active connections'
)

# ==================== OpenTelemetry链路追踪 ====================

# 配置TracerProvider
provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# ==================== FastAPI应用 ====================

app = FastAPI(title="Agent Monitoring Demo")

# 自动instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# 模拟缓存
mock_cache = {}

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    # 启动Prometheus指标服务器
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Agent Monitoring Dashboard", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/api/query")
async def query_knowledge_base(query: str):
    """
    查询知识库（带监控）
    
    演示：
    1. Prometheus指标收集
    2. 结构化日志
    3. 链路追踪
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("query_knowledge_base") as span:
        span.set_attribute("query.length", len(query))
        
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # 模拟缓存检查
            with tracer.start_as_current_span("check_cache") as cache_span:
                if query in mock_cache:
                    cache_span.set_attribute("cache.hit", True)
                    CACHE_HIT_RATE.labels(cache_type="memory").set(100.0)
                    logger.info("Cache hit")
                    
                    REQUEST_COUNT.labels(
                        method='POST',
                        endpoint='/api/query',
                        status='success'
                    ).inc()
                    
                    return {
                        "result": mock_cache[query],
                        "source": "cache",
                        "cached": True
                    }
                else:
                    cache_span.set_attribute("cache.hit", False)
                    CACHE_HIT_RATE.labels(cache_type="memory").set(0.0)
            
            # 模拟LLM调用
            with tracer.start_as_current_span("call_llm") as llm_span:
                logger.info("Calling LLM...")
                
                # 模拟延迟
                time.sleep(random.uniform(0.5, 2.0))
                
                # 模拟结果
                result = f"这是关于'{query}'的回答（模拟）"
                
                # 模拟token和成本
                tokens = random.randint(100, 500)
                cost = tokens * 0.000002  # $0.002 per 1K tokens
                
                llm_span.set_attribute("llm.tokens", tokens)
                llm_span.set_attribute("llm.cost_usd", cost)
                
                API_COST.labels(model='gpt-3.5-turbo').inc(cost)
                
                # 存入缓存
                mock_cache[query] = result
            
            # 记录成功
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method='POST',
                endpoint='/api/query',
                status='success'
            ).inc()
            
            REQUEST_DURATION.labels(
                method='POST',
                endpoint='/api/query'
            ).observe(duration)
            
            logger.info(f"Query completed in {duration:.2f}s")
            
            return {
                "result": result,
                "source": "llm",
                "cached": False,
                "tokens": tokens,
                "cost_usd": round(cost, 6),
                "duration_seconds": round(duration, 3)
            }
        
        except Exception as e:
            # 记录错误
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """
    Prometheus指标端点
    
    访问 http://localhost:8000/metrics 查看原始指标
    """
    from prometheus_client import generate_latest
    from starlette.responses import Response
    
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@app.get("/stats")
async def stats():
    """
    系统统计信息
    """
    return {
        "total_requests": REQUEST_COUNT._metrics,
        "cache_size": len(mock_cache),
        "active_connections": ACTIVE_CONNECTIONS._value.get(),
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("Agent监控系统 - 示例应用")
    print("=" * 80)
    print()
    print("📊 Prometheus指标: http://localhost:8000/metrics")
    print("🔍 API文档: http://localhost:8000/docs")
    print()
    print("启动服务...")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
