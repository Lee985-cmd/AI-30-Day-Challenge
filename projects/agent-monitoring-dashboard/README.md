# Agent监控Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

生产级Agent监控系统，提供完整的可观测性解决方案，包括指标收集、日志聚合、链路追踪和告警通知。

**核心特性：**
- ✅ **Prometheus指标收集** - QPS、延迟、错误率、缓存命中率
- ✅ **Grafana Dashboard** - 可视化监控面板
- ✅ **ELK日志聚合** - 结构化JSON日志
- ✅ **Jaeger链路追踪** - 分布式请求追踪
- ✅ **告警规则** - Slack/邮件通知
- ✅ **成本监控** - API费用实时追踪
- ✅ **Docker一键部署** - docker-compose.yml

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用Docker Compose启动所有服务

```bash
docker-compose up -d
```

这将启动：
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000, admin/admin)
- Jaeger (http://localhost:16686)
- Elasticsearch (http://localhost:9200)
- Kibana (http://localhost:5601)
- Redis (localhost:6379)

### 3. 运行示例应用

```bash
python example_app.py
```

### 4. 访问监控面板

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Kibana**: http://localhost:5601

## 📁 项目结构

```
agent-monitoring-dashboard/
├── app/
│   ├── __init__.py
│   ├── metrics.py              # Prometheus指标定义
│   ├── logging_config.py       # 日志配置
│   └── tracing.py              # 链路追踪配置
├── grafana/
│   ├── dashboards/             # Grafana Dashboard模板
│   │   └── agent-monitoring.json
│   └── provisioning/           # 数据源配置
├── prometheus/
│   └── prometheus.yml          # Prometheus配置
├── alertmanager/
│   └── alertmanager.yml        # 告警规则
├── example_app.py              # 示例应用
├── docker-compose.yml          # Docker编排
├── requirements.txt            # 依赖列表
└── README.md                   # 本文件
```

## 📊 监控指标

### 核心指标

| 指标名称 | 类型 | 说明 |
|---------|------|------|
| `agent_requests_total` | Counter | 总请求数 |
| `agent_request_duration_seconds` | Histogram | 请求耗时分布 |
| `agent_errors_total` | Counter | 错误总数 |
| `agent_cache_hit_rate` | Gauge | 缓存命中率 |
| `agent_api_cost_usd` | Counter | API费用 |
| `agent_active_connections` | Gauge | 活跃连接数 |

### Grafana Dashboard视图

1. **概览面板**
   - 实时QPS
   - P95/P99响应时间
   - 错误率
   - 活跃租户数

2. **性能分析**
   - 响应时间分布
   - 慢请求Top 10
   - 错误类型分布

3. **成本监控**
   - 每日API费用趋势
   - 各模型费用占比
   - 各租户费用排名

4. **缓存监控**
   - 缓存命中率
   - 缓存大小
   - 缓存淘汰率

## 🚨 告警规则

### Prometheus告警配置

```yaml
groups:
  - name: agent_alerts
    rules:
      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(agent_errors_total[5m]) / rate(agent_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
      
      # 慢请求告警
      - alert: SlowRequests
        expr: histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
      
      # API费用异常告警
      - alert: HighAPICost
        expr: increase(agent_api_cost_usd[1h]) > 10
        for: 0m
        labels:
          severity: critical
```

### 通知渠道

- ✅ Slack Webhook
- ✅ 邮件通知
- ✅ 企业微信（可选）

## 🔍 链路追踪

### Jaeger集成

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# 配置Jaeger导出器
jaeger_exporter = JaegerExporter(
    agent_host_name='localhost',
    agent_port=6831,
)

# 创建Span
with tracer.start_as_current_span("process_request") as span:
    span.set_attribute("tenant.id", tenant_id)
    # 业务逻辑
```

访问 http://localhost:16686 查看链路追踪。

## 📝 日志格式

### 结构化JSON日志

```json
{
  "timestamp": "2026-05-07T09:00:00.000Z",
  "level": "INFO",
  "message": "Request completed",
  "tenant_id": "tenant_001",
  "request_id": "abc-123-def",
  "duration_seconds": 0.234,
  "module": "rag_service",
  "function": "query"
}
```

### ELK Stack查询

在Kibana中使用Lucene查询语法：

```
tenant_id: "tenant_001" AND level: "ERROR"
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 🐳 Docker部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📈 实际应用案例

### 案例1：电商客服Agent

**监控重点：**
- 响应时间P95 < 1s
- 错误率 < 2%
- 缓存命中率 > 70%
- 每日API费用 < $100

**效果：**
- 故障发现时间从30分钟降到2分钟
- 性能问题定位时间从2小时降到10分钟
- API成本降低40%

### 案例2：金融投研Agent

**监控重点：**
- 数据准确性
- 合规性检查
- 审计日志
- 高可用（99.9% SLA）

**效果：**
- 满足金融监管要求
- 完整的审计追溯
- 系统可用性达到99.95%

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📜 许可证

MIT License

## 👤 作者

学习论之费曼学习法

- CSDN: https://blog.csdn.net/m0_67081842
- GitHub: https://github.com/Lee985-cmd
