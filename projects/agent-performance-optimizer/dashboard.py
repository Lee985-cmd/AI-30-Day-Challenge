"""
性能监控Dashboard
使用Streamlit实现实时可视化监控
"""

import streamlit as st
import time
import json
from datetime import datetime
from cache import MemoryCache, RedisCache


# 页面配置
st.set_page_config(
    page_title="Agent性能监控",
    page_icon="📊",
    layout="wide"
)

# 标题
st.title("📊 Agent性能监控 Dashboard")
st.markdown("---")

# 侧边栏
st.sidebar.header("⚙️ 配置")
cache_type = st.sidebar.selectbox(
    "缓存类型",
    ["内存缓存", "Redis缓存"]
)

if cache_type == "内存缓存":
    ttl = st.sidebar.slider("TTL (秒)", 60, 7200, 3600)
    max_size = st.sidebar.number_input("最大缓存数", 1000, 100000, 10000)
    cache = MemoryCache(ttl=ttl, max_size=max_size)
else:
    redis_host = st.sidebar.text_input("Redis主机", "localhost")
    redis_port = st.sidebar.number_input("Redis端口", 6379, 6380, 6379)
    try:
        cache = RedisCache(host=redis_host, port=redis_port)
    except:
        st.sidebar.error("Redis连接失败")
        cache = None

# 初始化session state
if 'requests' not in st.session_state:
    st.session_state.requests = []
if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0
if 'cache_misses' not in st.session_state:
    st.session_state.cache_misses = 0

# 主要指标卡片
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_requests = len(st.session_state.requests)
    st.metric("总请求数", total_requests, delta=None)

with col2:
    hit_rate = (
        st.session_state.cache_hits / total_requests * 100 
        if total_requests > 0 else 0
    )
    st.metric("缓存命中率", f"{hit_rate:.1f}%", delta=None)

with col3:
    avg_time = (
        sum(r['time'] for r in st.session_state.requests) / total_requests * 1000
        if total_requests > 0 else 0
    )
    st.metric("平均响应时间", f"{avg_time:.0f}ms", delta=None)

with col4:
    cache_stats = cache.stats() if cache else {}
    st.metric("缓存大小", cache_stats.get('size', 0), delta=None)

st.markdown("---")

# 两列布局
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 响应时间趋势")
    
    if st.session_state.requests:
        # 提取数据
        times = [r['time'] * 1000 for r in st.session_state.requests[-50:]]
        timestamps = [r['timestamp'] for r in st.session_state.requests[-50:]]
        
        # 绘制图表
        import pandas as pd
        df = pd.DataFrame({
            '时间': timestamps,
            '响应时间(ms)': times
        })
        
        st.line_chart(df.set_index('时间'))
    else:
        st.info("暂无数据，请先发送测试请求")

with col2:
    st.subheader("📊 缓存统计")
    
    if cache:
        stats = cache.stats()
        
        # 显示统计信息
        st.json(stats)
        
        # 缓存命中率饼图
        if total_requests > 0:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Pie(
                labels=['命中', '未命中'],
                values=[st.session_state.cache_hits, st.session_state.cache_misses],
                hole=.3
            )])
            
            fig.update_layout(title="缓存命中率")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 测试区域
st.subheader("🧪 性能测试")

test_query = st.text_input("测试查询", "什么是AI？")
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("发送测试请求", type="primary"):
        start_time = time.time()
        
        # 模拟缓存检查
        cached = cache.get(test_query) if cache else None
        
        if cached:
            response = cached
            st.session_state.cache_hits += 1
            st.success("✅ 缓存命中")
        else:
            # 模拟LLM调用（实际应该调用API）
            time.sleep(0.5)  # 模拟延迟
            response = f"这是'{test_query}'的回答..."
            if cache:
                cache.set(test_query, response)
            st.session_state.cache_misses += 1
            st.info("❌ 缓存未命中，调用LLM")
        
        elapsed = time.time() - start_time
        
        # 记录请求
        st.session_state.requests.append({
            'query': test_query,
            'time': elapsed,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'cached': cached is not None
        })
        
        st.write(f"**响应时间:** {elapsed*1000:.0f}ms")
        st.write(f"**回答:** {response}")

with col2:
    if st.button("批量测试 (10个请求)"):
        progress_bar = st.progress(0)
        
        test_queries = [
            "什么是机器学习？",
            "Python有哪些优点？",
            "如何学习编程？",
            "AI的未来发展？",
            "推荐一本好书",
            "什么是深度学习？",
            "数据分析流程？",
            "云计算优势？",
            "区块链应用？",
            "网络安全要点？"
        ]
        
        results = []
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            cached = cache.get(query) if cache else None
            
            if cached:
                st.session_state.cache_hits += 1
            else:
                time.sleep(0.3)  # 模拟延迟
                if cache:
                    cache.set(query, f"回答: {query}")
                st.session_state.cache_misses += 1
            
            elapsed = time.time() - start_time
            
            st.session_state.requests.append({
                'query': query,
                'time': elapsed,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'cached': cached is not None
            })
            
            progress_bar.progress((i + 1) / len(test_queries))
        
        progress_bar.empty()
        st.success(f"✅ 完成{len(test_queries)}个测试请求")

st.markdown("---")

# 最近请求列表
st.subheader("📋 最近请求")

if st.session_state.requests:
    # 显示最近10个请求
    recent = st.session_state.requests[-10:]
    
    for req in reversed(recent):
        status = "✅ 缓存" if req['cached'] else "❌ LLM"
        st.caption(
            f"{req['timestamp']} | {status} | "
            f"{req['time']*1000:.0f}ms | {req['query'][:50]}"
        )
else:
    st.info("暂无请求记录")

# 底部按钮
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("清空数据"):
        st.session_state.requests = []
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        if cache:
            cache.clear()
        st.rerun()

with col2:
    if st.button("导出结果"):
        import json
        data = {
            'requests': st.session_state.requests,
            'stats': {
                'total': len(st.session_state.requests),
                'hits': st.session_state.cache_hits,
                'misses': st.session_state.cache_misses
            }
        }
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label="下载JSON",
            data=json_str,
            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col3:
    if cache and st.button("清理过期缓存"):
        if hasattr(cache, 'cleanup_expired'):
            cleaned = cache.cleanup_expired()
            st.success(f"✅ 清理了{cleaned}个过期条目")
        else:
            st.info("当前缓存不支持自动清理")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>
        Agent性能优化工具包 | 
        <a href='https://github.com/Lee985-cmd/AI-30-Day-Challenge'>GitHub</a>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
