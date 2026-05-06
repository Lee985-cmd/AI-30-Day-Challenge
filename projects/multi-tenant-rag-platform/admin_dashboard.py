"""
多租户RAG平台 - Streamlit管理后台
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core_demo import TenantManager, AuthService, SimpleRAGService, Role
from uuid import uuid4

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="多租户RAG管理平台",
    page_icon="🏢",
    layout="wide"
)

# ==================== 初始化Session State ====================

if 'tenant_manager' not in st.session_state:
    st.session_state.tenant_manager = TenantManager()

if 'auth_service' not in st.session_state:
    st.session_state.auth_service = AuthService(secret_key="test-secret-key")

if 'rag_service' not in st.session_state:
    st.session_state.rag_service = SimpleRAGService()

# ==================== 侧边栏 ====================

st.sidebar.title("🏢 多租户RAG管理平台")
page = st.sidebar.radio(
    "导航",
    ["📊 概览", "🏢 租户管理", "👥 用户管理", "📄 文档管理", "🔍 RAG查询", "⚙️ 系统设置"]
)

# ==================== 概览页面 ====================

if page == "📊 概览":
    st.title("📊 系统概览")
    
    # 统计信息
    tenant_manager = st.session_state.tenant_manager
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("活跃租户数", len(tenant_manager.tenants))
    
    with col2:
        total_users = len(tenant_manager.users)
        st.metric("总用户数", total_users)
    
    with col3:
        total_docs = sum(t.used_documents for t in tenant_manager.tenants.values())
        st.metric("总文档数", total_docs)
    
    with col4:
        total_queries = sum(t.used_queries_today for t in tenant_manager.tenants.values())
        st.metric("今日查询数", total_queries)
    
    st.divider()
    
    # 租户列表
    st.subheader("租户列表")
    
    if tenant_manager.tenants:
        for tenant_id, tenant in tenant_manager.tenants.items():
            with st.expander(f"🏢 {tenant.name} (ID: {tenant_id})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**状态:** {'✅ 活跃' if tenant.is_active else '❌ 停用'}")
                    st.write(f"**隔离策略:** {tenant.isolation_strategy}")
                    st.write(f"**创建时间:** {tenant.created_at.strftime('%Y-%m-%d')}")
                
                with col2:
                    st.write(f"**文档使用:** {tenant.used_documents}/{tenant.max_documents}")
                    st.progress(tenant.used_documents / tenant.max_documents)
                    
                    st.write(f"**查询使用:** {tenant.used_queries_today}/{tenant.max_queries_per_day}")
                    st.progress(tenant.used_queries_today / tenant.max_queries_per_day)
    else:
        st.info("暂无租户，请在'租户管理'页面创建")

# ==================== 租户管理页面 ====================

elif page == "🏢 租户管理":
    st.title("🏢 租户管理")
    
    tenant_manager = st.session_state.tenant_manager
    
    # 创建新租户
    with st.expander("➕ 创建新租户", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            tenant_name = st.text_input("租户名称")
            max_documents = st.number_input("最大文档数", min_value=100, value=10000)
        
        with col2:
            max_queries = st.number_input("每日最大查询数", min_value=100, value=10000)
            max_storage = st.number_input("最大存储空间(GB)", min_value=1.0, value=10.0)
        
        if st.button("创建租户", type="primary"):
            if tenant_name:
                tenant = tenant_manager.create_tenant(
                    name=tenant_name,
                    max_documents=max_documents,
                    max_queries_per_day=max_queries,
                    max_storage_gb=max_storage
                )
                st.success(f"✅ 租户 '{tenant_name}' 创建成功！ID: {tenant.id}")
                st.rerun()
            else:
                st.error("❌ 请输入租户名称")
    
    st.divider()
    
    # 租户列表
    st.subheader("现有租户")
    
    if tenant_manager.tenants:
        for tenant_id, tenant in tenant_manager.tenants.items():
            cols = st.columns([3, 2, 2, 1])
            
            with cols[0]:
                st.write(f"**{tenant.name}**")
                st.caption(f"ID: {tenant_id}")
            
            with cols[1]:
                st.write(f"文档: {tenant.used_documents}/{tenant.max_documents}")
            
            with cols[2]:
                st.write(f"查询: {tenant.used_queries_today}/{tenant.max_queries_per_day}")
            
            with cols[3]:
                if st.button("删除", key=f"del_{tenant_id}"):
                    del tenant_manager.tenants[tenant_id]
                    st.success(f"租户 '{tenant.name}' 已删除")
                    st.rerun()
    else:
        st.info("暂无租户")

# ==================== 用户管理页面 ====================

elif page == "👥 用户管理":
    st.title("👥 用户管理")
    
    tenant_manager = st.session_state.tenant_manager
    auth_service = st.session_state.auth_service
    
    # 创建新用户
    with st.expander("➕ 创建新用户", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("用户名")
            email = st.text_input("邮箱")
        
        with col2:
            tenant_options = [f"{t.name} ({t.id})" for t in tenant_manager.tenants.values()]
            selected_tenant = st.selectbox("选择租户", tenant_options) if tenant_options else None
            
            role = st.selectbox("角色", [r.value for r in Role])
        
        if st.button("创建用户", type="primary"):
            if username and email and selected_tenant:
                tenant_id = selected_tenant.split("(")[-1].strip(")")
                
                user_id = str(uuid4())[:8]
                from core_demo import User
                
                user = User(
                    id=user_id,
                    username=username,
                    email=email,
                    tenant_id=tenant_id,
                    role=Role(role)
                )
                
                tenant_manager.users[user_id] = user
                
                # 生成Token
                token = auth_service.create_access_token(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    role=role
                )
                
                st.success(f"✅ 用户 '{username}' 创建成功！")
                st.code(token, language="text")
                st.caption("↑ 这是用户的JWT Token，请妥善保管")
            else:
                st.error("❌ 请填写所有必填字段")
    
    st.divider()
    
    # 用户列表
    st.subheader("现有用户")
    
    if tenant_manager.users:
        for user_id, user in tenant_manager.users.items():
            cols = st.columns([2, 2, 2, 2, 1])
            
            with cols[0]:
                st.write(f"**{user.username}**")
                st.caption(user.email)
            
            with cols[1]:
                tenant = tenant_manager.get_tenant(user.tenant_id)
                tenant_name = tenant.name if tenant else "Unknown"
                st.write(f"租户: {tenant_name}")
            
            with cols[2]:
                st.write(f"角色: {user.role.value}")
            
            with cols[3]:
                status = "✅ 活跃" if user.is_active else "❌ 停用"
                st.write(status)
            
            with cols[4]:
                if st.button("删除", key=f"del_user_{user_id}"):
                    del tenant_manager.users[user_id]
                    st.success(f"用户 '{user.username}' 已删除")
                    st.rerun()
    else:
        st.info("暂无用户")

# ==================== 文档管理页面 ====================

elif page == "📄 文档管理":
    st.title("📄 文档管理")
    
    tenant_manager = st.session_state.tenant_manager
    rag_service = st.session_state.rag_service
    
    # 上传文档
    with st.expander("📤 上传文档", expanded=True):
        tenant_options = [f"{t.name} ({t.id})" for t in tenant_manager.tenants.values()]
        selected_tenant = st.selectbox("选择租户", tenant_options) if tenant_options else None
        
        content = st.text_area("文档内容", height=200)
        metadata_str = st.text_input("元数据(JSON格式)", value='{"source": "manual"}')
        
        if st.button("上传文档", type="primary"):
            if selected_tenant and content:
                tenant_id = selected_tenant.split("(")[-1].strip(")")
                
                # 检查配额
                if not tenant_manager.check_quota(tenant_id, "documents"):
                    st.error("❌ 租户文档配额已满")
                else:
                    from core_demo import Document
                    
                    doc = Document(
                        id=str(uuid4())[:8],
                        tenant_id=tenant_id,
                        content=content,
                        metadata={"source": "manual"}
                    )
                    
                    rag_service.add_document(tenant_id, doc)
                    tenant_manager.increment_usage(tenant_id, "documents")
                    
                    st.success("✅ 文档上传成功！")
                    st.rerun()
            else:
                st.error("❌ 请选择租户并填写文档内容")
    
    st.divider()
    
    # 文档列表
    st.subheader("文档列表")
    
    if rag_service.vector_store:
        for tenant_id, docs in rag_service.vector_store.items():
            tenant = tenant_manager.get_tenant(tenant_id)
            tenant_name = tenant.name if tenant else tenant_id
            
            with st.expander(f"🏢 {tenant_name} ({len(docs)} 个文档)"):
                for doc in docs:
                    st.write(f"**{doc.id}**")
                    st.caption(doc.content[:100] + "...")
                    st.json(doc.metadata)
                    st.divider()
    else:
        st.info("暂无文档")

# ==================== RAG查询页面 ====================

elif page == "🔍 RAG查询":
    st.title("🔍 RAG知识库查询")
    
    tenant_manager = st.session_state.tenant_manager
    rag_service = st.session_state.rag_service
    
    tenant_options = [f"{t.name} ({t.id})" for t in tenant_manager.tenants.values()]
    selected_tenant = st.selectbox("选择租户", tenant_options) if tenant_options else None
    
    query = st.text_input("输入查询问题")
    top_k = st.slider("返回结果数量", min_value=1, max_value=10, value=3)
    
    if st.button("查询", type="primary"):
        if selected_tenant and query:
            tenant_id = selected_tenant.split("(")[-1].strip(")")
            
            # 检查配额
            if not tenant_manager.check_quota(tenant_id, "queries"):
                st.error("❌ 租户查询配额已满")
            else:
                results = rag_service.query(tenant_id, query, top_k=top_k)
                
                if results:
                    st.success(f"找到 {len(results)} 个相关文档")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.write(f"**结果 {i}** (相似度: {result['score']:.2f})")
                            st.write(result['document'])
                            st.json(result['metadata'])
                            st.divider()
                    
                    tenant_manager.increment_usage(tenant_id, "queries")
                else:
                    st.warning("未找到相关文档")
        else:
            st.error("❌ 请选择租户并输入查询问题")

# ==================== 系统设置页面 ====================

elif page == "⚙️ 系统设置":
    st.title("⚙️ 系统设置")
    
    st.info("系统设置功能开发中...")
    
    st.subheader("当前配置")
    st.json({
        "isolation_threshold": 1000,
        "default_max_documents": 10000,
        "default_max_queries": 10000,
        "jwt_algorithm": "HS256"
    })

# ==================== 页脚 ====================

st.sidebar.divider()
st.sidebar.caption("© 2026 Multi-Tenant RAG Platform")
st.sidebar.caption("Version 1.0.0")
