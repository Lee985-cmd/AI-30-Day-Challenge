"""
多租户RAG平台 - 简化版核心实现
包含：数据模型、认证、租户管理、RAG查询
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
from enum import Enum
import jwt
import hashlib
from uuid import uuid4
from pydantic import BaseModel

# ==================== 数据模型 ====================

class Role(str, Enum):
    """用户角色"""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    API_USER = "api_user"


class Tenant(BaseModel):
    """租户模型"""
    id: str
    name: str
    created_at: datetime = datetime.now()
    is_active: bool = True
    
    # 配额
    max_documents: int = 10000
    max_queries_per_day: int = 10000
    max_storage_gb: float = 10.0
    
    # 使用情况
    used_documents: int = 0
    used_queries_today: int = 0
    
    # 隔离策略
    isolation_strategy: str = "row"  # row or schema


class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    email: str
    tenant_id: str
    role: Role = Role.VIEWER
    created_at: datetime = datetime.now()
    is_active: bool = True


class Document(BaseModel):
    """文档模型"""
    id: str
    tenant_id: str
    content: str
    metadata: Dict = {}
    embedding: Optional[List[float]] = None
    created_at: datetime = datetime.now()


# ==================== 认证服务 ====================

class AuthService:
    """认证服务"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(
        self, 
        user_id: str, 
        tenant_id: str, 
        role: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建JWT Token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=1440)
        
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict:
        """验证Token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token已过期")
        except jwt.InvalidTokenError:
            raise Exception("无效的Token")


# ==================== 租户管理器 ====================

class TenantManager:
    """租户管理器"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, User] = {}
        self.documents: Dict[str, Document] = {}
    
    def create_tenant(self, name: str, **kwargs) -> Tenant:
        """创建租户"""
        tenant_id = str(uuid4())[:8]
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            **kwargs
        )
        
        self.tenants[tenant_id] = tenant
        print(f"✅ 租户创建成功: {name} (ID: {tenant_id})")
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """获取租户"""
        return self.tenants.get(tenant_id)
    
    def check_quota(self, tenant_id: str, resource: str) -> bool:
        """检查配额"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        if resource == "documents":
            return tenant.used_documents < tenant.max_documents
        elif resource == "queries":
            return tenant.used_queries_today < tenant.max_queries_per_day
        
        return False
    
    def increment_usage(self, tenant_id: str, resource: str):
        """增加使用量"""
        tenant = self.tenants.get(tenant_id)
        if tenant:
            if resource == "documents":
                tenant.used_documents += 1
            elif resource == "queries":
                tenant.used_queries_today += 1
    
    def get_isolation_strategy(self, tenant_id: str) -> str:
        """获取隔离策略"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return "row"
        
        # 根据文档数量自动选择
        if tenant.used_documents < 1000:
            return "row"
        else:
            return "schema"


# ==================== RAG服务（简化版）====================

class SimpleRAGService:
    """简化的RAG服务（用于演示）"""
    
    def __init__(self):
        self.vector_store: Dict[str, List[Document]] = {}
    
    def add_document(self, tenant_id: str, doc: Document):
        """添加文档到向量存储"""
        if tenant_id not in self.vector_store:
            self.vector_store[tenant_id] = []
        
        # 模拟生成embedding
        doc.embedding = self._generate_mock_embedding(doc.content)
        self.vector_store[tenant_id].append(doc)
        
        print(f"✅ 文档已添加到租户 {tenant_id}: {doc.id}")
    
    def query(self, tenant_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """查询相关文档（简化版 - 基于关键词匹配）"""
        docs = self.vector_store.get(tenant_id, [])
        
        if not docs:
            return []
        
        # 简化的相似度计算（实际应该用向量相似度）
        results = []
        for doc in docs:
            # 简单的关键词匹配
            score = self._calculate_similarity(query, doc.content)
            if score > 0.1:
                results.append({
                    "document": doc.content[:200],
                    "score": score,
                    "metadata": doc.metadata
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """生成模拟的embedding（实际应该调用OpenAI API）"""
        # 使用hash生成固定长度的向量
        hash_value = hashlib.md5(text.encode()).hexdigest()
        embedding = [int(hash_value[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        return embedding
    
    def _calculate_similarity(self, query: str, content: str) -> float:
        """计算相似度（简化版）"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union)


# ==================== 权限管理器 ====================

class PermissionManager:
    """权限管理器"""
    
    ROLE_PERMISSIONS = {
        Role.ADMIN: ["create", "read", "update", "delete", "manage_users"],
        Role.EDITOR: ["create", "read", "update"],
        Role.VIEWER: ["read"],
        Role.API_USER: ["read"],
    }
    
    @classmethod
    def has_permission(cls, role: Role, permission: str) -> bool:
        """检查权限"""
        return permission in cls.ROLE_PERMISSIONS.get(role, [])


# ==================== 示例用法 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("多租户RAG平台 - 核心功能演示")
    print("=" * 80)
    
    # 初始化服务
    auth_service = AuthService(secret_key="test-secret-key")
    tenant_manager = TenantManager()
    rag_service = SimpleRAGService()
    
    # 1. 创建租户
    print("\n1️⃣  创建租户")
    tenant1 = tenant_manager.create_tenant("Company A", max_documents=1000)
    tenant2 = tenant_manager.create_tenant("Company B", max_documents=5000)
    
    # 2. 创建用户
    print("\n2️⃣  创建用户")
    user1 = User(
        id=str(uuid4())[:8],
        username="admin_a",
        email="admin@company-a.com",
        tenant_id=tenant1.id,
        role=Role.ADMIN
    )
    tenant_manager.users[user1.id] = user1
    print(f"✅ 用户创建成功: {user1.username}")
    
    # 3. 生成Token
    print("\n3️⃣  生成JWT Token")
    token = auth_service.create_access_token(
        user_id=user1.id,
        tenant_id=tenant1.id,
        role=user1.role.value
    )
    print(f"✅ Token生成成功: {token[:50]}...")
    
    # 4. 验证Token
    print("\n4️⃣  验证Token")
    payload = auth_service.verify_token(token)
    print(f"✅ Token验证成功: user_id={payload['user_id']}, tenant_id={payload['tenant_id']}")
    
    # 5. 添加文档
    print("\n5️⃣  添加文档")
    if tenant_manager.check_quota(tenant1.id, "documents"):
        doc1 = Document(
            id=str(uuid4())[:8],
            tenant_id=tenant1.id,
            content="Python是一种广泛使用的编程语言，简单易学。",
            metadata={"source": "wiki"}
        )
        rag_service.add_document(tenant1.id, doc1)
        tenant_manager.increment_usage(tenant1.id, "documents")
        
        doc2 = Document(
            id=str(uuid4())[:8],
            tenant_id=tenant1.id,
            content="机器学习是人工智能的一个分支，让计算机从数据中学习。",
            metadata={"source": "article"}
        )
        rag_service.add_document(tenant1.id, doc2)
        tenant_manager.increment_usage(tenant1.id, "documents")
    
    # 6. 查询知识库
    print("\n6️⃣  查询知识库")
    if tenant_manager.check_quota(tenant1.id, "queries"):
        results = rag_service.query(tenant1.id, "Python编程")
        print(f"找到 {len(results)} 个相关文档:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {result['score']:.2f}")
            print(f"     内容: {result['document']}...")
        
        tenant_manager.increment_usage(tenant1.id, "queries")
    
    # 7. 检查权限
    print("\n7️⃣  权限检查")
    print(f"Admin是否有create权限: {PermissionManager.has_permission(Role.ADMIN, 'create')}")
    print(f"Viewer是否有delete权限: {PermissionManager.has_permission(Role.VIEWER, 'delete')}")
    
    # 8. 查看租户统计
    print("\n8️⃣  租户统计")
    tenant = tenant_manager.get_tenant(tenant1.id)
    print(f"租户: {tenant.name}")
    print(f"已用文档: {tenant.used_documents}/{tenant.max_documents}")
    print(f"已用查询: {tenant.used_queries_today}/{tenant.max_queries_per_day}")
    print(f"隔离策略: {tenant_manager.get_isolation_strategy(tenant1.id)}")
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
