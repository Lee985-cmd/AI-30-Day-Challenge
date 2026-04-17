"""
FastAPI 接口
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from data_loader import DataLoader
from pandas_agent import PandasAIAgent
from cleaning_agent import DataCleaningAgent
from visualization_agent import VisualizationAgent
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(__file__))

app = FastAPI(title="AI 数据分析 Agent API")

# 全局变量
pandas_agent = None
cleaning_agent = None
viz_agent = None
current_df = None

@app.on_event("startup")
async def startup():
    """初始化服务"""
    global pandas_agent, cleaning_agent, viz_agent
    
    # 使用本地模型
    base_url = os.getenv("LOCAL_LLM_URL", "")
    api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
    
    print(f"🤖 正在初始化 Agent，使用本地模型: {base_url}")
    
    pandas_agent = PandasAIAgent(api_key=api_key, base_url=base_url)
    cleaning_agent = DataCleaningAgent(api_key=api_key, base_url=base_url)
    viz_agent = VisualizationAgent(api_key=api_key, base_url=base_url)
    
    print("✅ 服务初始化完成！")

class QueryRequest(BaseModel):
    question: str

class CleaningRequest(BaseModel):
    request: str = "自动清洗"

class VizRequest(BaseModel):
    request: str = "自动选择合适的图表"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传数据文件"""
    global current_df
    
    file_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    current_df = DataLoader.load(file_path)
    
    return {
        "message": "文件上传成功",
        "shape": list(current_df.shape),
        "columns": current_df.columns.tolist(),
        "preview": current_df.head().to_dict()
    }

@app.post("/query")
async def query_data(request: QueryRequest):
    """自然语言查询数据"""
    if current_df is None:
        return {"error": "请先上传数据文件"}
    
    result = pandas_agent.query(current_df, request.question)
    return result

@app.post("/clean")
async def clean_data(request: CleaningRequest = CleaningRequest()):
    """清洗数据"""
    global current_df
    
    if current_df is None:
        return {"error": "请先上传数据文件"}
    
    current_df = cleaning_agent.clean(current_df, request.request)
    
    return {
        "message": "数据清洗完成",
        "shape": list(current_df.shape),
        "preview": current_df.head().to_dict()
    }

@app.post("/visualize")
async def visualize_data(request: VizRequest = VizRequest()):
    """生成可视化图表"""
    if current_df is None:
        return {"error": "请先上传数据文件"}
    
    chart_path = viz_agent.visualize(current_df, request.request)
    
    if chart_path:
        return {
            "message": "图表生成成功",
            "chart_path": chart_path
        }
    else:
        return {"error": "图表生成失败"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "AI Data Analysis Agent"}

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用 AI 数据分析 Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
