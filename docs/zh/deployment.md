# 部署指南

本指南涵盖在生产环境中部署 genrec 模型的方法。

## 生产环境部署

### 使用 FastAPI 进行模型服务

创建 REST API 服务器进行模型推理：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from genrec.models.tiger import Tiger
from genrec.models.rqvae import RqVae

app = FastAPI(title="genrec API", version="1.0.0")

class RecommendationRequest(BaseModel):
    user_id: int
    user_history: List[int]
    num_recommendations: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    scores: Optional[List[float]] = None

class ModelService:
    def __init__(self, rqvae_path: str, tiger_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.rqvae = RqVae.load_from_checkpoint(rqvae_path)
        self.rqvae.to(self.device)
        self.rqvae.eval()
        
        self.tiger = Tiger.load_from_checkpoint(tiger_path)
        self.tiger.to(self.device)
        self.tiger.eval()
    
    def get_recommendations(self, user_history: List[int], k: int) -> List[int]:
        """为用户生成推荐"""
        with torch.no_grad():
            # 将物品ID转换为语义ID
            semantic_sequence = self.items_to_semantic_ids(user_history)
            
            # 生成推荐
            input_seq = torch.tensor(semantic_sequence).unsqueeze(0).to(self.device)
            generated = self.tiger.generate(input_seq, max_length=k*3)  # 生成更多以处理重复
            
            # 转换回物品ID并去重
            recommendations = self.semantic_ids_to_items(generated.squeeze().tolist())
            
            # 移除用户历史中已有的物品
            recommendations = [item for item in recommendations if item not in user_history]
            
            return recommendations[:k]

# 初始化模型服务
model_service = ModelService(
    rqvae_path="checkpoints/rqvae.ckpt",
    tiger_path="checkpoints/tiger.ckpt"
)

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """为用户生成推荐"""
    try:
        recommendations = model_service.get_recommendations(
            request.user_history,
            request.num_recommendations
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker 部署

创建 Dockerfile：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行容器：

```bash
# 构建镜像
docker build -t generative-recommenders:latest .

# 运行容器
docker run -d -p 8000:8000 \
    -v /path/to/checkpoints:/app/checkpoints \
    generative-recommenders:latest
```

### Kubernetes 部署

创建 Kubernetes 配置：

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: generative-recommenders
spec:
  replicas: 3
  selector:
    matchLabels:
      app: generative-recommenders
  template:
    metadata:
      labels:
        app: generative-recommenders
    spec:
      containers:
      - name: api
        image: generative-recommenders:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: generative-recommenders-service
spec:
  selector:
    app: generative-recommenders
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

部署到 Kubernetes：

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## 批处理

### Apache Spark 集成

使用 Spark 处理大型数据集：

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, IntegerType
import torch

def create_spark_session():
    return SparkSession.builder \
        .appName("genrec") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def broadcast_model(spark, model_path):
    """将模型广播到所有工作节点"""
    model = Tiger.load_from_checkpoint(model_path)
    model.eval()
    return spark.sparkContext.broadcast(model)

def batch_recommend_udf(broadcast_model):
    """批量推荐的 UDF"""
    @udf(returnType=ArrayType(IntegerType()))
    def recommend(user_history):
        model = broadcast_model.value
        with torch.no_grad():
            # 转换为张量
            input_seq = torch.tensor(user_history).unsqueeze(0)
            
            # 生成推荐
            recommendations = model.generate(input_seq, max_length=20)
            
            return recommendations.squeeze().tolist()
    
    return recommend

# 主处理流程
spark = create_spark_session()
model_broadcast = broadcast_model(spark, "checkpoints/tiger.ckpt")

# 加载用户数据
user_data = spark.read.parquet("s3://data/user_interactions")

# 生成推荐
recommend_func = batch_recommend_udf(model_broadcast)
recommendations = user_data.withColumn(
    "recommendations", 
    recommend_func(col("interaction_history"))
)

# 保存结果
recommendations.write.mode("overwrite").parquet("s3://output/recommendations")
```

### Apache Airflow 流水线

创建推荐流水线：

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'genrec_pipeline',
    default_args=default_args,
    description='每日推荐生成',
    schedule_interval='@daily',
    catchup=False
)

def extract_user_data(**context):
    """提取用户交互数据"""
    # 实现逻辑
    pass

def generate_recommendations(**context):
    """使用 TIGER 模型生成推荐"""
    # 实现逻辑
    pass

def upload_recommendations(**context):
    """将推荐上传到推荐服务"""
    # 实现逻辑
    pass

# 定义任务
extract_task = PythonOperator(
    task_id='extract_user_data',
    python_callable=extract_user_data,
    dag=dag
)

recommend_task = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    dag=dag
)

upload_task = PythonOperator(
    task_id='upload_recommendations',
    python_callable=upload_recommendations,
    dag=dag
)

# 设置依赖关系
extract_task >> recommend_task >> upload_task
```

## 监控和可观测性

### Prometheus 指标

为 FastAPI 应用添加指标：

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# 指标
REQUEST_COUNT = Counter('recommendations_requests_total', '推荐请求总数')
REQUEST_LATENCY = Histogram('recommendations_request_duration_seconds', '请求延迟')
ERROR_COUNT = Counter('recommendations_errors_total', '错误总数')

@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        ERROR_COUNT.inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return Response(generate_latest(), media_type="text/plain")
```

### 日志配置

设置结构化日志：

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format='%(message)s'
)

logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(JSONFormatter())
```

## 性能优化

### 模型量化

减少模型大小和推理时间：

```python
import torch.quantization as quantization

def quantize_model(model, example_inputs):
    """量化模型以加快推理速度"""
    # 准备模型进行量化
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model_prepared = quantization.prepare(model, inplace=False)
    
    # 使用示例输入进行校准
    model_prepared.eval()
    with torch.no_grad():
        model_prepared(example_inputs)
    
    # 转换为量化模型
    model_quantized = quantization.convert(model_prepared, inplace=False)
    
    return model_quantized

# 使用示例
example_input = torch.randint(0, 1000, (1, 50))
quantized_tiger = quantize_model(tiger_model, example_input)
```

### ONNX 导出

将模型导出为 ONNX 格式以实现跨平台部署：

```python
def export_to_onnx(model, example_input, output_path):
    """将 PyTorch 模型导出为 ONNX"""
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )

# 导出模型
export_to_onnx(tiger_model, example_input, "models/tiger.onnx")
```

### TensorRT 优化

为 NVIDIA GPU 优化模型：

```python
import tensorrt as trt

def convert_onnx_to_tensorrt(onnx_path, engine_path, max_batch_size=32):
    """将 ONNX 模型转换为 TensorRT 引擎"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX 模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 构建引擎
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (max_batch_size, 512), (max_batch_size, 512))
    config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

## A/B 测试框架

### 实验配置

设置 A/B 测试进行模型比较：

```python
import hashlib
import random
from typing import Dict, Any

class ABTestFramework:
    def __init__(self, experiments: Dict[str, Any]):
        self.experiments = experiments
    
    def get_variant(self, user_id: int, experiment_name: str) -> str:
        """获取用户在实验中的变体"""
        if experiment_name not in self.experiments:
            return "control"
        
        experiment = self.experiments[experiment_name]
        
        # 使用一致性哈希进行用户分配
        hash_input = f"{user_id}_{experiment_name}_{experiment['salt']}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        cumulative_traffic = 0
        for variant, traffic in experiment['variants'].items():
            cumulative_traffic += traffic
            if bucket < cumulative_traffic:
                return variant
        
        return "control"
    
    def is_user_in_experiment(self, user_id: int, experiment_name: str) -> bool:
        """检查用户是否在实验中"""
        if experiment_name not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_name]
        if not experiment.get('active', False):
            return False
        
        # 检查资格条件
        if 'eligibility' in experiment:
            # 实现资格逻辑
            pass
        
        return True

# 实验配置示例
experiments_config = {
    "model_comparison": {
        "active": True,
        "salt": "experiment_salt_123",
        "variants": {
            "control": 50,  # 50% 流量
            "new_model": 50  # 50% 流量
        },
        "eligibility": {
            "min_interactions": 10
        }
    }
}

ab_tester = ABTestFramework(experiments_config)

@app.post("/recommend")
async def recommend_with_ab_test(request: RecommendationRequest):
    """使用 A/B 测试生成推荐"""
    variant = ab_tester.get_variant(request.user_id, "model_comparison")
    
    if variant == "new_model":
        # 使用新模型
        recommendations = new_model_service.get_recommendations(
            request.user_history, request.num_recommendations
        )
    else:
        # 使用对照模型
        recommendations = model_service.get_recommendations(
            request.user_history, request.num_recommendations
        )
    
    # 记录实验数据
    logger.info("推荐已提供", extra={
        "user_id": request.user_id,
        "variant": variant,
        "experiment": "model_comparison"
    })
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations
    )
```

## 安全考虑

### API 认证

添加 JWT 认证：

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 JWT 令牌"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据"
        )

@app.post("/recommend")
async def recommend(
    request: RecommendationRequest,
    token_data: dict = Depends(verify_token)
):
    """受保护的推荐端点"""
    # 验证用户访问权限
    if token_data.get("user_id") != request.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="访问被拒绝"
        )
    
    return await model_service.get_recommendations(request)
```

### 输入验证

验证和清理输入：

```python
from pydantic import validator

class RecommendationRequest(BaseModel):
    user_id: int
    user_history: List[int]
    num_recommendations: int = 10
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('用户ID必须为正数')
        return v
    
    @validator('user_history')
    def validate_user_history(cls, v):
        if len(v) > 1000:  # 限制历史长度
            raise ValueError('用户历史过长')
        if any(item <= 0 for item in v):
            raise ValueError('历史中包含无效的物品ID')
        return v
    
    @validator('num_recommendations')
    def validate_num_recommendations(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('推荐数量必须在1到100之间')
        return v
```

本部署指南涵盖了在生产环境中部署 genrec 模型的重要方面，从基本的 API 服务到高级的优化和监控技术。