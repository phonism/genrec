# Deployment Guide

This guide covers deploying genrec models in production environments.

## Production Deployment

### Model Serving with FastAPI

Create a REST API server for model inference:

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
        
        # Load models
        self.rqvae = RqVae.load_from_checkpoint(rqvae_path)
        self.rqvae.to(self.device)
        self.rqvae.eval()
        
        self.tiger = Tiger.load_from_checkpoint(tiger_path)
        self.tiger.to(self.device)
        self.tiger.eval()
    
    def get_recommendations(self, user_history: List[int], k: int) -> List[int]:
        """Generate recommendations for user"""
        with torch.no_grad():
            # Convert item IDs to semantic IDs
            semantic_sequence = self.items_to_semantic_ids(user_history)
            
            # Generate recommendations
            input_seq = torch.tensor(semantic_sequence).unsqueeze(0).to(self.device)
            generated = self.tiger.generate(input_seq, max_length=k*3)  # Generate more to account for duplicates
            
            # Convert back to item IDs and deduplicate
            recommendations = self.semantic_ids_to_items(generated.squeeze().tolist())
            
            # Remove items already in user history
            recommendations = [item for item in recommendations if item not in user_history]
            
            return recommendations[:k]

# Initialize model service
model_service = ModelService(
    rqvae_path="checkpoints/rqvae.ckpt",
    tiger_path="checkpoints/tiger.ckpt"
)

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Generate recommendations for a user"""
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
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run the container:

```bash
# Build the image
docker build -t generative-recommenders:latest .

# Run the container
docker run -d -p 8000:8000 \
    -v /path/to/checkpoints:/app/checkpoints \
    generative-recommenders:latest
```

### Kubernetes Deployment

Create Kubernetes manifests:

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

Deploy to Kubernetes:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Batch Processing

### Apache Spark Integration

Process large datasets with Spark:

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
    """Broadcast model to all workers"""
    model = Tiger.load_from_checkpoint(model_path)
    model.eval()
    return spark.sparkContext.broadcast(model)

def batch_recommend_udf(broadcast_model):
    """UDF for batch recommendations"""
    @udf(returnType=ArrayType(IntegerType()))
    def recommend(user_history):
        model = broadcast_model.value
        with torch.no_grad():
            # Convert to tensor
            input_seq = torch.tensor(user_history).unsqueeze(0)
            
            # Generate recommendations
            recommendations = model.generate(input_seq, max_length=20)
            
            return recommendations.squeeze().tolist()
    
    return recommend

# Main processing
spark = create_spark_session()
model_broadcast = broadcast_model(spark, "checkpoints/tiger.ckpt")

# Load user data
user_data = spark.read.parquet("s3://data/user_interactions")

# Generate recommendations
recommend_func = batch_recommend_udf(model_broadcast)
recommendations = user_data.withColumn(
    "recommendations", 
    recommend_func(col("interaction_history"))
)

# Save results
recommendations.write.mode("overwrite").parquet("s3://output/recommendations")
```

### Apache Airflow Pipeline

Create a recommendation pipeline:

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
    description='Daily recommendation generation',
    schedule_interval='@daily',
    catchup=False
)

def extract_user_data(**context):
    """Extract user interaction data"""
    # Implementation here
    pass

def generate_recommendations(**context):
    """Generate recommendations using TIGER model"""
    # Implementation here
    pass

def upload_recommendations(**context):
    """Upload recommendations to recommendation service"""
    # Implementation here
    pass

# Define tasks
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

# Set dependencies
extract_task >> recommend_task >> upload_task
```

## Monitoring and Observability

### Prometheus Metrics

Add metrics to your FastAPI application:

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('recommendations_requests_total', 'Total recommendation requests')
REQUEST_LATENCY = Histogram('recommendations_request_duration_seconds', 'Request latency')
ERROR_COUNT = Counter('recommendations_errors_total', 'Total errors')

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
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration

Set up structured logging:

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format='%(message)s'
)

logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(JSONFormatter())
```

## Performance Optimization

### Model Quantization

Reduce model size and inference time:

```python
import torch.quantization as quantization

def quantize_model(model, example_inputs):
    """Quantize model for faster inference"""
    # Prepare model for quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibrate with example inputs
    model_prepared.eval()
    with torch.no_grad():
        model_prepared(example_inputs)
    
    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared, inplace=False)
    
    return model_quantized

# Usage
example_input = torch.randint(0, 1000, (1, 50))
quantized_tiger = quantize_model(tiger_model, example_input)
```

### ONNX Export

Export models to ONNX for cross-platform deployment:

```python
def export_to_onnx(model, example_input, output_path):
    """Export PyTorch model to ONNX"""
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

# Export models
export_to_onnx(tiger_model, example_input, "models/tiger.onnx")
```

### TensorRT Optimization

Optimize models for NVIDIA GPUs:

```python
import tensorrt as trt

def convert_onnx_to_tensorrt(onnx_path, engine_path, max_batch_size=32):
    """Convert ONNX model to TensorRT engine"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (max_batch_size, 512), (max_batch_size, 512))
    config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

## A/B Testing Framework

### Experiment Configuration

Set up A/B testing for model comparisons:

```python
import hashlib
import random
from typing import Dict, Any

class ABTestFramework:
    def __init__(self, experiments: Dict[str, Any]):
        self.experiments = experiments
    
    def get_variant(self, user_id: int, experiment_name: str) -> str:
        """Get user's variant for an experiment"""
        if experiment_name not in self.experiments:
            return "control"
        
        experiment = self.experiments[experiment_name]
        
        # Use consistent hashing for user assignment
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
        """Check if user is in experiment"""
        if experiment_name not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_name]
        if not experiment.get('active', False):
            return False
        
        # Check eligibility criteria
        if 'eligibility' in experiment:
            # Implement eligibility logic
            pass
        
        return True

# Example experiment configuration
experiments_config = {
    "model_comparison": {
        "active": True,
        "salt": "experiment_salt_123",
        "variants": {
            "control": 50,  # 50% traffic
            "new_model": 50  # 50% traffic
        },
        "eligibility": {
            "min_interactions": 10
        }
    }
}

ab_tester = ABTestFramework(experiments_config)

@app.post("/recommend")
async def recommend_with_ab_test(request: RecommendationRequest):
    """Generate recommendations with A/B testing"""
    variant = ab_tester.get_variant(request.user_id, "model_comparison")
    
    if variant == "new_model":
        # Use new model
        recommendations = new_model_service.get_recommendations(
            request.user_history, request.num_recommendations
        )
    else:
        # Use control model
        recommendations = model_service.get_recommendations(
            request.user_history, request.num_recommendations
        )
    
    # Log experiment data
    logger.info("Recommendation served", extra={
        "user_id": request.user_id,
        "variant": variant,
        "experiment": "model_comparison"
    })
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations
    )
```

## Security Considerations

### API Authentication

Add JWT authentication:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
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
            detail="Invalid authentication credentials"
        )

@app.post("/recommend")
async def recommend(
    request: RecommendationRequest,
    token_data: dict = Depends(verify_token)
):
    """Protected recommendation endpoint"""
    # Verify user access
    if token_data.get("user_id") != request.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return await model_service.get_recommendations(request)
```

### Input Validation

Validate and sanitize inputs:

```python
from pydantic import validator

class RecommendationRequest(BaseModel):
    user_id: int
    user_history: List[int]
    num_recommendations: int = 10
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('User ID must be positive')
        return v
    
    @validator('user_history')
    def validate_user_history(cls, v):
        if len(v) > 1000:  # Limit history length
            raise ValueError('User history too long')
        if any(item <= 0 for item in v):
            raise ValueError('Invalid item IDs in history')
        return v
    
    @validator('num_recommendations')
    def validate_num_recommendations(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('Number of recommendations must be between 1 and 100')
        return v
```

This deployment guide covers the essential aspects of deploying genrec models in production, from basic API serving to advanced optimization and monitoring techniques.