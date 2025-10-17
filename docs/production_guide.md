# Production Deployment Guide

## 🎯 Production-Ready Implementation

This guide covers deploying InstrumentTimbre in production environments with real users and performance requirements.

## 📋 Pre-Deployment Checklist

### System Requirements Validation

- [ ] **CPU**: 8+ cores for concurrent processing
- [ ] **Memory**: 32GB RAM for production loads
- [ ] **Storage**: SSD with 100GB+ free space
- [ ] **Network**: Stable internet for model updates
- [ ] **OS**: Linux (Ubuntu 20.04+) or CentOS 8+ recommended

### Performance Benchmarks

- [ ] **Latency**: <5 seconds per 3-minute audio file
- [ ] **Throughput**: 100+ concurrent predictions
- [ ] **Accuracy**: >95% on validation data
- [ ] **Uptime**: 99.9% availability target

## 🚀 Deployment Strategies

### Option 1: Docker Containerization

```bash
# Build production container
docker build -t instrumenttimbre:production .

# Run with resource limits
docker run -d \
  --name instrumenttimbre-prod \
  --memory=16g \
  --cpus=8 \
  -p 8080:8080 \
  -v /data:/app/data \
  instrumenttimbre:production
```

### Option 2: Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: instrumenttimbre
spec:
  replicas: 3
  selector:
    matchLabels:
      app: instrumenttimbre
  template:
    metadata:
      labels:
        app: instrumenttimbre
    spec:
      containers:
      - name: instrumenttimbre
        image: instrumenttimbre:production
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
```

### Option 3: Cloud Deployment

#### AWS EC2 Setup

```bash
# Launch optimized instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type c5.4xlarge \
  --key-name production-key \
  --security-group-ids sg-12345678
```

#### Google Cloud Platform

```bash
# Create compute instance
gcloud compute instances create instrumenttimbre-prod \
  --machine-type=c2-standard-16 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud
```

## 🔧 Configuration for Production

### Environment Variables

```bash
export INSTRUMENTTIMBRE_ENV=production
export INSTRUMENTTIMBRE_LOG_LEVEL=WARNING
export INSTRUMENTTIMBRE_CACHE_SIZE=10000
export INSTRUMENTTIMBRE_MAX_WORKERS=16
export INSTRUMENTTIMBRE_BATCH_SIZE=64
```

### Production Configuration File

```yaml
# config/production.yaml
environment: production
performance:
  batch_size: 64
  max_workers: 16
  cache_size: 10000
  timeout_seconds: 30

models:
  default: enhanced_cnn  # Balanced performance
  fast: cnn             # For real-time requirements
  accurate: transformer # For highest accuracy

logging:
  level: WARNING
  file: /var/log/instrumenttimbre/app.log
  rotation: daily
  retention_days: 30

monitoring:
  metrics_enabled: true
  health_check_port: 8081
  prometheus_port: 9090
```

## 📊 Monitoring & Observability

### Health Check Endpoint

```python
# Health check implementation
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'memory_usage': get_memory_usage(),
        'uptime_seconds': get_uptime()
    }
```

### Prometheus Metrics

```python
# Key metrics to monitor
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('errors_total', 'Total errors', ['error_type'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Currently active requests')
```

### Grafana Dashboard Config

```json
{
  "dashboard": {
    "title": "InstrumentTimbre Production",
    "panels": [
      {
        "title": "Predictions per Second",
        "targets": [{"expr": "rate(predictions_total[5m])"}]
      },
      {
        "title": "Average Latency",
        "targets": [{"expr": "histogram_quantile(0.95, prediction_duration_seconds)"}]
      },
      {
        "title": "Error Rate",
        "targets": [{"expr": "rate(errors_total[5m])"}]
      }
    ]
  }
}
```

## 🛡️ Security Considerations

### API Security

```python
# Rate limiting
from flask_limiter import Limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

# Input validation
def validate_audio_file(file):
    allowed_extensions = {'.wav', '.mp3', '.flac'}
    max_size = 100 * 1024 * 1024  # 100MB
  
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise ValueError("Unsupported file format")
  
    if len(file.read()) > max_size:
        raise ValueError("File too large")
```

### Network Security

```bash
# Firewall configuration
ufw allow 22/tcp      # SSH
ufw allow 8080/tcp    # Application
ufw allow 8081/tcp    # Health check
ufw deny 9090/tcp     # Prometheus (internal only)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: Production Deployment
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
  
    - name: Run Tests
      run: |
        python -m pytest tests/
        python scripts/test_architecture.py
  
    - name: Build Docker Image
      run: |
        docker build -t instrumenttimbre:${{ github.ref_name }} .
        docker tag instrumenttimbre:${{ github.ref_name }} instrumenttimbre:latest
  
    - name: Deploy to Production
      run: |
        docker push instrumenttimbre:${{ github.ref_name }}
        kubectl set image deployment/instrumenttimbre instrumenttimbre=instrumenttimbre:${{ github.ref_name }}
```

## 📈 Performance Optimization

### Model Optimization

```python
# Model quantization for faster inference
import torch
model = torch.load('model.pth')
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model, 'model_quantized.pth')
```

### Caching Strategy

```python
import redis
import pickle

class PredictionCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.ttl = 3600  # 1 hour
  
    def get_prediction(self, audio_hash):
        cached = self.redis_client.get(f"pred:{audio_hash}")
        return pickle.loads(cached) if cached else None
  
    def cache_prediction(self, audio_hash, prediction):
        self.redis_client.setex(
            f"pred:{audio_hash}", 
            self.ttl, 
            pickle.dumps(prediction)
        )
```

### Load Balancing

```nginx
# nginx.conf
upstream instrumenttimbre {
    server 127.0.0.1:8080 weight=3;
    server 127.0.0.1:8081 weight=3;
    server 127.0.0.1:8082 weight=3;
}

server {
    listen 80;
    location /predict {
        proxy_pass http://instrumenttimbre;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🎯 Quality Assurance

### Automated Testing

```python
def test_production_accuracy():
    """Test model accuracy on validation set"""
    predictor = InstrumentPredictor('production_model.pth')
  
    correct = 0
    total = 0
  
    for audio_file, true_label in validation_set:
        prediction = predictor.predict_file(audio_file)
        if prediction['top_prediction']['class'] == true_label:
            correct += 1
        total += 1
  
    accuracy = correct / total
    assert accuracy > 0.95, f"Production accuracy {accuracy} below threshold"
```

### Performance Testing

```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 -T 'audio/wav' -p test_audio.wav http://localhost:8080/predict

# Expected results:
# - 99% of requests < 5 seconds
# - No failed requests
# - Steady memory usage
```

## 🚨 Incident Response

### Alert Configuration

```yaml
# alertmanager.yml
alerts:
  - name: high_error_rate
    condition: rate(errors_total[5m]) > 0.1
    message: "Error rate above 10%"
    severity: critical
  
  - name: high_latency
    condition: histogram_quantile(0.95, prediction_duration_seconds) > 10
    message: "95th percentile latency above 10 seconds"
    severity: warning
```

### Rollback Procedure

```bash
# Quick rollback script
#!/bin/bash
PREVIOUS_VERSION=$(kubectl get deployment instrumenttimbre -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')
kubectl rollout undo deployment/instrumenttimbre --to-revision=$((PREVIOUS_VERSION-1))
kubectl rollout status deployment/instrumenttimbre
```

## 📋 Maintenance Procedures

### Regular Tasks

- **Daily**: Check logs for errors, monitor performance metrics
- **Weekly**: Update dependencies, run accuracy validation
- **Monthly**: Model retraining with new data, performance optimization
- **Quarterly**: Security audit, disaster recovery testing

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf "backup_$DATE.tar.gz" models/ config/ data/
aws s3 cp "backup_$DATE.tar.gz" s3://instrumenttimbre-backups/
```

## 🎉 Success Metrics

### KPIs to Track

- **Accuracy**: >95% on production data
- **Latency**: 95th percentile < 5 seconds
- **Availability**: >99.9% uptime
- **Throughput**: >100 predictions/minute
- **User Satisfaction**: >4.5/5 rating

### Production Readiness Verification

- [ ] Load testing passed (1000+ concurrent users)
- [ ] Monitoring and alerting configured
- [ ] Security audit completed
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated
- [ ] Team training completed

---

**🚀 Your InstrumentTimbre system is now production-ready!**
