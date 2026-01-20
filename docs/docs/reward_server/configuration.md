# Reward Server Configuration

Comprehensive guide to configuring the Reward Server for different use cases and hardware.

## Environment Variables

All configuration is managed through environment variables using Pydantic Settings.

### Core Configuration

```bash
# Docking method: autodock_gpu or pyscreener
export DOCKING_ORACLE=autodock_gpu

# Path to data directory containing PDB files and targets
export DATA_PATH=data

# Path to log file
export LOG_FILE=mfput.log
```

### Docking Parameters

```bash
# Number of docking runs (higher = more accurate, slower)
# Recommended: 8
export SCORER_EXHAUSTIVENESS=8

# Number of CPU threads per docking run
# Recommended: match exhaustiveness value
export SCORER_NCPUS=8

# Vina command mode (for AutoDock-Vina)
# Options: autodock_gpu_256wi, autodock_gpu_128wi, autodock_gpu_64wi
export VINA_MODE=autodock_gpu_256wi
```

### GPU Configuration

```bash
# GPU utilization per docking run (0.0-1.0)
# Lower values = more parallel docking jobs, slower per-molecule
# Default: 0.05 (allows ~20 parallel jobs on V100)
export GPU_UTILIZATION_GPU_DOCKING=0.05

# GPU device ID (if multiple GPUs available)
export GPU_ID=0

# Maximum VRAM usage (GB)
export MAX_GPU_MEMORY=32
```

### Server Configuration

```bash
# Maximum number of concurrent HTTP requests
export MAX_CONCURRENT_REQUESTS=128

# Request timeout in seconds
export REQUEST_TIMEOUT=300

# Host and port (usually set via uvicorn args)
# uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

### Caching Configuration

```bash
# Enable result caching (True/False)
export ENABLE_CACHE=true

# Maximum cache size (number of molecules)
export CACHE_SIZE=10000

# Cache expiration time (seconds)
export CACHE_EXPIRATION=3600
```

## Configuration Profiles

### Development Setup (CPU)

Suitable for testing on CPU or small datasets:

```bash
export DOCKING_ORACLE=pyscreener
export SCORER_EXHAUSTIVENESS=4
export SCORER_NCPUS=4
export MAX_CONCURRENT_REQUESTS=2
export DATA_PATH=data
```

Start server:
```bash
uvicorn --reload --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

### Production Setup (Single GPU)

Optimized for NVIDIA GPU with good throughput:

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=8
export SCORER_NCPUS=8
export GPU_UTILIZATION_GPU_DOCKING=0.05
export MAX_CONCURRENT_REQUESTS=128
export REQUEST_TIMEOUT=300
export ENABLE_CACHE=true
export CACHE_SIZE=10000
export DATA_PATH=/data
export LOG_FILE=/var/log/molgen_server.log
```

Start server:
```bash
uvicorn --host 0.0.0.0 --port 8000 \
  --workers 4 \
  --loop uvloop \
  mol_gen_docking.server:app
```

### High-Throughput Setup (Multi-GPU)

For servers with multiple GPUs:

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=8
export SCORER_NCPUS=8
export GPU_UTILIZATION_GPU_DOCKING=0.1  # Slightly higher for more parallelism
export MAX_CONCURRENT_REQUESTS=256
export GPU_ID=0,1,2,3  # All GPUs
export ENABLE_CACHE=true
export CACHE_SIZE=50000
```

### Accuracy-Focused Setup

For maximum docking accuracy (slower):

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=16
export SCORER_NCPUS=16
export GPU_UTILIZATION_GPU_DOCKING=0.2
export MAX_CONCURRENT_REQUESTS=8
export REQUEST_TIMEOUT=600
```

## Docking Oracle Selection

### AutoDock-GPU (Recommended)

**Pros**:
- Fast GPU-accelerated computation
- Accurate binding affinity predictions
- Supports diverse ligand types
- Well-documented

**Cons**:
- Requires NVIDIA GPU
- Requires AutoDock-GPU installation

**Parameters**:
- `exhaustiveness`: 4-16 (default: 8)
- `num_modes`: Number of binding modes to report
- `energy_range`: Variation in predicted energies

**Configuration**:
```bash
export DOCKING_ORACLE=autodock_gpu
export VINA_MODE=autodock_gpu_256wi
```

### PyScreener (CPU Fallback)

**Pros**:
- Works on CPU
- Multiple docking tools supported
- No GPU required

**Cons**:
- Significantly slower than GPU
- Memory intensive

**Configuration**:
```bash
export DOCKING_ORACLE=pyscreener
export SCORER_NCPUS=8
```

## Database Configuration

### Data Directory Structure

```bash
export DATA_PATH=data

# Contents should include:
data/
├── molgendata/
│   ├── docking_targets.json
│   ├── pockets_info.json
│   ├── pdb_files/
│   │   ├── SAIR/
│   │   └── SIU/
│   └── names_mapping.json
├── fragments.json
└── properties.csv
```

### Loading Custom Targets

Edit or create `data/molgendata/docking_targets.json`:

```json
{
  "my_protein_001": {
    "pdb_code": "1ABC",
    "protein_name": "My Target Protein",
    "pocket": {
      "center_x": 10.5,
      "center_y": 20.3,
      "center_z": 15.8,
      "radius": 15.0
    },
    "pdb_file": "data/molgendata/pdb_files/my_protein.pdb"
  }
}
```

## Performance Tuning

### For Throughput (molecules/second)

Priority: Process as many molecules as possible

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=4      # Lower for speed
export GPU_UTILIZATION_GPU_DOCKING=0.05  # Parallel jobs
export MAX_CONCURRENT_REQUESTS=256
export ENABLE_CACHE=true
```

**Expected**: ~50-100 molecules/sec on V100

### For Accuracy (binding affinity correlation)

Priority: Accurate binding predictions

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=16     # Higher for accuracy
export GPU_UTILIZATION_GPU_DOCKING=0.2   # Fewer parallel jobs
export MAX_CONCURRENT_REQUESTS=8
```

**Expected**: ~10-20 molecules/sec on V100, higher accuracy

### For Memory Efficiency

On GPU with limited VRAM:

```bash
export GPU_UTILIZATION_GPU_DOCKING=0.1
export MAX_CONCURRENT_REQUESTS=16
export CACHE_SIZE=1000
export BATCH_SIZE=1
```

## Docker Configuration

### Docker Compose Example

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  molgen-server:
    image: fransou/molgendata:latest
    environment:
      DOCKING_ORACLE: autodock_gpu
      SCORER_EXHAUSTIVENESS: 8
      SCORER_NCPUS: 8
      GPU_UTILIZATION_GPU_DOCKING: 0.05
      MAX_CONCURRENT_REQUESTS: 128
      DATA_PATH: /data
      LOG_FILE: /var/log/molgen.log
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data:ro
      - ./logs:/var/log
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run:
```bash
docker-compose up -d
```

### Kubernetes Configuration

Example `molgen-server-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: molgen-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: molgen-server
  template:
    metadata:
      labels:
        app: molgen-server
    spec:
      containers:
      - name: server
        image: fransou/molgendata:latest
        ports:
        - containerPort: 8000
        env:
        - name: DOCKING_ORACLE
          value: "autodock_gpu"
        - name: SCORER_EXHAUSTIVENESS
          value: "8"
        - name: GPU_UTILIZATION_GPU_DOCKING
          value: "0.05"
        resources:
          limits:
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: molgen-data
```

## Monitoring and Logging

### Log Configuration

```bash
export LOG_FILE=mfput.log
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# View logs
tail -f mfput.log

# Filter for errors
grep ERROR mfput.log
```

### Metrics Exposure

Server exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `molgen_requests_total`: Total requests
- `molgen_request_duration_seconds`: Request latency
- `molgen_docking_computations_total`: Docking computations
- `molgen_cache_hits_total`: Cache hits
- `molgen_gpu_memory_usage_bytes`: GPU memory usage

## Security Configuration

### HTTPS Setup

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with SSL
uvicorn --host 0.0.0.0 --port 8000 \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem \
  mol_gen_docking.server:app
```

### Authentication

Add API key validation:

```python
# In server configuration
API_KEYS = ["your-secret-key-1", "your-secret-key-2"]

# Client sends Authorization header
requests.post(
    "http://localhost:8000/get_reward",
    headers={"Authorization": f"Bearer {api_key}"},
    json={...}
)
```

### Rate Limiting

```bash
export RATE_LIMIT_REQUESTS=1000
export RATE_LIMIT_PERIOD=3600  # Per hour
```

## Troubleshooting Configuration

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `GPU_UTILIZATION_GPU_DOCKING` or `MAX_CONCURRENT_REQUESTS` |
| Slow requests | Increase `SCORER_EXHAUSTIVENESS` if accuracy is priority, or decrease for speed |
| Cache issues | Clear cache: `export CACHE_SIZE=0` or restart server |
| GPU not detected | Check: `nvidia-smi`, verify CUDA installation |
| Timeout errors | Increase `REQUEST_TIMEOUT` or reduce `SCORER_EXHAUSTIVENESS` |

## Configuration Examples

### Complete `.env` File

```bash
# Core
DOCKING_ORACLE=autodock_gpu
DATA_PATH=/data
LOG_FILE=/var/log/molgen.log
LOG_LEVEL=INFO

# Docking
SCORER_EXHAUSTIVENESS=8
SCORER_NCPUS=8
VINA_MODE=autodock_gpu_256wi

# GPU
GPU_UTILIZATION_GPU_DOCKING=0.05
GPU_ID=0
MAX_GPU_MEMORY=32

# Server
MAX_CONCURRENT_REQUESTS=128
REQUEST_TIMEOUT=300

# Caching
ENABLE_CACHE=true
CACHE_SIZE=10000
CACHE_EXPIRATION=3600
```

Load with:
```bash
set -a
source .env
set +a
uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

## Next Steps

- [Getting Started](getting_started.md) - Quick setup guide
- [API Reference](api.md) - Complete API documentation
