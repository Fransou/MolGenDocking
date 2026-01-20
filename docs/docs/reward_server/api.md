| Docking (exhaustiveness=8) | 2000-5000 | GPU: V100 |
| Bioactivity (single) | 50-100 | Neural network |
| Total (all props) | 2500-5500 | Parallelized on GPU |

### Throughput

| Hardware | molecules/sec | Notes |
|----------|---------------|-------|
| V100 GPU | 50-100 | exhaustiveness=4 |
| A100 GPU | 100-200 | exhaustiveness=4 |
| CPU (8-core) | 1-5 | pyscreener |

## Compliance and Standards

- **API Version**: 1.0
- **OpenAPI Version**: 3.0.0
- **JSON Schema**: Draft 7
- **CORS**: Enabled for development (configurable)

## Related Documentation

- [Getting Started](getting_started.md) - Quick setup
- [Configuration](configuration.md) - Server configuration
- [Supported Properties](../datasets/properties.md) - Property definitions
# Reward Server API Reference

Complete API documentation for the MolGenDocking Reward Server.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses optional API key authentication via headers.

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### POST /get_reward

Main endpoint for scoring molecular structures.

#### Request

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "query": string,
  "prompt": string,
  "metadata": [
    {
      "target": string,
      "properties": [string],
      "pocket_path": string (optional)
    }
  ]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Completion containing molecule SMILES in `<answer>...</answer>` tags |
| `prompt` | string | Yes | Original prompt used for generation |
| `metadata` | array | Yes | List of evaluation contexts |
| `metadata.target` | string | Yes | Protein or property target (e.g., "GSK3B") |
| `metadata.properties` | array | Yes | List of properties to compute |
| `metadata.pocket_path` | string | No | Path to protein PDB file |

#### Response

**Status**: 200 OK

**Response Body**:
```json
{
  "query": string,
  "prompt": string,
  "status": "success" | "error",
  "reward": [
    {
      "target": string,
      "smiles": string,
      "validity": boolean,
      "docking_score": number (optional),
      "qed": number (optional),
      "sa": number (optional),
      "gsk3b": number (optional),
      "jnk3": number (optional),
      "drd2": number (optional),
      "molecular_weight": number (optional),
      "hba": number (optional),
      "hbd": number (optional),
      "logp": number (optional),
      "rotatable_bonds": number (optional),
      "aromatic_rings": number (optional),
      "computation_time": number
    }
  ],
  "error": string (optional)
}
```

#### Example Request

```python
import requests
import json

url = "http://localhost:8000/get_reward"

payload = {
    "query": "<answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
    "prompt": "Generate a drug-like molecule that inhibits GSK3B",
    "metadata": [{
        "target": "GSK3B",
        "properties": ["docking", "qed", "sa", "gsk3b", "molecular_weight"],
        "pocket_path": "data/molgendata/pockets/GSK3B.pdb"
    }]
}

response = requests.post(url, json=payload)
result = response.json()

print(json.dumps(result, indent=2))
```

#### Example Response

```json
{
  "query": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "prompt": "Generate a drug-like molecule that inhibits GSK3B",
  "status": "success",
  "reward": [
    {
      "target": "GSK3B",
      "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
      "validity": true,
      "docking_score": -8.5,
      "qed": 0.82,
      "sa": 3.5,
      "gsk3b": 6.5,
      "molecular_weight": 180.25,
      "hba": 2,
      "hbd": 1,
      "logp": 2.1,
      "rotatable_bonds": 3,
      "aromatic_rings": 1,
      "computation_time": 5.2
    }
  ],
  "error": null
}
```

#### Error Response

```json
{
  "query": "INVALID_SMILES",
  "prompt": "Generate a molecule",
  "status": "error",
  "reward": null,
  "error": "Invalid SMILES: INVALID_SMILES"
}
```

#### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | All properties computed successfully |
| 400 | Bad Request | Invalid JSON or missing required fields |
| 422 | Validation Error | Invalid field types or values |
| 504 | Gateway Timeout | Docking computation exceeded timeout |
| 500 | Internal Server Error | Server error (check logs) |

---

### GET /health

Health check endpoint.

#### Response

```json
{
  "status": "healthy",
  "docking_oracle": "autodock_gpu",
  "max_concurrent_requests": 128,
  "current_requests": 2,
  "gpu_available": true,
  "uptime_seconds": 3600
}
```

#### Example

```bash
curl http://localhost:8000/health
```

---

### GET /metrics

Prometheus metrics endpoint.

#### Response

Plain text Prometheus format:

```
# HELP molgen_requests_total Total requests processed
# TYPE molgen_requests_total counter
molgen_requests_total{method="get_reward"} 1234

# HELP molgen_request_duration_seconds Request duration in seconds
# TYPE molgen_request_duration_seconds histogram
molgen_request_duration_seconds_bucket{le="1.0"} 800
molgen_request_duration_seconds_bucket{le="5.0"} 1200
molgen_request_duration_seconds_bucket{le="+Inf"} 1234

# HELP molgen_docking_computations_total Total docking computations
# TYPE molgen_docking_computations_total counter
molgen_docking_computations_total 800

# HELP molgen_cache_hits_total Total cache hits
# TYPE molgen_cache_hits_total counter
molgen_cache_hits_total 400

# HELP molgen_gpu_memory_usage_bytes GPU memory usage in bytes
# TYPE molgen_gpu_memory_usage_bytes gauge
molgen_gpu_memory_usage_bytes 8589934592
```

#### Example

```bash
curl http://localhost:8000/metrics
```

---

### GET /docs

Interactive API documentation (Swagger UI).

Open browser to: `http://localhost:8000/docs`

---

### GET /redoc

Alternative API documentation (ReDoc).

Open browser to: `http://localhost:8000/redoc`

---

## Data Types

### Property List

Available properties that can be requested:

```python
AVAILABLE_PROPERTIES = [
    # Docking
    "docking",
    "docking_score",

    # Drug-likeness
    "qed",
    "sa",

    # Bioactivity
    "gsk3b",
    "jnk3",
    "drd2",

    # Physicochemical
    "molecular_weight",
    "hba",
    "hbd",
    "logp",
    "rotatable_bonds",
    "aromatic_rings"
]
```

### Target List

Pre-configured docking targets:

```python
AVAILABLE_TARGETS = [
    "GSK3B",
    "JNK3",
    "DRD2",
    # ... plus 50+ others
]
```

## Rate Limiting

Default rate limits (configurable):

- **Per IP**: 1000 requests/hour
- **Per API key**: 10,000 requests/hour
- **Concurrent requests**: 128 (configurable)

Rate limit info is returned in response headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642603200
```

## Pagination

Currently not applicable (single molecule per request).

For batch processing, use separate requests or implement client-side batching.

## Error Handling

### Common Errors

#### Invalid SMILES

```json
{
  "status": "error",
  "error": "Invalid SMILES: CCX"
}
```

**Solution**: Use valid SMILES notation. Validate with RDKit:
```python
from rdkit import Chem
Chem.MolFromSmiles("CCX")  # Returns None
```

#### Missing Required Fields

```json
{
  "status": "error",
  "error": "Missing required field: metadata"
}
```

**Solution**: Include all required fields in request

#### Protein File Not Found

```json
{
  "status": "error",
  "error": "Protein file not found: data/pockets/NONEXISTENT.pdb"
}
```

**Solution**: Verify `DATA_PATH` and file paths

#### Timeout

```json
{
  "status": "error",
  "error": "Request timeout: docking computation exceeded 300 seconds"
}
```

**Solution**: Reduce `SCORER_EXHAUSTIVENESS` or increase timeout limit

#### Out of Memory

```json
{
  "status": "error",
  "error": "CUDA out of memory"
}
```

**Solution**: Reduce `GPU_UTILIZATION_GPU_DOCKING` or `MAX_CONCURRENT_REQUESTS`

## Code Examples

### Python with Requests

```python
import requests

def score_molecule(smiles, target="GSK3B"):
    response = requests.post(
        "http://localhost:8000/get_reward",
        json={
            "query": f"<answer>{smiles}</answer>",
            "prompt": "Generate molecules",
            "metadata": [{
                "target": target,
                "properties": ["docking", "qed", "sa"]
            }]
        }
    )

    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            return data['reward'][0]
        else:
            print(f"Error: {data['error']}")
    else:
        print(f"HTTP {response.status_code}")

    return None

# Example usage
result = score_molecule("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
if result:
    print(f"Docking Score: {result['docking_score']:.2f}")
    print(f"QED: {result['qed']:.3f}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function scoreMolecule(smiles, target = 'GSK3B') {
  try {
    const response = await axios.post('http://localhost:8000/get_reward', {
      query: `<answer>${smiles}</answer>`,
      prompt: 'Generate molecules',
      metadata: [{
        target: target,
        properties: ['docking', 'qed', 'sa']
      }]
    });

    if (response.data.status === 'success') {
      return response.data.reward[0];
    } else {
      console.error('Error:', response.data.error);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }

  return null;
}

// Example usage
scoreMolecule('CC(C)Cc1ccc(cc1)C(C)C(=O)O').then(result => {
  if (result) {
    console.log(`Docking Score: ${result.docking_score.toFixed(2)}`);
  }
});
```

### Shell/curl

```bash
#!/bin/bash

SMILES="CC(C)Cc1ccc(cc1)C(C)C(=O)O"
TARGET="GSK3B"

curl -X POST "http://localhost:8000/get_reward" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"<answer>$SMILES</answer>\",
    \"prompt\": \"Generate molecules\",
    \"metadata\": [{
      \"target\": \"$TARGET\",
      \"properties\": [\"docking\", \"qed\", \"sa\"]
    }]
  }" | jq '.reward[0]'
```

### Streaming/Batch

```python
import requests
import json
from typing import List, Dict

def batch_score_molecules(
    smiles_list: List[str],
    target: str = "GSK3B",
    properties: List[str] = None
) -> List[Dict]:

    if properties is None:
        properties = ["docking", "qed", "sa"]

    results = []

    for smiles in smiles_list:
        response = requests.post(
            "http://localhost:8000/get_reward",
            json={
                "query": f"<answer>{smiles}</answer>",
                "prompt": f"Score for {target}",
                "metadata": [{
                    "target": target,
                    "properties": properties
                }]
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                results.append(data['reward'][0])
            else:
                results.append({'error': data['error']})
        else:
            results.append({'error': f'HTTP {response.status_code}'})

    return results

# Usage
molecules = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CCOc1ccc(cc1)C(=O)O",
    "CC(=O)c1ccccc1"
]

results = batch_score_molecules(molecules, target="GSK3B")
for i, (smiles, result) in enumerate(zip(molecules, results)):
    print(f"{i+1}. {smiles}")
    if 'error' in result:
        print(f"   Error: {result['error']}")
    else:
        print(f"   Docking: {result['docking_score']:.2f}")
        print(f"   QED: {result['qed']:.3f}")
```

## Performance Specifications

### Latency

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| QED computation | 2-5 | Per molecule |
| SA computation | 3-8 | Per molecule |
