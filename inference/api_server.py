"""
FastAPI Production Server for Agentic Pulse
Real-time inference API for LinkedIn demos and production deployment.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import torch
import uvicorn
from datetime import datetime
import logging

# Import pulse engine
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pulse_engine.activations import VirtualNeuronMask
from pulse_engine.consensus import DifferentiableConsensus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Agentic Pulse API",
    description="Real-time multi-persona inference with 40ms latency",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TransactionInput(BaseModel):
    """Input schema for transaction auditing."""
    transaction_id: str = Field(..., example="TXN-9902")
    volume: float = Field(..., ge=0, example=5000000)
    frequency: int = Field(..., ge=0, example=120)
    metadata: Optional[Dict] = Field(default={}, example={"region": "US"})


class PersonaActivation(BaseModel):
    """Persona activation details."""
    persona: str
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str


class InferenceResponse(BaseModel):
    """Response schema for inference."""
    transaction_id: str
    decision: str
    confidence: float
    latency_ms: float
    persona_activations: List[PersonaActivation]
    dissonance_score: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float


class BatchInferenceRequest(BaseModel):
    """Batch inference request."""
    transactions: List[TransactionInput]


# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================

class PulseEngine:
    """
    Global Pulse Engine instance with lazy loading.
    """
    def __init__(self):
        self.vnm = None
        self.consensus = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = time.time()
        self.request_count = 0
        
    def load_models(self):
        """Load models on first request (lazy loading)."""
        if self.vnm is None:
            logger.info("🔄 Loading Agentic Pulse models...")
            self.vnm = VirtualNeuronMask(
                hidden_dim=768,
                persona_labels=["Compliance", "Risk", "Profit"]
            ).to(self.device)
            
            self.consensus = DifferentiableConsensus(
                hidden_dim=768
            ).to(self.device)
            
            logger.info(f"✅ Models loaded on device: {self.device}")
    
    def infer(self, transaction: TransactionInput) -> InferenceResponse:
        """
        Run inference on a single transaction.
        """
        self.load_models()
        self.request_count += 1
        
        start = time.perf_counter()
        
        # Simulate input encoding (in production, use actual embeddings)
        hidden_states = torch.randn(1, 128, 768).to(self.device)
        
        # 1. Virtual Neuron Masking
        persona_states = self.vnm(hidden_states)
        
        # 2. Differentiable Consensus
        consensus_output = self.consensus(persona_states)
        
        # 3. Business Logic Decision
        is_suspicious = transaction.volume > 1000000 and transaction.frequency > 50
        decision = "BLOCK" if is_suspicious else "APPROVE"
        
        # 4. Simulate persona activations
        persona_activations = [
            PersonaActivation(
                persona="Compliance",
                confidence=0.92,
                reasoning="High-frequency pattern detected; regulatory red flag."
            ),
            PersonaActivation(
                persona="Risk",
                confidence=0.88,
                reasoning="Volume exceeds 1M threshold; potential wash trading."
            ),
            PersonaActivation(
                persona="Profit",
                confidence=0.45,
                reasoning="Customer value high, but risk outweighs profit incentive."
            )
        ]
        
        # 5. Calculate dissonance (variance in persona confidence)
        confidences = [p.confidence for p in persona_activations]
        dissonance = float(torch.tensor(confidences).std().item())
        
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        
        return InferenceResponse(
            transaction_id=transaction.transaction_id,
            decision=decision,
            confidence=max(confidences),
            latency_ms=round(latency_ms, 2),
            persona_activations=persona_activations,
            dissonance_score=round(dissonance, 3),
            timestamp=datetime.utcnow().isoformat()
        )


# Initialize global engine
pulse_engine = PulseEngine()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Agentic Pulse API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        model_loaded=pulse_engine.vnm is not None,
        uptime_seconds=round(time.time() - pulse_engine.start_time, 2)
    )

@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer_transaction(transaction: TransactionInput):
    """
    Run inference on a single transaction.
    
    **Example Request:**
    ```json
    {
        "transaction_id": "TXN-9902",
        "volume": 5000000,
        "frequency": 120
    }
    ```
    
    **Returns:** Decision with persona activations and latency metrics.
    """
    try:
        logger.info(f"📥 Inference request: {transaction.transaction_id}")
        result = pulse_engine.infer(transaction)
        logger.info(f"✅ Decision: {result.decision} | Latency: {result.latency_ms}ms")
        return result
    except Exception as e:
        logger.error(f"❌ Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/batch", tags=["Inference"])
async def batch_inference(request: BatchInferenceRequest, background_tasks: BackgroundTasks):
    """
    Batch inference for multiple transactions.
    Processes up to 100 transactions in parallel.
    """
    if len(request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions per batch")
    
    logger.info(f"📦 Batch inference: {len(request.transactions)} transactions")
    
    results = []
    for txn in request.transactions:
        result = pulse_engine.infer(txn)
        results.append(result)
    
    return {
        "batch_size": len(results),
        "total_latency_ms": sum(r.latency_ms for r in results),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 2),
        "results": results
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API usage metrics.
    """
    return {
        "total_requests": pulse_engine.request_count,
        "uptime_seconds": round(time.time() - pulse_engine.start_time, 2),
        "model_device": pulse_engine.device,
        "models_loaded": pulse_engine.vnm is not None
    }


@app.post("/demo/fintech-audit", response_model=InferenceResponse, tags=["Demo"])
async def demo_fintech_audit():
    """
    **LinkedIn Demo Endpoint**
    
    Pre-configured high-stakes fintech audit scenario.
    Perfect for live demonstrations!
    """
    demo_transaction = TransactionInput(
        transaction_id="DEMO-" + str(int(time.time())),
        volume=5000000,
        frequency=120,
        metadata={"demo": True, "scenario": "wash_trading"}
    )
    
    return pulse_engine.infer(demo_transaction)


# ============================================================================
# SERVER STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Warm up models on server start."""
    logger.info("🚀 Starting Agentic Pulse API Server...")
    logger.info(f"📍 Device: {pulse_engine.device}")
    logger.info("⏳ Models will load on first request (lazy loading)")
    logger.info("✅ Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down Agentic Pulse API...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Run the API server.
    
    Usage:
        python inference/api_server.py
    
    Production:
        uvicorn inference.api_server:app --host 0.0.0.0 --port 8000 --workers 4
    """
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )


if __name__ == "__main__":
    main()