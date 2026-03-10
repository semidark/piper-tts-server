source: https://deepwiki.com/search/is-litserve-a-good-choice-if-i_19ff37cc-b33b-40ac-a3f3-be0675f53955?mode=fast

# DeepWiki Q&A with Code Context for Repository: Lightning-AI/LitServe
## Q1
is litserve a good choice if i just want to use it purely an API/serving framework. Managing the HTTP server, lifecycle hooks but running the inference somewhere else?
### Answer
Yes, LitServe is an excellent choice for using purely as an API/serving framework while running inference elsewhere. The architecture cleanly separates HTTP server management from inference logic.

## Quick Answer

LitServe's `LitServer` class provides a production-ready HTTP server with lifecycle hooks, middleware support, and request management, while the `LitAPI.predict()` method can be implemented to make external inference calls instead of running local models LitServe:405-657 .

## Detailed Analysis

### HTTP Server Management

LitServe provides a robust HTTP server built on FastAPI with:
- Automatic endpoint registration (`/predict`, `/health`, `/info`) LitServe:1186-1346 
- SSL support and production-ready configuration LitServe:1281-1290 
- Multiple API server processes for load balancing LitServe:1236-1243 
- Built-in middleware support LitServe:532-546 

### Lifecycle Hooks

The framework supports comprehensive lifecycle management through:
- Callback system with events like `BEFORE_SETUP`, `AFTER_SETUP`, `ON_RESPONSE` LitServe:79-87 
- Worker setup status tracking LitServe:846-865 
- Graceful shutdown handling LitServe:1431-1432 

### External Inference Integration

You can implement `LitAPI.predict()` to call external services:

```python
import litserve as ls
import requests

class ExternalInferenceAPI(ls.LitAPI):
    def setup(self, device):
        self.inference_url = "https://your-inference-service.com/predict"
    
    def predict(self, request):
        # Make external API call instead of local inference
        response = requests.post(self.inference_url, json=request)
        return response.json()

server = ls.LitServer(ExternalInferenceAPI())
server.run(port=8000)
```

### Request/Response Customization

For more control over HTTP handling, use `LitSpec` to customize request decoding and response encoding LitServe:61-73 :

```python
from litserve.specs.base import LitSpec

class CustomAPISpec(LitSpec):
    def decode_request(self, request, meta_kwargs):
        # Custom request processing
        return processed_request
    
    def encode_response(self, output, meta_kwargs):
        # Custom response formatting
        return formatted_response
```

## Benefits for Your Use Case

- **Zero MLOps overhead**: No model loading, GPU management, or batching logic needed LitServe:66-69 
- **Production-ready**: Includes health checks, monitoring, and scaling features LitServe:488-515 
- **FastAPI optimized**: 2x faster than plain FastAPI for AI workloads LitServe:255-258 
- **Flexible deployment**: Self-host or deploy to Lightning AI Cloud LitServe:197-204 

## Notes

LitServe is designed specifically for AI inference serving, so while it excels as an API framework, it includes AI-specific optimizations like request batching and streaming that you may not need. However, these features can be disabled by using default configurations (`max_batch_size=1`, `stream=False`) LitServe:70-77 .

Wiki pages you might want to explore:
- [Core Concepts (Lightning-AI/LitServe)](/wiki/Lightning-AI/LitServe#2)
### Citations
**File:** src/litserve/server.py (L405-657)
```python
class LitServer:
    """Initialize a LitServer for high-performance AI model serving.

    LitServer transforms AI models into production-ready APIs with automatic scaling,
    batching, streaming, and multi-GPU support.

    Quick Start:
        ```python
        import litserve as ls

        # Define inference pipeline
        class MyAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_model()  # model loading logic

            def predict(self, x):
                return self.model(x)

        # Create and run server
        server = ls.LitServer(MyAPI())
        server.run(port=8000)
        ```

    Args:
        lit_api:
            The core component - one or more LitAPI instances defining model logic.

            - Single API: `MyAPI()` for serving one model
            - Multiple APIs: `[API1(), API2()]` for multi-model serving

            Each LitAPI must implement:
            - `setup(device)`: Initialize the model
            - `predict(x)`: Run inference
            - Optional: `decode_request()`, `encode_response()` for custom I/O

    Hardware Configuration:
        accelerator:
            Hardware type for inference. Defaults to "auto".

            - "auto": Automatically detects best available (CUDA > MPS > CPU)
            - "cpu": Force CPU usage
            - "cuda": Use NVIDIA GPUs
            - "mps": Use Apple Metal Performance Shaders

        devices:
            Number of devices to use. Defaults to "auto".

            - "auto": Use all available devices
            - int: Use specific number (e.g., 2 for 2 GPUs)

        workers_per_device:
            Worker processes per device for parallel inference. Defaults to 1.

            - Higher values = better throughput but more memory usage
            - Good starting point: 1-4 depending on model size
            - For CPU, set to the number of cores available on the machine (e.g., 8 for 8-core CPU)
            - Monitor GPU memory when increasing

    Performance & Scaling:
        timeout:
            Request timeout in seconds. Defaults to 30.

            - Set to False or -1 to disable timeouts
            - Increase for slow models (e.g., 300 for large LLMs)
            - Decrease for fast models (e.g., 5 for lightweight models)

        fast_queue:
            Enable ZeroMQ for high-throughput scenarios (>100 RPS). Defaults to False.

            - Use when serving hundreds of requests per second
            - Not supported on Windows

        track_requests:
            Track active requests across all API servers for monitoring and load management. Defaults to False.

            When enabled, tracks the total number of active requests in the queue across all API servers
            and makes this count available via callbacks using the `on_request` hook. Essential for
            monitoring concurrent request load and implementing custom load management logic.

            - Recommended for production deployments
            - Access count via callbacks or `server.active_requests` property
            - Useful for monitoring and handling concurrent requests effectively

    API Configuration:
        healthcheck_path:
            Health check endpoint for load balancers. Defaults to "/health".

            - Returns 200 when all workers are ready
            - Critical for Kubernetes/Docker deployments

        info_path:
            Server information endpoint. Defaults to "/info".

            - Shows model metadata, device info, server config
            - Useful for debugging and monitoring

        disable_openapi_url:
            If True, disables the OpenAPI schema endpoint ("/openapi.json").

            - Useful for production environments where exposing API schemas is not desired.
            - Defaults to False (the OpenAPI schema is enabled).

        shutdown_path:
            Graceful shutdown endpoint. Defaults to "/shutdown".

        enable_shutdown_api:
            Enable remote shutdown capability. Defaults to False.

            - Requires authentication token (set LIT_SHUTDOWN_API_KEY env var)
            - Useful for automated deployment pipelines

        restart_workers:
            Enable this option to automatically restart
            workers if a critical error occurs. Defaults to False.

            - When enabled, the worker loop will exit using `os._exit(1)`,
            allowing the main process to recreate the worker.


    Content & Middleware:
        max_payload_size:
            Maximum request size. Defaults to "100MB".

            - String format: "10MB", "1GB"
            - Integer format: bytes (1048576 for 1MB)
            - Increase for large images/videos

        middlewares:
            HTTP middleware for cross-cutting concerns. Defaults to None.

            Example:
            ```python
            from starlette.middleware.cors import CORSMiddleware

            server = LitServer(
                api,
                middlewares=[
                    (CORSMiddleware, {"allow_origins": ["*"]}),
                    # Add more middleware as needed
                ]
            )
            ```

        model_metadata:
            Metadata about the model displayed at info endpoint. Defaults to None.

            Example:
            ```python
            metadata = {
                "model_name": "bert-base-uncased",
                "version": "1.0.0",
                "description": "Text classification model"
            }
            ```

    Monitoring & Debugging:
        callbacks:
            Event handlers for server lifecycle. Defaults to None.

            - Built-in callbacks for logging, metrics, custom logic
            - Triggers on request start/end, server start/stop

        loggers:
            Custom loggers for metrics and events. Defaults to None.

            - Integrate with monitoring stack
            - Track performance metrics, error rates

    Advanced Configuration:
        max_batch_size, batch_timeout, spec, stream, api_path, loop:
            **Deprecated**: Configure these in LitAPI implementation instead.

            Migration example:
            ```python
            # Old way (deprecated)
            server = LitServer(api, max_batch_size=8, stream=True)

            # New way (recommended)
            api = MyAPI(max_batch_size=8, stream=True)
            server = LitServer(api)
            ```

    Examples:
        Basic Usage:
        ```python
        import litserve as ls

        class SimpleAPI(ls.LitAPI):
            def setup(self, device):
                self.model = lambda x: x * 2  # model here

            def predict(self, x):
                return self.model(x)

        server = ls.LitServer(SimpleAPI())
        server.run()
        ```

        Production Setup:
        ```python
        server = ls.LitServer(
            MyAPI(max_batch_size=8),
            accelerator="cuda",
            devices=2,
            workers_per_device=4,
            fast_queue=True,
            track_requests=True,
            max_payload_size="50MB",
            timeout=60
        )
        server.run(port=8000, num_api_servers=4)
        ```

        Multi-Model Serving:
        ```python
        # Serve multiple models on different endpoints
        text_api = TextClassifierAPI(api_path="/classify")
        image_api = ImageClassifierAPI(api_path="/vision")

        server = ls.LitServer([text_api, image_api])
        server.run()
        ```

        Streaming Response:
        ```python
        class StreamingAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_llm()

            def predict(self, prompt):
                for token in self.model.generate(prompt):
                    yield {"token": token}

        server = ls.LitServer(StreamingAPI(stream=True))
        ```

    Deployment:
        Self-hosted:
        ```bash
        python server.py  # Run locally
        ```

        Lightning AI Cloud:
        ```bash
        lightning deploy server.py --cloud  # One-click deploy
        ```

    See Also:
        - LitAPI: Base class for defining model logic
        - LitSpec: API specifications (OpenAI compatibility)
        - Documentation: https://lightning.ai/docs/litserve

    """
```
**File:** src/litserve/server.py (L846-865)
```python
            self.workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.STARTING

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=inference_worker,
                args=(
                    lit_api,
                    device,
                    worker_id,
                    self._get_request_queue(lit_api.api_path),
                    self._transport,
                    self.workers_setup_status,
                    self._callback_runner,
                    self.restart_workers,
                ),
                name="inference-worker",
            )
            process.start()
            process_list.append(process)
        return process_list
```
**File:** src/litserve/server.py (L1186-1346)
```python
    def run(
        self,
        host: str = "0.0.0.0",
        port: Union[str, int] = 8000,
        num_api_servers: Optional[int] = None,
        log_level: str = "info",
        generate_client_file: bool = True,
        api_server_worker_type: Literal["process", "thread"] = "process",
        pretty_logs: bool = False,
        **kwargs,
    ):
        """Start the LitServer to serve AI model requests with production-ready performance.

        This method launches the complete serving infrastructure: initializes worker processes,
        starts the HTTP server, and begins handling requests. The server runs until manually
        stopped (Ctrl+C) or programmatically shut down.

        Quick Start:
            ```python
            # Basic usage - starts server on localhost:8000
            server.run()

            # Production - multiple servers and custom port
            server.run(port=8080, num_api_servers=4)
            ```

        Server Lifecycle:
            1. **Initialize**: Sets up worker processes and communication queues
            2. **Health Check**: Verifies all workers are ready to serve requests
            3. **Start HTTP Server**: Begins accepting requests on specified host/port
            4. **Serve Requests**: Distributes requests to workers and returns responses
            5. **Graceful Shutdown**: Properly terminates workers when stopped

        Args:
            host:
                Network interface to bind the server to. Defaults to "0.0.0.0".

                - "0.0.0.0": Accept connections from any IP (public access)
                - "127.0.0.1": Only accept local connections (localhost only)
                - "::": IPv6 equivalent of "0.0.0.0"

                For development, use "127.0.0.1" for security. For production/Docker, use "0.0.0.0".

            port:
                Port number to listen on. Defaults to 8000.

                - Must be between 1024-65535 (privileged ports require admin)
                - Ensure the port is available and not blocked by firewalls
                - Common choices: 8000, 8080, 3000, 5000

        Performance Configuration:
            num_api_servers:
                Number of parallel HTTP server processes. Defaults to None (auto-detect).

                - None: Uses same count as inference workers (recommended)
                - Higher values improve HTTP throughput but use more memory
                - Good starting point: 2-8 depending on expected load
                - Each server handles HTTP requests independently

            api_server_worker_type:
                Process architecture for HTTP servers. Defaults to "process".

                - "process": Better isolation, CPU utilization, and fault tolerance
                - "thread": Lower memory usage but shared memory space
                - Windows automatically uses "thread" (process forking not supported)

        Development & Debugging:
            log_level:
                Logging verbosity level. Defaults to "info".

                - "critical": Only severe errors
                - "error": Error conditions
                - "warning": Warning messages (good for production)
                - "info": General information (default)
                - "debug": Detailed debugging info (development)
                - "trace": Very verbose output (troubleshooting)

            pretty_logs:
                Enable enhanced log formatting with colors and rich formatting. Defaults to False.

                - Requires: `pip install rich`
                - Great for development and local debugging
                - May not display properly in some production log aggregators

            generate_client_file:
                Auto-generate a Python client file for easy API interaction. Defaults to True.

                - Creates `client.py` in current directory with typed methods
                - Useful for testing and integration
                - Safe to disable in production environments

        Advanced Configuration:
            **kwargs:
                Additional uvicorn server configuration options.

                Common SSL options:
                ```python
                server.run(
                    ssl_keyfile="path/to/key.pem",
                    ssl_certfile="path/to/cert.pem"
                )
                ```

                Other uvicorn options: ssl_ca_certs, ssl_ciphers, ssl_version,
                workers, backlog, etc. See uvicorn documentation for full list.

        Examples:
            Basic Development:
            ```python
            # Simple local development
            server.run()
            # Access at: http://localhost:8000
            # API docs at: http://localhost:8000/docs
            ```

            Production Configuration:
            ```python
            # High-performance production setup
            server.run(
                host="0.0.0.0",
                port=8000,
                num_api_servers=8,
                log_level="warning",
                pretty_logs=False,
                generate_client_file=False
            )
            ```

            Development with Debug:
            ```python
            # Development with detailed logging
            server.run(
                host="127.0.0.1",
                port=8000,
                log_level="debug",
                pretty_logs=True,
                num_api_servers=1
            )
            ```

            Multi-API Server:
            ```python
            # Balance load across multiple HTTP servers
            server.run(
                port=8000,
                num_api_servers=4,  # 4 parallel HTTP servers
                api_server_worker_type="process"
            )
            ```

        Server Endpoints:
            Once running, the server provides several built-in endpoints:

            - **Main API**: `POST /predict` (or custom path from LitAPI)
            - **Health Check**: `GET /health` - Returns 200 when ready
            - **Server Info**: `GET /info` - Shows configuration and metadata
            - **API Documentation**: `GET /docs` - Interactive Swagger UI
            - **OpenAPI Schema**: `GET /openapi.json` - API specification

        Stopping the Server:
            - **Ctrl+C**: Graceful shutdown (recommended)
```
**File:** src/litserve/server.py (L1431-1432)
```python
        finally:
            self._perform_graceful_shutdown(manager, uvicorn_workers, shutdown_reason)
```
**File:** src/litserve/loops/loops.py (L79-87)
```python
    callback_runner.trigger_event(EventTypes.BEFORE_SETUP.value, lit_api=lit_api)
    try:
        lit_api.setup(device)
    except Exception:
        logger.exception(f"Error setting up worker {worker_id}.")
        workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.ERROR
        return
    lit_api.device = device
    callback_runner.trigger_event(EventTypes.AFTER_SETUP.value, lit_api=lit_api)
```
**File:** src/litserve/specs/base.py (L61-73)
```python
    @abstractmethod
    def decode_request(self, request, meta_kwargs):
        """Convert the request payload to your model input."""
        pass

    @abstractmethod
    def encode_response(self, output, meta_kwargs):
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        pass
```
**File:** README.md (L66-69)
```markdown
# Why LitServe?
LitServe lets you build your own inference engine. Serving engines such as vLLM serve specific model types (LLMs) with rigid abstractions. LitServe gives you the low-level control to serve any model (vision, audio, text, multi-modal), and define exactly how inference works - from batching, caching, streaming, and routing, to multi-model orchestration and custom logic. LitServe is perfect for building inference APIs, agents, chatbots, MCP servers, RAG, pipelines and more.

Self host LitServe or deploy in one-click to [Lightning AI](https://lightning.ai/litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme).
```
**File:** README.md (L197-204)
```markdown
# Host anywhere

Self-host with full control, or deploy with [Lightning AI](https://lightning.ai/?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) in seconds with autoscaling, security, and 99.995% uptime.  
**Free tier included. No setup required. Run on your cloud**   

```bash
lightning deploy server.py --cloud
```
```
**File:** README.md (L255-258)
```markdown
# Performance  
LitServe is designed for AI workloads. Specialized multi-worker handling delivers a minimum **2x speedup over FastAPI**.    

Additional features like batching and GPU autoscaling can drive performance well beyond 2x, scaling efficiently to handle more simultaneous requests than FastAPI and TorchServe.
```
**File:** src/litserve/api.py (L70-77)
```python
    Configuration:
        max_batch_size: Batch multiple requests for better GPU utilization. Defaults to 1.
        batch_timeout: Wait time for batch to fill (seconds). Defaults to 0.0.
        stream: Enable streaming responses for real-time output. Defaults to False.
        api_path: URL endpoint path. Defaults to "/predict".
        enable_async: Enable async/await for non-blocking operations. Defaults to False.
        spec: API specification (e.g., OpenAISpec for OpenAI compatibility). Defaults to None.
        mcp: Model Context Protocol integration for AI assistants. Defaults to None.
```