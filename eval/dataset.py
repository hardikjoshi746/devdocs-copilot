"""
eval/dataset.py

50 hand-written Q&A pairs for evaluating the RAG pipeline.
Stratified across 4 categories:
- Factual (15):     specific API facts, parameter names, status codes
- Conceptual (15):  how/why questions about FastAPI design
- Cross-source (10): questions that span docs + source code + issues
- Debug/code (10):  questions about common errors and fixes

Each entry has:
- question:              the user query
- expected_answer:       reference answer for correctness scoring
- expected_source_ids:   chunk ids that SHOULD appear in top-5 retrieval
                         (populated after running bootstrap_source_ids())

Run bootstrap_source_ids() once to auto-populate expected_source_ids
by running the pipeline against each question and saving the top result.
Then manually verify the ids are actually correct.
"""

import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATASET: list[dict] = [

    # ------------------------------------------------------------------
    # FACTUAL (15) — specific API facts, parameters, return types
    # ------------------------------------------------------------------
    {
        "id": "f01",
        "category": "factual",
        "question": "What parameters does HTTPException accept?",
        "expected_answer": "HTTPException accepts status_code (required), detail (optional, any data), and headers (optional dict).",
        "expected_source_ids": [],
    },
    {
        "id": "f02",
        "category": "factual",
        "question": "What is the default response class in FastAPI?",
        "expected_answer": "The default response class in FastAPI is JSONResponse.",
        "expected_source_ids": [],
    },
    {
        "id": "f03",
        "category": "factual",
        "question": "How do you declare a path parameter in FastAPI?",
        "expected_answer": "Declare a path parameter by including it in the path string with curly braces and as a function argument: @app.get('/items/{item_id}') def read_item(item_id: int).",
        "expected_source_ids": [],
    },
    {
        "id": "f04",
        "category": "factual",
        "question": "What decorator is used to define a POST endpoint in FastAPI?",
        "expected_answer": "@app.post('/path') is used to define a POST endpoint.",
        "expected_source_ids": [],
    },
    {
        "id": "f05",
        "category": "factual",
        "question": "How do you make a query parameter optional in FastAPI?",
        "expected_answer": "Make a query parameter optional by setting a default value, typically None: def read_items(q: str | None = None).",
        "expected_source_ids": [],
    },
    {
        "id": "f06",
        "category": "factual",
        "question": "What is the purpose of the Depends function in FastAPI?",
        "expected_answer": "Depends is used to declare dependencies — functions that FastAPI will call and inject as arguments into your path operation functions.",
        "expected_source_ids": [],
    },
    {
        "id": "f07",
        "category": "factual",
        "question": "How do you define a request body in FastAPI?",
        "expected_answer": "Define a request body by creating a Pydantic BaseModel subclass and declaring it as a parameter in the path operation function.",
        "expected_source_ids": [],
    },
    {
        "id": "f08",
        "category": "factual",
        "question": "What HTTP status code does a successful POST typically return in FastAPI?",
        "expected_answer": "By default FastAPI returns 200, but for POST you can set status_code=201 using the status_code parameter in the decorator.",
        "expected_source_ids": [],
    },
    {
        "id": "f09",
        "category": "factual",
        "question": "How do you add tags to a FastAPI route?",
        "expected_answer": "Add tags using the tags parameter in the route decorator: @app.get('/items', tags=['items']).",
        "expected_source_ids": [],
    },
    {
        "id": "f10",
        "category": "factual",
        "question": "What is BackgroundTasks in FastAPI?",
        "expected_answer": "BackgroundTasks allows you to run functions in the background after returning a response to the client.",
        "expected_source_ids": [],
    },
    {
        "id": "f11",
        "category": "factual",
        "question": "How do you set cookies in a FastAPI response?",
        "expected_answer": "Set cookies using response.set_cookie(key, value) where response is a Response parameter injected into the path operation.",
        "expected_source_ids": [],
    },
    {
        "id": "f12",
        "category": "factual",
        "question": "What is the purpose of response_model in FastAPI?",
        "expected_answer": "response_model filters and validates the response data, ensuring only the declared fields are returned and sensitive fields are excluded.",
        "expected_source_ids": [],
    },
    {
        "id": "f13",
        "category": "factual",
        "question": "How do you declare a header parameter in FastAPI?",
        "expected_answer": "Declare a header parameter using Header: from fastapi import Header, then def read_items(user_agent: str | None = Header(default=None)).",
        "expected_source_ids": [],
    },
    {
        "id": "f14",
        "category": "factual",
        "question": "What is APIRouter used for in FastAPI?",
        "expected_answer": "APIRouter is used to organize routes into separate modules/files, which are then included in the main app with app.include_router().",
        "expected_source_ids": [],
    },
    {
        "id": "f15",
        "category": "factual",
        "question": "How do you return a custom HTTP status code in FastAPI?",
        "expected_answer": "Set the status_code parameter in the route decorator: @app.post('/items', status_code=201), or return a Response object with the desired status code.",
        "expected_source_ids": [],
    },

    # ------------------------------------------------------------------
    # CONCEPTUAL (15) — how/why questions about FastAPI design
    # ------------------------------------------------------------------
    {
        "id": "c01",
        "category": "conceptual",
        "question": "How does dependency injection work in FastAPI?",
        "expected_answer": "FastAPI's dependency injection system calls functions declared with Depends() automatically, passing their return values to path operation functions. Dependencies can have their own dependencies, forming a dependency tree.",
        "expected_source_ids": [],
    },
    {
        "id": "c02",
        "category": "conceptual",
        "question": "Why does FastAPI use Pydantic for data validation?",
        "expected_answer": "Pydantic provides automatic data validation, serialization, and JSON schema generation. FastAPI uses it to validate request bodies, query parameters, and response models with Python type hints.",
        "expected_source_ids": [],
    },
    {
        "id": "c03",
        "category": "conceptual",
        "question": "What is the difference between async and sync path operations in FastAPI?",
        "expected_answer": "Async path operations (async def) run in the event loop and should be used with async I/O. Sync path operations (def) are run in a thread pool to avoid blocking the event loop.",
        "expected_source_ids": [],
    },
    {
        "id": "c04",
        "category": "conceptual",
        "question": "How does FastAPI generate OpenAPI documentation automatically?",
        "expected_answer": "FastAPI inspects type hints, Pydantic models, and route decorators to generate an OpenAPI schema. This schema powers the automatic Swagger UI at /docs and ReDoc at /redoc.",
        "expected_source_ids": [],
    },
    {
        "id": "c05",
        "category": "conceptual",
        "question": "What is the lifespan event in FastAPI and when would you use it?",
        "expected_answer": "The lifespan context manager handles startup and shutdown events. Use it to initialize resources (database connections, ML models) on startup and clean them up on shutdown.",
        "expected_source_ids": [],
    },
    {
        "id": "c06",
        "category": "conceptual",
        "question": "How does FastAPI handle request validation errors?",
        "expected_answer": "FastAPI automatically returns a 422 Unprocessable Entity response with details about validation errors when request data doesn't match the declared types or constraints.",
        "expected_source_ids": [],
    },
    {
        "id": "c07",
        "category": "conceptual",
        "question": "What is the difference between Path, Query, and Body parameters in FastAPI?",
        "expected_answer": "Path parameters are part of the URL path, Query parameters are in the URL query string, and Body parameters come from the request body. FastAPI infers which is which from the function signature and type hints.",
        "expected_source_ids": [],
    },
    {
        "id": "c08",
        "category": "conceptual",
        "question": "How does FastAPI handle CORS?",
        "expected_answer": "FastAPI handles CORS via CORSMiddleware from starlette. Add it with app.add_middleware(CORSMiddleware, allow_origins=[...], allow_methods=[...]).",
        "expected_source_ids": [],
    },
    {
        "id": "c09",
        "category": "conceptual",
        "question": "What is middleware in FastAPI and how does it work?",
        "expected_answer": "Middleware is code that runs before and after every request. It wraps the entire application, allowing you to modify requests/responses, add headers, log requests, or handle authentication globally.",
        "expected_source_ids": [],
    },
    {
        "id": "c10",
        "category": "conceptual",
        "question": "How does FastAPI handle file uploads?",
        "expected_answer": "FastAPI handles file uploads using UploadFile and File from fastapi. Declare the parameter as: async def upload(file: UploadFile = File(...)). The file content is accessible via await file.read().",
        "expected_source_ids": [],
    },
    {
        "id": "c11",
        "category": "conceptual",
        "question": "What is the purpose of yield in FastAPI dependencies?",
        "expected_answer": "Using yield in a dependency allows setup code before yield and teardown code after. FastAPI runs teardown after the response is sent, making it ideal for database sessions and resource cleanup.",
        "expected_source_ids": [],
    },
    {
        "id": "c12",
        "category": "conceptual",
        "question": "How does FastAPI support WebSockets?",
        "expected_answer": "FastAPI supports WebSockets via the WebSocket class. Declare a route with @app.websocket('/ws') and accept connections with await websocket.accept(). Send/receive with send_text() and receive_text().",
        "expected_source_ids": [],
    },
    {
        "id": "c13",
        "category": "conceptual",
        "question": "What is the difference between include_router and mounting a sub-application in FastAPI?",
        "expected_answer": "include_router adds routes from an APIRouter to the main app, sharing middleware and exception handlers. Mounting a sub-application creates a separate ASGI app with its own middleware stack.",
        "expected_source_ids": [],
    },
    {
        "id": "c14",
        "category": "conceptual",
        "question": "How does FastAPI handle authentication?",
        "expected_answer": "FastAPI provides security utilities like OAuth2PasswordBearer, HTTPBasic, and APIKey. These are used as dependencies that extract and validate credentials from requests.",
        "expected_source_ids": [],
    },
    {
        "id": "c15",
        "category": "conceptual",
        "question": "What is the role of Starlette in FastAPI?",
        "expected_answer": "FastAPI is built on top of Starlette, which provides the ASGI framework, routing, middleware, WebSocket support, and request/response handling. FastAPI adds type hints, Pydantic validation, and OpenAPI generation on top.",
        "expected_source_ids": [],
    },

    # ------------------------------------------------------------------
    # CROSS-SOURCE (10) — span docs + source code + issues
    # ------------------------------------------------------------------
    {
        "id": "x01",
        "category": "cross-source",
        "question": "How is HTTPException defined in the FastAPI source code?",
        "expected_answer": "HTTPException is defined in fastapi/exceptions.py. It inherits from Starlette's HTTPException and adds support for custom headers.",
        "expected_source_ids": [],
    },
    {
        "id": "x02",
        "category": "cross-source",
        "question": "What does the FastAPI source say about how Depends is resolved?",
        "expected_answer": "Depends is resolved by the dependency injection system in FastAPI's routing layer. The dependant object stores the dependency tree and FastAPI calls each dependency in order.",
        "expected_source_ids": [],
    },
    {
        "id": "x03",
        "category": "cross-source",
        "question": "How does BackgroundTasks work according to both the docs and source code?",
        "expected_answer": "BackgroundTasks collects functions to run after the response is sent. In the source, it's implemented using Starlette's BackgroundTasks which runs tasks after the response completes.",
        "expected_source_ids": [],
    },
    {
        "id": "x04",
        "category": "cross-source",
        "question": "What are the known issues or limitations with FastAPI's dependency injection?",
        "expected_answer": "Known issues include difficulties with async generators in some contexts, and challenges with testing dependencies. Some issues relate to dependency caching behavior.",
        "expected_source_ids": [],
    },
    {
        "id": "x05",
        "category": "cross-source",
        "question": "How does FastAPI's routing work internally?",
        "expected_answer": "FastAPI routing extends Starlette's Router. Routes are stored as APIRoute objects that wrap path operations with validation, serialization, and dependency injection logic.",
        "expected_source_ids": [],
    },
    {
        "id": "x06",
        "category": "cross-source",
        "question": "What is the FastAPI application class and what does it inherit from?",
        "expected_answer": "The FastAPI class is defined in fastapi/applications.py and inherits from Starlette. It adds OpenAPI schema generation, automatic documentation, and enhanced routing.",
        "expected_source_ids": [],
    },
    {
        "id": "x07",
        "category": "cross-source",
        "question": "How are response models validated in FastAPI source code?",
        "expected_answer": "Response models are validated using Pydantic's model_validate in the response serialization step. FastAPI's routing layer calls serialize_response() which applies the response_model filter.",
        "expected_source_ids": [],
    },
    {
        "id": "x08",
        "category": "cross-source",
        "question": "What does FastAPI do when a path operation raises an unhandled exception?",
        "expected_answer": "Unhandled exceptions propagate to Starlette's exception handler middleware, which returns a 500 Internal Server Error. Custom exception handlers can be registered with app.add_exception_handler().",
        "expected_source_ids": [],
    },
    {
        "id": "x09",
        "category": "cross-source",
        "question": "How does FastAPI generate JSON schemas from Pydantic models?",
        "expected_answer": "FastAPI calls Pydantic's model_json_schema() to generate JSON Schema from BaseModel subclasses. These schemas are included in the OpenAPI spec for request/response documentation.",
        "expected_source_ids": [],
    },
    {
        "id": "x10",
        "category": "cross-source",
        "question": "What is the FastAPI TestClient and how is it implemented?",
        "expected_answer": "TestClient is re-exported from Starlette and uses httpx under the hood to make synchronous HTTP requests to the ASGI app without running a real server.",
        "expected_source_ids": [],
    },

    # ------------------------------------------------------------------
    # DEBUG/CODE (10) — common errors and fixes
    # ------------------------------------------------------------------
    {
        "id": "d01",
        "category": "debug",
        "question": "Why would a sync route block the event loop in FastAPI?",
        "expected_answer": "A sync route (def, not async def) runs in a thread pool by default. However, if you use async def but call blocking I/O without await, it blocks the event loop and prevents other requests from being processed.",
        "expected_source_ids": [],
    },
    {
        "id": "d02",
        "category": "debug",
        "question": "How do I fix a 422 Unprocessable Entity error in FastAPI?",
        "expected_answer": "A 422 error means request validation failed. Check that the request body, query params, or path params match the declared types. The error response body contains details about which fields failed and why.",
        "expected_source_ids": [],
    },
    {
        "id": "d03",
        "category": "debug",
        "question": "Why am I getting a 'field required' validation error in FastAPI?",
        "expected_answer": "A 'field required' error means a required field is missing from the request. Either provide the field, make it optional with a default value (= None), or use Optional[type] in the Pydantic model.",
        "expected_source_ids": [],
    },
    {
        "id": "d04",
        "category": "debug",
        "question": "How do I handle CORS errors in FastAPI?",
        "expected_answer": "Add CORSMiddleware: app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*']). Restrict allow_origins to specific domains in production.",
        "expected_source_ids": [],
    },
    {
        "id": "d05",
        "category": "debug",
        "question": "Why is my FastAPI dependency not being called?",
        "expected_answer": "Dependencies must be declared with Depends() in the function signature. Ensure you're using Depends(your_function) not Depends(your_function()) — calling the function directly bypasses the injection system.",
        "expected_source_ids": [],
    },
    {
        "id": "d06",
        "category": "debug",
        "question": "How do I return a custom error response in FastAPI?",
        "expected_answer": "Raise an HTTPException with the desired status_code and detail: raise HTTPException(status_code=400, detail='Bad request'). For custom formats, use a custom exception handler registered with app.add_exception_handler().",
        "expected_source_ids": [],
    },
    {
        "id": "d07",
        "category": "debug",
        "question": "Why is my Pydantic model not validating nested objects?",
        "expected_answer": "Ensure nested models are also Pydantic BaseModel subclasses. FastAPI validates nested models recursively, but plain dicts won't be validated unless the type hint specifies a BaseModel.",
        "expected_source_ids": [],
    },
    {
        "id": "d08",
        "category": "debug",
        "question": "How do I test a FastAPI endpoint that uses dependencies?",
        "expected_answer": "Use app.dependency_overrides to replace dependencies in tests: app.dependency_overrides[original_dep] = mock_dep. This allows testing without real databases or external services.",
        "expected_source_ids": [],
    },
    {
        "id": "d09",
        "category": "debug",
        "question": "Why is FastAPI returning 404 for my route?",
        "expected_answer": "Common causes: route not registered (missing include_router or wrong prefix), path parameter mismatch, or trailing slash issues. Check that the route decorator path exactly matches the request URL.",
        "expected_source_ids": [],
    },
    {
        "id": "d10",
        "category": "debug",
        "question": "How do I handle request timeouts in FastAPI?",
        "expected_answer": "FastAPI itself doesn't have built-in timeout handling. Use anyio's move_on_after or fail_after context managers, or configure timeouts at the uvicorn/reverse proxy level.",
        "expected_source_ids": [],
    },
]


async def bootstrap_source_ids(n_results: int = 5) -> None:
    """
    Auto-populate expected_source_ids by running the pipeline against each
    question and saving the top result's id.

    Run once, then manually verify the ids are actually correct.
    Saves results to eval/dataset_with_ids.json.
    """
    from query.pipeline import pipeline

    print(f"Bootstrapping source ids for {len(DATASET)} questions...")
    results = []

    for i, item in enumerate(DATASET):
        print(f"[{i+1}/{len(DATASET)}] {item['id']}: {item['question'][:60]}...")
        try:
            docs = await pipeline(item["question"])
            source_ids = [doc.id for doc in docs[:n_results]]
        except Exception as e:
            print(f"  ERROR: {e}")
            source_ids = []

        results.append({**item, "expected_source_ids": source_ids})

    output = Path("eval/dataset_with_ids.json")
    output.parent.mkdir(exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output}")
    print("Review the ids manually before using for evaluation.")


if __name__ == "__main__":
    asyncio.run(bootstrap_source_ids())