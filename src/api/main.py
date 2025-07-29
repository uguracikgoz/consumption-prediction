import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

# Add parent directory to path to import from sibling modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our components
from api.routers import prediction_router, system_router


class ApiApplication:
    def __init__(self):
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="Pod Consumption Predictor API",
            description="API for predicting frontend and backend pod requirements based on business metrics",
            version="1.0.0",
            lifespan=self._lifespan
        )
        
        # Register routers
        app.include_router(system_router)
        app.include_router(prediction_router)
        
        return app
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        print("Starting up API service...")
        yield
        print("Shutting down API service...")
    
    def run(self, host: str = "0.0.0.0", port: int = None):
        port = port or int(os.environ.get("PORT", 8000))
        uvicorn.run("main:api.app", host=host, port=port, reload=True)

api = ApiApplication()
app = api.app

if __name__ == "__main__":
    api.run()
