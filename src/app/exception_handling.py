from typing import Callable
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse


def _exception_handler_wrapper(callback: Callable = None):
    """
    Decorator to wrap exception handler with a callback function that is triggered before exception handler.
    """
    def decorator(exc_handler: Callable):
        def wrapper(*exc_handler_args, **exc_handler_kwargs):
            if callback:
                callback()

            return exc_handler(*exc_handler_args, **exc_handler_kwargs)
        return wrapper
    return decorator

def handle_exceptions(app: FastAPI, on_error: Callable = None):
    """
    Register exception handlers for the FastAPI app.
    
    Args:
        app: FastAPI app instance
        on_error: Callback function to be called when any exception is raised
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )