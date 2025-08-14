from fastapi import APIRouter, Depends
from ..models.schemas import QueryRequest, QueryResponse
from ..controllers.query_controller import QueryController

router = APIRouter(prefix="/api/v1", tags=["Query"])


def get_query_controller() -> QueryController:
    # INJECTED VIA DEPENDENCY INJECTION
    from ..main import query_controller

    return query_controller


@router.post("/query", response_model=QueryResponse)
async def query_books(
    request: QueryRequest, controller: QueryController = Depends(get_query_controller)
):
    """Main endpoint for querying books"""
    return await controller.process_query(request)
