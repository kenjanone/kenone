from fastapi import APIRouter
from database import get_connection

router = APIRouter()

@router.get("/health")
def health_check():
    try:
        conn = get_connection()
        conn.close()
        return {"status": "healthy", "version": "1.0.0", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
