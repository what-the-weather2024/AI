import logging

import inject
from fastapi import APIRouter

from fastapi_server.services.weather_classification_service import (
    WeatherClassificationService,
)

router = APIRouter(
    prefix="/wtw-ai",
    tags=["Classification"],
)

weather_classification_service = inject.instance(WeatherClassificationService)


logger = logging.getLogger("FastAPIServer")
# 로거 설정
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# CRUD 엔드포인트 구성
@router.post("/image_classification")
def create_asset(image_url: str):
    result = weather_classification_service(
        input_image_url=image_url,
    )

    logger.info(f"result: {result}")

    return result
