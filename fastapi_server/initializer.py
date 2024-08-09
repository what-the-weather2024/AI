import inject

from fastapi_server.services.weather_classification_service import (
    WeatherClassificationService,
)
from model.model import WeatherClassificationModel


class Initializer:
    def __init__(self):
        self._configure()

    def _configure(self):
        inject.configure(self._bind)

    def _bind(self, binder):
        weather_classification_model = WeatherClassificationModel()
        weather_classification_service = WeatherClassificationService(
            weather_classification_model=weather_classification_model,
        )

        binder.bind(WeatherClassificationService, weather_classification_service)
