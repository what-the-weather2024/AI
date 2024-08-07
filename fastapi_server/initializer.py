import inject

from model.model import WeatherClassificationModel


class Initializer:
    def __init__(self):
        self._configure()

    def _configure(self):
        inject.configure(self._bind)

    def _bind(self, binder):
        weather_classification_model = WeatherClassificationModel()

        binder.bind(WeatherClassificationModel, weather_classification_model)
