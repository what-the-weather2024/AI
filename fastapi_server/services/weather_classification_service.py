from model.model import WeatherClassificationModel


class WeatherClassificationService:
    def __init__(
        self, weather_classificaiton_model: WeatherClassificationModel
    ) -> None:
        self.model = weather_classificaiton_model
        self.label2weather = {
            0: "맑음",
            1: "흐림",
            2: "눈",
            3: "비",
            4: "기타",
        }

    def __call__(self, input_image_url: str):
        """날씨분류 수행
        이미지를 입력받아 날씨 text로 분류한다.

        Args:
            input_image_url (str): 입력 public 이미지 url

        Returns:
            weather (dict): weather, status_code, error를 포함한 결과물
        """
        weather = ""
        status_code, output, error = self.model(image_url=input_image_url)
        if status_code != "200":
            weather = "(Model Error)"
        else:
            label = output["label_index"]
            weather = self.label2weather[label]

        return {
            "status_code": status_code,
            "weather": weather,
            "error": error,
        }
