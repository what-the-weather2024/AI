from io import BytesIO

import open_clip
import requests
import torch
from PIL import Image


class WeatherClassificationModel:
    """날씨 분류 모델
    주어진 이미지를 날씨로 분류한다.
    Clip 모델을 활용하는데 주어진 입력 이미지와
    미리 준비한 날씨 정의 텍스트(a photo of sunny weather)의 표현을 비교하여
    해당 이미지와 가장 가까운 날씨를 반환한다.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_source: str = "laion2b_s34b_b79k",
    ) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_source,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.labels = [
            "a photo of sunny weather",
            "a photo of cloudy sky",
            "a photo of snowy sky",
            "a photo of rainny weather",
            "etc",
        ]

    def __call__(self, image_url: str):
        """추론
        clip모델을 활용해 주어진 이미지를 추론하고 사전 정의된 self.labels와 가장 가까운 값의 index를 반환한다.

        Args:
            image_url (str): 사용자의 입력 이미지. public s3 url을 전제한다.

        Returns:
            status_code (str): 상태 코드
            output (dict): label_index를 키로 가지는 dict를 가진다.
            error (str): expression of exceptional error
        """
        status_code, output, error = "200", {}, ""

        response = requests.get(url=image_url)
        image_data = BytesIO(response.content)
        image = self.preprocess(Image.open(image_data)).unsqueeze(0)
        text = self.tokenizer(self.labels)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            label_index = torch.argmax(text_probs)
            output["label_index"] = label_index.detach().numpy().item()

        return status_code, output, error


if __name__ == "__main__":
    model = WeatherClassificationModel()

    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkKMkJ8POG_HmwzOE5pqLfIq2cakh27fCrHw&s"
    res = model(image_url=image_url)
    print(res)  # ('200', {'label_index': 3}, '')
