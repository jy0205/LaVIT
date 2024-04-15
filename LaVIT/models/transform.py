import re
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class LaVITImageProcessor:
    def __init__(self, image_size=224):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        transform_list = [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        self.transform = transforms.Compose(transform_list)

    def __call__(self, item):
        return self.transform(item)


class LaVITQuestionProcessor:
    """
    Adapting from BLIP2, for processing the question in VQA tasks
    """
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question