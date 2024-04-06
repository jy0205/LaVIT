import argparse
import base64
from mimetypes import guess_type


class Prompt:

    def __init__(self, prompt_file, k) -> None:
        self.k = k
        with open(prompt_file) as p:
            self.prompt_template = "".join(p.readlines()).strip()

    def prepare_message(self, caption, num_sample_images):
        num_sample_images = min(self.k, num_sample_images)
        if self.k > 0:
            prompt = self.prompt_template.format(caption=caption, num=num_sample_images)
        else:
            prompt = self.prompt_template.format(caption=caption)
        return prompt

    def encode_image_as_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default=False, help="Prompt file")
    parser.add_argument(
        "--k", default=0, help="Number of retrieved images included in the prompt"
    )

    args = parser.parse_args()

    p = Prompt(args.prompt_file, args.k)
    print(p.prepare_message(caption="Generate a dog", num_sample_images=2))


if __name__ == "__main__":
    main()
