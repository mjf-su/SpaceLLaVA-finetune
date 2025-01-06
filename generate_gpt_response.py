from typing import Union, List, Dict
from io import BytesIO
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

import pathlib, base64, argparse, json
import numpy as np
client = OpenAI(api_key = "<API KEY HERE>")

# Function to encode the image
def encode_image(image: Union[pathlib.Path, np.ndarray]) -> str:
    if isinstance(image, pathlib.Path):
        with open(image, "rb") as file: # passed image filepath
            return base64.b64encode(file.read()).decode('utf-8')
    
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image) # conver to PIL image
        im_file = BytesIO()
        image.save(im_file, format="JPEG") # save image to file object, not to disk

        return base64.b64encode(im_file.getvalue()).decode('utf-8') # direct image processing
    
    else:
        raise TypeError("Provide the image's filepath or the image directly as a numpy array.")

def main(args : argparse.Namespace) -> None:
    questions = json.load(open(pathlib.Path(args.question_file).expanduser(), "r"))
    answers_file = pathlib.Path(args.answers_file).expanduser()
    answers_file.parent.mkdir(parents = True, exist_ok = True)
    answers = json.load(open(answers_file, "r")) if answers_file.is_file() else {}
    for line in tqdm(questions):
        idx = line["id"] # unique identifier
        assert idx in answers # must generate sota zero-shot answers following
        if args.evaluation_task and args.evaluation_task not in idx: # target a particular task if "evaluation-task" passed
            continue 

        image_file = pathlib.Path(args.image_folder).expanduser() / line["image"]
        qs = answers[idx]["question"]

        base64_images = [encode_image(image_file)]
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": qs + " Limit your response to three to five sentences.", # define prompt
                        }
                    ]
                }
        ]
        image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]
        messages[0]["content"] += image_content # define image(s)

        response = client.chat.completions.create(
            model = args.model,
            messages = messages,
            max_tokens = 512,
            temperature = 0.,
        )
        try:
            output = response.choices[0].message.content
        except Exception as e:
            output = e
        answers[idx][args.model + "-answer"] = output
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type = str, default = "/SpaceLLaVA-finetune/playground/data/ai4mars/vqa/terrain_comparison/eval_dataset.json")
    parser.add_argument("--answers-file", type = str, default = "/SpaceLLaVA-finetune/playground/data/ai4mars/vqa/terrain_comparison/answer.json")
    parser.add_argument("--evaluation-task", type = str, default = None, help = "the task within the question file, e.g., eval_dataset.json file, you wish to process.")
    parser.add_argument("--image-folder", type = str, default = "playground/")
    parser.add_argument("--model", type = str, default = "gpt-4o")
    args = parser.parse_args()
    
    main(args)