from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
from tqdm import tqdm
import json, argparse, torch, pathlib

def eval_model(args : argparse.Namespace) -> None:
    disable_torch_init()

    if args.model_base:
        model_path = pathlib.Path(args.model_path).expanduser()
    else:
        model_path = args.model_path
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = json.load(open(pathlib.Path(args.question_file).expanduser(), "r"))
    answers_file = pathlib.Path(args.answers_file).expanduser()
    answers_file.parent.mkdir(parents = True, exist_ok = True)
    answers = json.load(open(answers_file, "r")) if answers_file.is_file() else {}
    for line in tqdm(questions):
        idx = line["id"] # unique identifier
        image_file = line["image"]
        qs = "what visual characteristics separate prominent terrain classes in this image?"
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv_templates[args.conv_mode].system = ""
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt().strip()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(pathlib.Path(args.image_folder).expanduser() / image_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        if idx not in answers:
            answers[idx] = {
                "question": cur_prompt,
                "ground-truth": line["conversations"][1]["value"]    
            }

        if args.model_base:
            answers[idx]["finetune-answer"] = outputs
        else:
            answers[idx]["base-answer"] = outputs
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="playground/")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--question-file", type=str, required = True)
    parser.add_argument("--answers-file", type=str, required = True)
    args = parser.parse_args()

    eval_model(args)
