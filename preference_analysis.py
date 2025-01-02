from openai import OpenAI
from tqdm import tqdm
import numpy as np

import argparse, json, re, pathlib

client = OpenAI(api_key = "<API KEY HERE>")

def main(args : argparse.Namespace) -> None:
    with open(pathlib.Path(args.response_file).expanduser(), 'r') as f:
        answer_comparisons = json.load(f)

    sys_message = """You will act as the judge of natural language responses from two students in planetary science. You will be presented with a 'QUESTION', the desired 'GROUND-TRUTH' answer, and the responses from two students to be evaluated. Your job is to score each response and decide which of the two answers is most similar to the 'GROUND-TRUTH' response based on the response's content, i.e., disregard whether a response simply has similar structure to the 'GROUND-TRUTH' answer."""
    user_message = """OVERVIEW: Two graduate students in planetary science are presented with an open ended 'QUESTION' which evaluates each student's ability to compare and contrast the characteristics of at least two different terrain types in a camera image of Mars' landscape. Your job is to score each student's response with a numeric grade and determine which student's response is most similar to the 'GROUND-TRUTH' response.

RULES:
1) In your response, you will return three scores, i.e, a PREFERENCE score, a numeric score for the response from 'STUDENT 1' and a numeric score for the response from 'STUDENT 2'.
2) The PREFERENCE score should be as a single number corresponding to the student with the answer most similar in content and meaning to the 'GROUND-TRUTH' answer, e.g., you should return a 0 if 'STUDENT 0' is preferable to 'STUDENT 1', and conversely, you should return a 1 if 'STUDENT 1' is more preferable to 'STUDENT 0'.
3) The score you give to 'STUDENT 0' and 'STUDENT 1' should be an integer number between 0 (worst) and 100 (exemplar) reflecting the degree of similarity between the student's response with the 'GROUND-TRUTH' answer. For example, a very similar response to the ground-truth answer should receive a high score.
4) If you are unsure which student's response is preferable or the exact numeric grade to assign either student, please use your best judgement.
5) Give your answer in the following format:
PREFERENCE: <YOUR PREFERENCE SCORE HERE>
STUDENT 0 SCORE: <YOUR SCORE FOR STUDENT 0 HERE>
STUDENT 1 SCORE: <YOUR SCORE FOR STUDENT 1 HERE>
6) Finally, strictly follow the format above and do not provide an explanation to justify your evaluation.

CONTENT:
Question: [QUESTION]
'GROUND-TRUTH' answer: [GROUND-TRUTH]
'STUDENT 0' response: [STUDENT 0]
'STUDENT 1' response: [STUDENT 1]"""    

    preference_counts = [0, 0]
    student0_scores = []
    student1_scores = []
    for _, comparison in tqdm(answer_comparisons.items()):
        prompt = user_message.replace("[QUESTION]", comparison["question"]).replace("[GROUND-TRUTH]", comparison["ground-truth"])
        prompt = prompt.replace("[STUDENT 0]", comparison[args.gpt_model + "-answer"] if args.compare_to_gpt else comparison["base-answer"]).replace("[STUDENT 1]", comparison["finetune-answer"])
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            evaluation = completion.choices[0].message.content.lower()
            preference = int(evaluation.split("preference: ")[-1].split("student 0 score:")[0].strip())
            if preference < 2:
                preference_counts[preference] += 1
                student0_scores.append(int(evaluation.split("student 0 score:")[-1].split("student 1 score:")[0].strip()))
                student1_scores.append(int(evaluation.split("student 1 score:")[-1].strip()))
        except Exception:
            print(evaluation)
    print(preference_counts)
    print(np.mean(student0_scores))
    print(np.mean(student1_scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-file", type=str, required = True, help = "path to evaluation file.")
    parser.add_argument("--compare-to-gpt", action = "store_true", help = "true if comparing against gpt model else compare against base model.")
    parser.add_argument("--gpt-model", type = str, default = "gpt-4o")
    args = parser.parse_args()
    
    main(args)
    