import argparse
import json
import os
from tqdm import tqdm
from desta import DeSTA25AudioModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--data_root", type=str)

    return parser.parse_args()

def main(args):

    if args.model_id == "desta25":
        model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")

        model.to("cuda")
        model.eval()
            

    # load MMAU data
    with open(args.input_path, "r") as f:
        data = json.load(f)

    # inference
    results = []
    for item in tqdm(data):
        audio_path = os.path.join(
            args.data_root,
            item["audio_id"].replace("./", "", 1)
        )
        print(audio_path)

        system_prompt = "Focus on the audio clips and instructions. Choose one of the options without any explanation."
        
        question = f"{item['question']} "
        question += "Choose one of the following options: "

        # use "or" for last option
        for i, option in enumerate(item["choices"]):
            question += f"\"{option}\""
            if i == len(item["choices"]) - 2:
                question += " or "
            else:
                question += ", "
        
        question = question.rstrip(", ")

        messages = [
            {'role': 'system', 'content': system_prompt}, 
            {"role": "user", 
                "content": f"<|AUDIO|>\n\n{question}",
                "audios": [{"audio": audio_path}]
            }
        ]
        
        outputs = model.generate(messages=messages, max_new_tokens=512, do_sample=False)

        response = outputs.text[0]

        item["model_output"] = response
        results.append(item)


    # save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/results@{args.model_id}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)