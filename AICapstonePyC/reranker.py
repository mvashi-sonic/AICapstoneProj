from google import genai
import sys
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from pydantic import BaseModel
import torch
import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    "./triple-mountain-483601-k3-3a823d61bdb7.json"


print("info")
print(sys.executable)
print(sys.version)
print(torch.__version__)
print(transformers.__version__)
print(torch.__file__)

print("info")


tokenizer = AutoTokenizer.from_pretrained(
    "./model/finbert_relevance_mv"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./model/finbert_relevance_mv"
)
#device = torch.device(
#    "cuda" if torch.cuda.is_available() else "cpu"
#)
device = torch.device("cpu")

#########

print(torch.__version__)

x = torch.randn(2, 3)
y = torch.randn(3, 4)

print("Before matmul")

z = torch.matmul(x, y)

print("After matmul")
print(z)


print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
#########
#########

def test():
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    inputs = tokenizer(
        "Hello world",
        return_tensors="pt"
    )

    print("Running model")

    output = model(**inputs)

    print(output)
 ##############

def rerank(question, records):

    # Your reranking code

    #test()
    #question = "What supplier risks does NVIDIA face in 2025?"
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    reranked = []

    model.eval()

    for chunk in records:
        # print (f"Ref: {chunk["reference"]}")
        inputs = tokenizer(
            question,
            chunk["reference"],
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move tensors to GPU
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except RuntimeError as e:
            print(f"1RuntimeError while scoring chunk: {e}")
            continue
        except Exception as e:
            print(f"1Unexpected error while scoring chunk: {e}")
            continue
        with torch.no_grad():
            try:
                print("Running model on cpu...")
                output = model(**inputs)
                print("Success")
                print(output.logits)
                logits = output.logits

                probability = torch.softmax(
                    logits,
                    dim=1
                    )[0, 1].item()
                reranked.append({
                    "score": probability,
                    "chunk": chunk
                })
            except RuntimeError as e:
                print(f"RuntimeError while scoring chunk: {e}")
                continue

            except Exception as e:
                print(f"Unexpected error while scoring chunk: {e}")
                continue

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
            )

    top5 = reranked[:5]
    evidence = ""

    for i, r in enumerate(top5):
        chunk = r["chunk"]

        evidence += f"""

    Evidence {i + 1}

    Company: {chunk['ticker']}

    Section: {chunk['section']}

    Confidence: {r['score']:.3f}

    {chunk['reference']}

    ----------------------------------------

    """

    prompt = """Question:
    """f"""{question} Evidence:"""f""" {evidence}""""""Answer using only the evidence above. Cite the section and the company"""

    client = genai.Client(
        vertexai=True,
        project="triple-mountain-483601-k3",
        location="us-central1"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    if response.text:
        answer = response.text
    else:
        answer = "No answer generated"
    print(answer)
    return answer