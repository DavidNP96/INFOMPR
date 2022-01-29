from transformers import pipeline, GPT2Tokenizer

paths = ['gpt2-36','gpt2-360','gpt2-3600','gpt2-7200','gpt2-36000']
albums = ['Fix yourself not the world','=','30','Dawn FM','Fragments','The boy named If','Ds4Ever','Between us','Sour','The Highlights']
model_path = './models/'

for path in paths:
    model_path = './models/' + path

    print("Loading in Tokenizer..", end="")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("Done")

    print(f"Loading in Model {path}..", end="")
    pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer)
    print("Done")

    for album in albums:
        string = "<reviewPrompt> " + album+ " <review>"
        print("Prompt: ", string)
        print("-"*100)
        new = pipe(string, max_length=400, min_length=300)
        with open('output.txt', 'a') as f:
            f.write('-'*100 + '\n')
            f.write(f"Model= {path}" + '\n')
            f.write(new[0]['generated_text'] + '\n')
            f.write('-'*100 + '\n')
        print("-"*100)