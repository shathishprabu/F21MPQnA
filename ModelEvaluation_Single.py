import streamlit as st
import string, re

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

device="cuda"
MODELPATH1="data/benchmarks/bestmodels/bert-large-uncased-whole-word-masking-squad2/"
TOKENIZERPATH1="data/benchmarks/bestmodels/bert-large-uncased-whole-word-masking-squad2/"
MODELPATH2="data/benchmarks/bestmodels/distilbert-base-uncased-distilled-squad/"
TOKENIZERPATH2="data/benchmarks/bestmodels/distilbert-base-uncased-distilled-squad/"
MODELPATH3="data/benchmarks/bestmodels/roberta-base-squad2/"
TOKENIZERPATH3="data/benchmarks/bestmodels/roberta-base-squad2/"


@st.cache(allow_output_mutation=True)
def init(modelPath,tokenizerPath):    
    tokenizer = AutoTokenizer.from_pretrained(modelPath)    
    model = AutoModelForQuestionAnswering.from_pretrained(tokenizerPath)
    model.to(device)
    model.eval()
    return model,tokenizer

def getAnswer(question,context,model,tokenizer):
    query_context=[context]
    query_question=[question]
    query_encodings = tokenizer(query_context, query_question,
                                truncation=True, max_length=171,
                                padding='max_length',return_tensors='pt')



    
    
    input_ids = query_encodings['input_ids'].to(device)
    attention_mask = query_encodings['attention_mask'].to(device)   
    
    # make predictions
    outputs = model(input_ids, attention_mask=attention_mask)
    # pull preds out
    start_pred = torch.argmax(outputs['start_logits'], dim=1)
    end_pred = torch.argmax(outputs['end_logits'], dim=1)
    
    model_out=ConvertIdstoQueryAnswer(tokenizer,input_ids,start_pred,end_pred)
    return model_out[0]
    
def ConvertIdstoQueryAnswer(tokenizer,inputIds,startId,endId):
    answer=''
    results=[]
    for i in range(len(inputIds)):
        token=tokenizer.convert_ids_to_tokens(inputIds[i])      
        if endId[i] >= startId[i]:
            answer = token[startId[i]]
            for i in range(startId[i]+1, endId[i]+1):
                if token[i][0:2] == "##":               
                    answer += token[i][2:]                    
                else:
                    answer += " " + token[i]          
            results.append(answer)   
        else:
            results.append("None")  

    return results

def normalize_text(s):
    """Typically, text processing steps include removing articles and punctuation and standardizing whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


model1, tokenizer1=init(MODELPATH1,TOKENIZERPATH1)
model2, tokenizer2=init(MODELPATH2,TOKENIZERPATH2)
model3, tokenizer3=init(MODELPATH3,TOKENIZERPATH3)
modelDict = {
    "Bert":{
        "model":model1,
        "tokenizer":tokenizer2
    },
    "DistilBert":{
        "model":model2,
        "tokenizer":tokenizer2
    },
    "Roberta":{
        "model":model3,
        "tokenizer":tokenizer3
    }

}
st.header("Evaluations of Question Answering")

context= st.text_input("Enter the context here")
question= st.text_input("Enter your question here")
modelType= st.selectbox("Select Model",list(modelDict))
if st.button("Query"):    
    selectedModel=modelDict[modelType]["model"]
    selectedTokenizer=modelDict[modelType]["tokenizer"]
    answer=getAnswer(question,context,selectedModel, selectedTokenizer)
    st.success(normalize_text(answer).replace('ġ',''))
    
    

