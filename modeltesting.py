import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

device="cuda"
MODELPATH="deepset/bert-large-uncased-whole-word-masking-squad2"
TOKENIZERPATH="deepset/bert-large-uncased-whole-word-masking-squad2"

def init(modelPath,tokenizerPath):
    global model
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(modelPath)    
    model = BertForQuestionAnswering.from_pretrained(tokenizerPath)
    model.to(device)
    model.eval()

def getAnswer(question,context):
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
    
    model_out=ConvertIdstoQueryAnswer(input_ids,start_pred,end_pred)
    return model_out[0]
    
def ConvertIdstoQueryAnswer(inputIds,startId,endId):
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

if __name__== "__main__":
    query_context='Laundry opens at 7AM in the morning'
    query_question='what time does the laundry opens?'
    init(MODELPATH,TOKENIZERPATH)
    print(getAnswer(query_question,query_context))