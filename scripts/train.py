"""A reproducible, small synthetic LoRA classification experiment. No paid API."""
import os,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.model-cache'))
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from peft import LoraConfig,get_peft_model,TaskType
torch.set_num_threads(2);torch.manual_seed(7);random.seed(7)
MODEL='google/bert_uncased_L-2_H-128_A-2'
LABELS=['quote','booking','handoff']
TRAIN={
'quote':['How much does window cleaning cost?','Please send a price estimate.','What is the rate for twenty windows?','I need a quote for exterior cleaning.','Tell me the price before I decide.','Can you estimate my cleaning bill?','What would inside and outside cleaning cost?','Give me a cost breakdown.','How much for a two-story house?','Can I get pricing for this service?'],
'booking':['Please book a visit for next week.','I want to schedule an appointment.','Can we reserve a cleaning time?','Arrange a visit for Tuesday.','Please put me on the calendar.','I would like an appointment tomorrow.','Book the crew for Friday.','Schedule a window cleaning visit.','Reserve an afternoon slot.','Can you arrange our next cleaning?'],
'handoff':['I need to speak to a human.','Please connect me to a manager.','I want to make a complaint.','A person needs to handle my refund.','Get me a real staff member.','I need human help with a damaged window.','Let me talk to your supervisor.','I have a complaint about the last job.','Please transfer this to a person.','A manager should review this issue.']}
TEST={
'quote':['Could you provide an estimate for our exterior windows?','What price would I pay for this job?','I need to know the cost of the cleaning.','Send pricing for the whole property.'],
'booking':['Could you book us in for Wednesday?','I am ready to schedule the cleaning.','Is there an appointment I can reserve?','Put our next visit on the calendar.'],
'handoff':['A human should look at this complaint.','I would prefer to talk with the manager.','Please transfer me to a real person.','I want a supervisor to review the refund.']}
def records(data,prefix):return [{'id':f'{prefix}-{label}-{i+1}','text':text,'label':label} for label,texts in data.items() for i,text in enumerate(texts)]
train,test=records(TRAIN,'train'),records(TEST,'test')
assert not ({row['text'].lower() for row in train}&{row['text'].lower() for row in test})
tokenizer=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=3,id2label=dict(enumerate(LABELS)),label2id={label:i for i,label in enumerate(LABELS)})
def predict(model,rows):
 model.eval()
 with torch.no_grad():
  values=tokenizer([row['text'] for row in rows],padding=True,truncation=True,max_length=64,return_tensors='pt')
  return [LABELS[index] for index in model(**values).logits.argmax(-1).tolist()]
def accuracy(predictions):return sum(pred==row['label'] for pred,row in zip(predictions,test))/len(test)
base_predictions=predict(base,test)
model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['query','value'],modules_to_save=['classifier']))
optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.003)
started=time.perf_counter();losses=[]
for epoch in range(int(os.environ.get('TRAIN_EPOCHS','35'))):
 random.shuffle(train);model.train()
 for offset in range(0,len(train),10):
  batch=train[offset:offset+10]
  values=tokenizer([row['text'] for row in batch],padding=True,truncation=True,max_length=64,return_tensors='pt')
  labels=torch.tensor([LABELS.index(row['label']) for row in batch]);optimizer.zero_grad()
  loss=model(**values,labels=labels).loss;loss.backward();optimizer.step();losses.append(float(loss.detach()))
predictions=predict(model,test)
def rule(text):
 import re
 return 'handoff' if re.search('human|person|manager|complaint|refund',text,re.I) else 'booking' if re.search('book|schedule|appointment|visit',text,re.I) else 'quote'
baseline_predictions=[rule(row['text']) for row in test]
report={'model':MODEL,'seed':7,'dataset':'Synthetic service-intent examples; not customer research.','trainCount':len(train),'testCount':len(test),'baselineAccuracy':accuracy(baseline_predictions),'baseAccuracy':accuracy(base_predictions),'adapterAccuracy':accuracy(predictions),'trainableParameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'totalParameters':sum(p.numel() for p in model.parameters()),'trainingSeconds':round(time.perf_counter()-started,3),'initialLoss':losses[0],'finalLoss':losses[-1],'cases':[{**row,'baseline':baseline,'base':before,'adapter':after} for row,baseline,before,after in zip(test,baseline_predictions,base_predictions,predictions)],'limits':'Small hand-authored synthetic split. The unadapted classification head is newly initialized. This does not measure customer generalization or prove a benefit over rules.'}
output=ROOT/'outputs/adapter';output.mkdir(parents=True,exist_ok=True);model.save_pretrained(output);tokenizer.save_pretrained(output)
(ROOT/'examples/training-report.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'examples/train.json').write_text(json.dumps(train,indent=2)+'\n');(ROOT/'examples/test.json').write_text(json.dumps(test,indent=2)+'\n')
(ROOT/'public/benchmark.js').write_text('export default '+json.dumps(report)+';\n')
print(json.dumps(report))
