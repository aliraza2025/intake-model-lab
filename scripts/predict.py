import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('HF_HOME',str(ROOT/'.model-cache'))
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from peft import PeftModel
torch.set_num_threads(2)
payload=json.load(sys.stdin);text=payload['text']
if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('Enter a request of 1–4000 characters.')
adapter=Path(os.environ.get('INTAKE_ADAPTER',ROOT/'outputs/adapter'))
if not (adapter/'adapter_config.json').exists():raise ValueError('Run scripts/train.py first to create a local adapter.')
labels=['quote','booking','handoff'];tokenizer=AutoTokenizer.from_pretrained(adapter)
base=AutoModelForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2',num_labels=3,id2label=dict(enumerate(labels)),label2id={label:i for i,label in enumerate(labels)})
model=PeftModel.from_pretrained(base,adapter);model.eval()
with torch.no_grad():values=model(**tokenizer(text,return_tensors='pt',truncation=True,max_length=64)).logits.softmax(-1)[0]
index=int(values.argmax());report=json.loads((ROOT/'examples/training-report.json').read_text())
print(json.dumps({'label':labels[index],'score':float(values[index]),'report':report}))
