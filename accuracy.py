# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric."""

import datasets
from sklearn.metrics import accuracy_score

import evaluate
import pandas as pd

_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.

Returns:
    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.

Examples:

    Example 1-A simple example
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
        >>> print(results)
        {'accuracy': 0.5}

    Example 2-The same as Example 1, except with `normalize` set to `False`.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
        >>> print(results)
        {'accuracy': 3.0}

    Example 3-The same as Example 1, except with `sample_weight` set.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
        >>> print(results)
        {'accuracy': 0.8778625954198473}
"""


_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""

from prompter import Prompter

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }
        
import torch   
import numpy as np   
from sklearn.metrics import (r2_score,
                             roc_auc_score) 

prompt_template="alpaca"
prompter = Prompter(prompt_template) 
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('/gpfsnyu/scratch/jd5849/GIMLET/llama_lora')    
import csv


labels_list = []
predict_list = []   
def eval_result(model, loader,label_dict,tokenizer,task_type,transformer_backbone, data_name):
    if task_type=='cla':
        model.eval()
        y_true, y_scores = [], []

        id_y=label_dict[1][0]
        id_n=label_dict[0][0]
        id_invalid=label_dict['invalid'][0]
        
       
        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            with torch.no_grad():
                # print(f'batch={batch}')
                labels=batch["labels"]  
                # print(f'labels={labels}')   labels=tensor([[-100, -100, -100, -100, -100, -100, 3782,    2]], device='cuda:1')
                if labels.shape[1]>1 and not transformer_backbone in ['kvplm']: # Yes <s>
                    assert all((labels[:,1]==tokenizer.eos_token_id) + (labels[:,1]==id_invalid))
                    labels=labels[:,-2]
                    # print(f'labels={labels}')
                    labels=tokenizer.decode(labels[0])
                # print(f'labels={labels}')
                del batch["labels"]

                if transformer_backbone in ['gimlet']: #Ours
                    batch["max_length"] = 3 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS

                elif transformer_backbone in ['galactica']:  # galactica
                    batch["max_new_tokens"] = 1 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS

                elif transformer_backbone in ['gpt3']:
                    # prompt = tokenizer.batch_decode(batch["input_ids"])[0]  # llm only supports batch_size = 1
                    # print(f'prompt: {prompt}')
                    # output = model.generate(prompt)
                    output = model.generate(input_ids=batch["input_ids"])
                    # print(f'output: {output}')  tensor([[]])
                    text = tokenizer.decode(output[0])
                    # print(f'text: {text}')
#                     text: </s></s></s></s></s></s><s> Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# Only output Yes or No!!! Estrogen receptor alpha (ER aplha) is Nuclear hormone receptor. The steroid hormones and their receptors are involved in the regulation of eukaryotic gene expression and affect cellular proliferation and differentiation in target tissues. Ligand-dependent nuclear transactivation involves either direct homodimer binding to a palindromic estrogen response element (ERE) sequence or association with other DNA-binding transcription factors, such as AP-1/c-Jun, c-Fos, ATF-2, Sp1 and Sp3, to mediate ERE-independent signaling. Is this molecule effective to this assay?

# ### Input:
# CCCC[C@H]1CN(CC2CCOCC2)C(=O)OC12CCN(C1(C)CCN(C(=O)c3c(C)ncnc3C)CC1)CC2

# ### Response:
# No</s>
                    re=prompter.get_response(text)
                    # print(f're: {re}')
                    re=tokenizer(re)
                    # print(f're: {re}')
                    re=tokenizer.decode(re['input_ids'][1:-1])
                    # label.append(labels)
                    # predict.append(re)
                    # with open('output.csv', 'w', newline='') as csvfile:
                    #     writer = csv.writer(csvfile)

                    #     # 写入标题行
                    #     writer.writerow(['Label', 'Predict'])

                    #     # 写入数据行
                    #     for l, p in zip(label, predict):
                    #         writer.writerow([l, p])
                    
                    labels_list.append(labels)
                    predict_list.append(re)
                    print(f'labels: {labels} re: {re}')
        df = pd.DataFrame({
            'label': labels_list,
            'predict': predict_list
            })
        
        df.to_csv(('/gpfsnyu/scratch/jd5849/GIMLET/test_results/{}_0shot_1_epoch.csv').format(data_name), index=False)
                    
        score = (len([1 for x, y in zip(predict_list, labels_list) if x==y]) / len(labels_list))      
                    # print(f'labels: {labels} re: {re}')
                    
                    # print(f'rrre: {rrre}')
                    # logits = output["choices"][0]["logprobs"]["top_logprobs"][0]

            #     else: #kvplm and momu
            #         logits = model(**batch)['logits']

            # index = labels != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]

            
            
            # if not isinstance(logits,dict): # for generative model
            #     assert logits[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]

            #     pred=(logits[index][:, id_y] - logits[index][:, id_n]).view([-1,1])
            #     true = labels[index].view(pred.shape)
            #     true[true == id_y] = 1
            #     true[true == id_n] = 0
            #     true[true == id_invalid] = -100
                
        #     if not isinstance(re,dict): # for generative model
        #         assert re[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]

        #         pred=(re[index][:, id_y] - re[index][:, id_n]).view([-1,1])
        #         true = labels[index].view(pred.shape)
        #         true[true == id_y] = 1
        #         true[true == id_n] = 0
        #         true[true == id_invalid] = -100

        #     else: # for contrastive model and gpt, logits is dict

        #         if transformer_backbone in ['gpt3']:
        #             positive_words = ["Yes", "yes", "YES", "Y", "y",'1']
        #             negative_words = ["No", "no", "NO", "N", "n",'0']
        #             positive_score = []
        #             for word in positive_words:
        #                 if word in re:
        #                 # if word in logits:
        #                     positive_score.append(logits[word])
        #             positive_score = np.array(positive_score).max()
        #             negative_score = []
        #             for word in negative_words:
        #                 # if word in logits:
        #                 if word in re:
        #                     negative_score.append(logits[word])
        #             negative_score = np.array(negative_score).max()
        #             pred = torch.tensor([positive_score - negative_score > 0]).unsqueeze(1)

        #         else: #Momu
        #             pred = (logits['pos'].unsqueeze(1)[index] - logits['neg'].unsqueeze(1)[index]).view([-1, 1]) #shape of logits['pos] and logits['pos] are [batch]

        #         true = labels[index].view(pred.shape)
        #         true[true == id_y] = 1
        #         true[true == id_n] = 0
        #         true[true == id_invalid] = -100
        #         assert torch.sum(true == id_invalid) == 0 # For contrastive model, invalid label is previously replaced by id_invalid(-100). Replace it here. Not necessary, because only valid label are selected

        #     y_true.append(true)
        #     y_scores.append(pred)

        # y_true = torch.cat(y_true, dim=0).cpu().numpy()
        # y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

        # roc_list = []
        # for i in range(y_true.shape[1]):
        #     # AUC is only defined when there is at least one positive data.
        #     if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
        #         is_valid = y_true[:, i]  >= 0
        #         roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
        #     else:
        #         print('{} is invalid'.format(i))

        # if len(roc_list) < y_true.shape[1]:
        #     print(len(roc_list))
        #     print('Some target is missing!')
        #     print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))

        # if len(roc_list)==0:
        #     return {'score':0},0, y_true, y_scores
        # else:
        #     return {'score':sum(roc_list) / len(roc_list)}, 0, y_true, y_scores
        return {'score': score}, 0, 0, 0
    else: # for regression

        model.eval()
        y_true, y_scores = [], []

        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            with torch.no_grad():
                labels=batch["labels"]
                del batch["labels"]
                if "decoder_attention_mask" in batch:
                    del batch["decoder_attention_mask"]

                if transformer_backbone in ['gimlet']: #Ours
                    batch["max_length"] = labels.shape[1]+1 # additional <pad> in the begining
                    ids = model.generate(
                        **batch,
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    pred = []
                    for i in range(ids.shape[0]):
                        pred.append(tokenizer.decode(ids[i, :]))

                elif transformer_backbone in ['galactica']:  # galactica
                    batch["max_new_tokens"] = labels.shape[1]+1 # <PAD> CLASS <EOS>
                    ids = model.generate(
                        **batch
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    ids=ids[:,batch['input_ids'].shape[1]:]
                    pred = []
                    for i in range(ids.shape[0]):
                        pred.append(tokenizer.decode(ids[i, :]))

                else: #kvplm
                    logits = model(**batch)['logits']
                    ids=logits.argmax(2)
                    pred = []
                    for i in range(ids.shape[0]):
                        ind_valid = labels[i, :] >= 0
                        if ind_valid.shape[0] > ids.shape[1]:
                            ind_valid = ind_valid[0:(ids.shape[1])]
                        pred.append(tokenizer.decode(ids[i, ind_valid]))

            pred_number=[]
            for result in pred:
                number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
                try:
                    decoded_number=eval(number_list[0])
                except:
                    decoded_number=float(np.nan)
                pred_number.append(decoded_number)

            true=[]
            for i in range(labels.shape[0]):
                true.append(tokenizer.decode(labels[i, labels[i, :]>0]))

            true_number=[]
            for result in true:
                number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result.replace(" ",""))
                true_number.append(eval((number_list[0])) if len(number_list)>0 else float(np.nan))

            y_true+=true_number
            y_scores+=pred_number

        y_true = torch.tensor(y_true)
        y_scores = torch.tensor(y_scores)

        ind = (~y_scores.isnan())
        ratio=ind.float().mean()
        y_true=y_true[ind]
        y_scores=y_scores[ind]

        mrs=(y_true-y_scores).std()
        naive_msr=(y_true-y_true.mean()).std()

        corrcoef=np.corrcoef(y_true,y_scores)[0,1]

        try:
            r2=r2_score(y_true,y_scores)
        except:
            r2=np.nan

        # if args.plot_regression:
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(
        #         x=y_true,
        #         y=y_scores,
        #         mode='markers',
        #         marker=dict(
        #             size=25,
        #             opacity=0.5,
        #             line=dict(width=2,
        #                       ), symbol="diamond"),
        #     ))
        #     fig.update_layout(
        #         title=args.dataset.replace('_',' '),
        #     )
        #     fig.update_layout(title={'font': {'size': 50}})

        #     fig.update_layout(
        #         xaxis_title='True Value',
        #         yaxis_title='Predicted Value',
        #         width=1000,
        #         height=1000,
        #         font=dict(
        #             size=30,
        #             color="Black"
        #         )
        #     )

        #     global fig_number
        #     fig.write_image('cache/'+('{}_{}_fig{}.png'.format(args.dataset,args.model_name_or_path,fig_number)).replace('/','_'))
        #     fig_number+=1

        return {'ratio':float(ratio),'RMSE':float(mrs),'corrcoef':float(corrcoef),'R-Square':float(r2),'score':float(mrs)}, 0, y_true, y_scores
