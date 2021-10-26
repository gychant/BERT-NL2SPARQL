"""Model Definition"""

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import CrossEntropyLoss, KLDivLoss, BCELoss
from data_loader import DataLoader
from bert import BertPreTrainedModel, BertModel
import random
# (batch_tokens, batch_sub_heads, batch_sub_tails, batch_sub_weights), \
#                 (batch_sub_and_tokens, batch_obj_heads, batch_obj_tails, batch_obj_weights)

class BertNTR(BertPreTrainedModel):
    def __init__(self, config, num_class=49):
        super(BertNTR, self).__init__(config)
        print("__init__ is called!!!")
        self.bert = BertModel(config)
        self.sub_outputs = nn.Linear(config.hidden_size, 2)
        self.obj_head_outputs = nn.Linear(config.hidden_size, num_class)
        self.obj_tail_outputs = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, batch_sub_task, batch_obj_task, teacher_logits=None, teacher_only=False, get_logits=False):
        # subject task
        batch_tokens, batch_sub_heads, batch_sub_tails, batch_sub_weights = batch_sub_task
        batch_sub_masks = batch_tokens.gt(0)

        print("batch_tokens:", batch_tokens)
        print("batch_sub_masks:", batch_sub_masks)
        print("batch_sub_heads:", batch_sub_heads)
        print("batch_sub_heads size:", batch_sub_heads.size())
        print("batch_sub_tails:", batch_sub_tails)
        print("batch_sub_weights:", batch_sub_weights)

        sub_sequence_output, _ = self.bert(batch_tokens, None, batch_sub_masks, output_all_encoded_layers=False)
        sub_logits = self.sub_outputs(sub_sequence_output)

        sub_head_logits, sub_tail_logits = sub_logits.split(1, dim=-1) # (batch_size, seq_len, 2)
        sub_head_logits = sub_head_logits.squeeze(-1) # (batch_size, seq_len)
        sub_tail_logits = sub_tail_logits.squeeze(-1) # (batch_size, seq_len)

        # object task
        batch_query_tokens, batch_token_types, batch_obj_heads, batch_obj_tails, batch_obj_weights = batch_obj_task
        batch_obj_masks = batch_query_tokens.gt(0)

        print("batch_token_types:", batch_token_types)
        print("batch_obj_heads:", batch_obj_heads[0, 15, :])
        print("batch_obj_heads size:", batch_obj_heads.size())
        print("batch_obj_masks:", batch_obj_masks)
        print(hello)

        obj_sequence_output, _ = self.bert(batch_query_tokens, batch_token_types, batch_obj_masks, output_all_encoded_layers=False)
        obj_head_logits = self.obj_head_outputs(obj_sequence_output) # (batch_size, seq_len, num_class)
        obj_tail_logits = self.obj_tail_outputs(obj_sequence_output).squeeze(-1) # (batch_size, seq_len)
        if get_logits:
            return sub_head_logits, sub_tail_logits, obj_head_logits, obj_tail_logits
        if teacher_logits and teacher_only:
            teacher_sub_head_logits, teacher_sub_tail_logits, teacher_obj_head_logits, teacher_obj_tail_logits = teacher_logits
            loss_sub_head_kl = binary_cross_entropy_with_logits(sub_head_logits, teacher_sub_head_logits.sigmoid(), reduction='mean')
            loss_sub_tail_kl = binary_cross_entropy_with_logits(sub_tail_logits, teacher_sub_tail_logits.sigmoid(), reduction='mean')
            loss_obj_head_kl = binary_cross_entropy_with_logits(obj_head_logits, teacher_obj_head_logits.sigmoid(), reduction='none').sum(-1).mean()
            loss_obj_tail_kl = binary_cross_entropy_with_logits(obj_tail_logits, teacher_obj_tail_logits.sigmoid(), reduction='mean')
            loss = loss_sub_head_kl + loss_sub_tail_kl + loss_obj_head_kl + loss_obj_tail_kl
        else:
            loss_sub_head = binary_cross_entropy_with_logits(sub_head_logits, batch_sub_heads, reduction='mean')
            loss_sub_tail = binary_cross_entropy_with_logits(sub_tail_logits, batch_sub_tails, reduction='mean')
            loss_obj_head = binary_cross_entropy_with_logits(obj_head_logits, batch_obj_heads, reduction='none').sum(-1).mean()

            #extract the col for #cls, give more penatly (select the object columns)
            loss_obj_head_v2 = binary_cross_entropy_with_logits(obj_head_logits, batch_obj_heads, reduction='none')[:,-2, :].sum(-1).mean()

            loss_obj_tail = binary_cross_entropy_with_logits(obj_tail_logits, batch_obj_tails, reduction='mean')
            loss = loss_sub_head + loss_sub_tail + loss_obj_head + loss_obj_tail #+ loss_obj_head_v2 * 5
            if teacher_logits:
                teacher_sub_head_logits, teacher_sub_tail_logits, teacher_obj_head_logits, teacher_obj_tail_logits = teacher_logits
                loss_sub_head_kl = binary_cross_entropy_with_logits(sub_head_logits, teacher_sub_head_logits.sigmoid(), reduction='mean')
                loss_sub_tail_kl = binary_cross_entropy_with_logits(sub_tail_logits, teacher_sub_tail_logits.sigmoid(), reduction='mean')
                loss_obj_head_kl = binary_cross_entropy_with_logits(obj_head_logits, teacher_obj_head_logits.sigmoid(), reduction='none').sum(-1).mean()
                loss_obj_tail_kl = binary_cross_entropy_with_logits(obj_tail_logits, teacher_obj_tail_logits.sigmoid(), reduction='mean')
                loss += (loss_sub_head_kl + loss_sub_tail_kl + loss_obj_head_kl + loss_obj_tail_kl)
                loss = loss / 2.
        return loss

    def extract(self, batch_tokens, sub_threshold=0.4, obj_threshold=0.4):
        res = []
        assert batch_tokens.size()[0] == 1
        batch_sub_masks = batch_tokens.gt(0)
        sub_sequence_output, _ = self.bert(batch_tokens, None, batch_sub_masks, output_all_encoded_layers=False) # (1, seq_len, hidden_size)
        sub_logits = self.sub_outputs(sub_sequence_output) # (1, seq_len, 2)
        sub_head_logits, sub_tail_logits = sub_logits.split(1, dim=-1) # 2 * (1, seq_len, 1)
        sub_head_logits = sub_head_logits.squeeze(-1).sigmoid() # (1, seq_len)
        sub_tail_logits = sub_tail_logits.squeeze(-1).sigmoid() # (1, seq_len)
        # print('sub_head_logits: %r'%sub_head_logits)

        ## 1. multi
        head_indexes = torch.cat(((sub_head_logits > sub_threshold).nonzero(),
                                  torch.tensor([[0, batch_tokens.size()[1]-1]]).to('cuda')), dim=0)

        ## 2. the maximum
        # head_indexes = torch.cat((torch.tensor([[0, int(sub_head_logits.argmax(dim=-1)[0])]]).to('cuda'),
        #                           torch.tensor([[0, batch_tokens.size()[1] - 1]]).to('cuda')), dim=0)

        # print('head_indexes: %r'%head_indexes)
        for i in range(head_indexes.size()[0]-1):
        # for i in range(head_indexes.size()[0] - head_indexes.size()[0] + 1):
            head_pos = head_indexes[i][1]
            dead_pos = head_indexes[i+1][1]
            mask = torch.zeros_like(sub_tail_logits)
            mask[0, head_pos:dead_pos] = 1
            # print('mask_sub:%r'%mask)
            masked_sub_tail_logits = mask * sub_tail_logits
            # print('masked_sub_tail_logits: %r'%masked_sub_tail_logits)
            tail_pos = masked_sub_tail_logits.argmax(-1)[0]
            # print('sub_span: %r, %r'%(head_pos, tail_pos))
            # concat query with tokens
            query = batch_tokens[:, head_pos:tail_pos+1] #subject
            batch_query_tokens = torch.cat((batch_tokens[:, :1], query, batch_tokens[:, -1:],batch_tokens[:, 1:]), dim=-1)
            batch_token_types = torch.zeros_like(batch_query_tokens)
            batch_token_types[:, (tail_pos-head_pos+2):] = 1
            batch_obj_masks = batch_query_tokens.gt(0)
            obj_sequence_output, _ = self.bert(batch_query_tokens, batch_token_types, batch_obj_masks, output_all_encoded_layers=False)
            obj_head_logits = self.obj_head_outputs(obj_sequence_output).sigmoid() # (1, seq_len, num_class)
            obj_tail_logits = self.obj_tail_outputs(obj_sequence_output).squeeze(-1).sigmoid()
            # print('obj_head_logits: %r'%obj_head_logits)
            # print(obj_head_logits.max(dim=-1)[0])
            # print((obj_head_logits.max(dim=-1)[0]).argmax(dim=1)[0])

            ## 1. multi
            obj_head_indexes = torch.cat(((obj_head_logits.max(dim=-1)[0] > obj_threshold).nonzero(),
                                          torch.tensor([[0, batch_query_tokens.size()[1]-1]]).to('cuda')), dim=0)

            ## 2. maximum
            # obj_head_indexes = torch.cat((torch.tensor([[0, int((obj_head_logits.max(dim=-1)[0]).argmax(dim=1)[0])]]).to('cuda'),
            #                               torch.tensor([[0, batch_query_tokens.size()[1] - 1]]).to('cuda')), dim=0)


            # print('obj_head_indexes: %r'%obj_head_indexes)
            for j in range(obj_head_indexes.size()[0]-1):
            # if obj_head_indexes.size()[0] >=2:
            #     for j in range(obj_head_indexes.size()[0] - obj_head_indexes.size()[0] + 1):
                obj_head_pos = obj_head_indexes[j][1]
                len_query = query.size()[1]+1
                if obj_head_pos > len_query:
                    obj_dead_pos = obj_head_indexes[j+1][1]
                    obj_mask = torch.zeros_like(obj_tail_logits)
                    obj_mask[0, obj_head_pos:obj_dead_pos] = 1
                    # print('mask_obj:%r'%obj_mask)
                    masked_obj_tail_logits = obj_mask * obj_tail_logits
                    # print('masked_obj_tail_logits:%r'%masked_obj_tail_logits)
                    obj_tail_pos = masked_obj_tail_logits.argmax(-1)[0]
                    ## 1. multi
                    pre_indexes = (obj_head_logits[0, obj_head_pos, :] > obj_threshold).nonzero().flatten()

                    # pre_indexes = obj_head_logits[0, obj_head_pos, :].argsort(dim=-1, descending=True).flatten()[:5]
                    # print(pre_indexes[:10])

                    ## 2. maximum
                    # pre_indexes = obj_head_logits[0, obj_head_pos, :].argmax(dim=-1,keepdim=True).flatten()

                    # print(pre_indexes)

                    for pre_index in pre_indexes:
                        # print(pre_index, float(obj_head_logits[0, obj_head_pos, :][pre_index]))
                        sub_span = (int(head_pos), int(tail_pos)+1)
                        obj_span = (int(obj_head_pos-len_query), int(obj_tail_pos+1-len_query))
                        # 1.
                        res.append((sub_span, int(pre_index), obj_span))

                        # 2. todo for simple question, add the probability terms
                        # res.append((sub_span, int(pre_index), obj_span, float(obj_head_logits[0, obj_head_pos, :][pre_index])))

                        #             break   #for sq only
            #     break  #for sq only
            # break  #for sq only
        # while len(res) > 1:
        #     index = random.randint(0, len(res)-1)
        #     res.pop(1-index)

        return res

    def dynamic_extract(self, batch_tokens, sub_threshold=0.5, obj_threshold=0.5):
        res = []
        assert batch_tokens.size()[0] == 1
        batch_sub_masks = batch_tokens.gt(0)
        sub_sequence_output, _ = self.bert(batch_tokens, None, batch_sub_masks, output_all_encoded_layers=False) # (1, seq_len, hidden_size)
        sub_logits = self.sub_outputs(sub_sequence_output) # (1, seq_len, 2)
        sub_head_logits, sub_tail_logits = sub_logits.split(1, dim=-1) # 2 * (1, seq_len, 1)
        sub_head_logits = sub_head_logits.squeeze(-1).sigmoid() # (1, seq_len)
        sub_tail_logits = sub_tail_logits.squeeze(-1).sigmoid() # (1, seq_len)
        # print('sub_head_logits: %r'%sub_head_logits)
        head_indexes = torch.cat(((sub_head_logits > sub_threshold).nonzero(), torch.tensor([[0, batch_tokens.size()[1]-1]]).to('cuda')), dim=0)
        if head_indexes.size()[0] < 2:
            sub_threshold_tmp = sub_threshold
            while sub_threshold_tmp > 0.4 and head_indexes.size()[0] < 2:
                sub_threshold_tmp -= 0.02
                head_indexes = torch.cat(((sub_head_logits > sub_threshold_tmp).nonzero(), torch.tensor([[0, batch_tokens.size()[1]-1]]).to('cuda')), dim=0)
        # print('head_indexes: %r'%head_indexes)
        for i in range(head_indexes.size()[0]-1):
            head_pos = head_indexes[i][1]
            dead_pos = head_indexes[i+1][1]
            mask = torch.zeros_like(sub_tail_logits)
            mask[0, head_pos:dead_pos] = 1
            # print('mask_sub:%r'%mask)
            masked_sub_tail_logits = mask * sub_tail_logits
            # print('masked_sub_tail_logits: %r'%masked_sub_tail_logits)
            tail_pos = masked_sub_tail_logits.argmax(-1)[0]
            # print('sub_span: %r, %r'%(head_pos, tail_pos))
            # concat query with tokens
            query = batch_tokens[:, head_pos:tail_pos+1] #subject
            batch_query_tokens = torch.cat((batch_tokens[:, :1], query, batch_tokens[:, -1:],batch_tokens[:, 1:]), dim=-1)
            batch_token_types = torch.zeros_like(batch_query_tokens)
            batch_token_types[:, (tail_pos-head_pos+2):] = 1
            batch_obj_masks = batch_query_tokens.gt(0)
            obj_sequence_output, _ = self.bert(batch_query_tokens, batch_token_types, batch_obj_masks, output_all_encoded_layers=False)
            obj_head_logits = self.obj_head_outputs(obj_sequence_output).sigmoid() # (1, seq_len, num_class)
            obj_tail_logits = self.obj_tail_outputs(obj_sequence_output).squeeze(-1).sigmoid()
            obj_head_indexes = torch.cat(((obj_head_logits.max(dim=-1)[0] > obj_threshold).nonzero(), torch.tensor([[0, batch_query_tokens.size()[1]-1]]).to('cuda')), dim=0)
            obj_threshold_tmp = obj_threshold
            if obj_head_indexes.size()[0] < 2:
                while obj_threshold_tmp > 0.4 and obj_head_indexes.size()[0] < 2:
                    obj_threshold_tmp -= 0.02
                    obj_head_indexes = torch.cat(((obj_head_logits.max(dim=-1)[0] > obj_threshold_tmp).nonzero(), torch.tensor([[0, batch_query_tokens.size()[1]-1]]).to('cuda')), dim=0)
            # print('obj_head_indexes: %r'%obj_head_indexes)
            for j in range(obj_head_indexes.size()[0]-1):
                obj_head_pos = obj_head_indexes[j][1]
                len_query = query.size()[1]+1
                if obj_head_pos > len_query:
                    obj_dead_pos = obj_head_indexes[j+1][1]
                    obj_mask = torch.zeros_like(obj_tail_logits)
                    obj_mask[0, obj_head_pos:obj_dead_pos] = 1
                    # print('mask_obj:%r'%obj_mask)
                    masked_obj_tail_logits = obj_mask * obj_tail_logits
                    # print('masked_obj_tail_logits:%r'%masked_obj_tail_logits)                    
                    obj_tail_pos = masked_obj_tail_logits.argmax(-1)[0]
                    pre_indexes = (obj_head_logits[0, obj_head_pos, :] > obj_threshold_tmp).nonzero().flatten()
                    for pre_index in pre_indexes:
                        sub_span = (int(head_pos), int(tail_pos)+1)
                        obj_span = (int(obj_head_pos-len_query), int(obj_tail_pos+1-len_query))
                        res.append((sub_span, int(pre_index), obj_span))
        return res

    def sigmoid_extract(self, batch_tokens, sub_threshold=0.5, obj_threshold=0.5):
        res = []
        assert batch_tokens.size()[0] == 1
        batch_sub_masks = batch_tokens.gt(0)
        sub_sequence_output, _ = self.bert(batch_tokens, None, batch_sub_masks, output_all_encoded_layers=False) # (1, seq_len, hidden_size)
        sub_logits = self.sub_outputs(sub_sequence_output) # (1, seq_len, 2)
        sub_head_logits, sub_tail_logits = sub_logits.split(1, dim=-1) # 2 * (1, seq_len, 1)
        sub_head_logits = sub_head_logits.squeeze(-1).sigmoid() # (1, seq_len)
        sub_tail_logits = sub_tail_logits.squeeze(-1).sigmoid() # (1, seq_len)
        head_indexes = (sub_head_logits > sub_threshold).nonzero()
        tail_indexes = (sub_tail_logits > sub_threshold).nonzero()
        for i in range(head_indexes.size()[0]):
            head_pos = head_indexes[i][1]
            tail_pos = None
            for tail_index in tail_indexes:
                if tail_index[1] >= head_pos:
                    tail_pos = tail_index[1]
                    break
            if tail_pos:
                # concat query with tokens
                query = batch_tokens[:, head_pos:tail_pos+1] #subject
                batch_query_tokens = torch.cat((batch_tokens[:, :1], query, batch_tokens[:, -1:],batch_tokens[:, 1:]), dim=-1)
                batch_token_types = torch.zeros_like(batch_query_tokens)
                batch_token_types[:, (tail_pos-head_pos+2):] = 1
                batch_obj_masks = batch_query_tokens.gt(0)
                obj_sequence_output, _ = self.bert(batch_query_tokens, batch_token_types, batch_obj_masks, output_all_encoded_layers=False)
                obj_head_logits = self.obj_head_outputs(obj_sequence_output).sigmoid() # (1, seq_len, num_class)
                obj_tail_logits = self.obj_tail_outputs(obj_sequence_output).squeeze(-1).sigmoid()
                obj_head_indexes = (obj_head_logits.max(dim=-1)[0] > obj_threshold).nonzero()
                obj_tail_indexes = (obj_tail_logits > obj_threshold).nonzero()
                for j in range(obj_head_indexes.size()[0]):
                    obj_head_pos = obj_head_indexes[j][1]
                    len_query = query.size()[1]+1
                    if obj_head_pos > len_query:
                        obj_tail_pos = None
                        for obj_tail_index in obj_tail_indexes:
                            if obj_tail_index[1] >= obj_head_pos:
                                obj_tail_pos = obj_tail_index[1]
                                break
                        if obj_tail_pos:
                            pre_indexes = (obj_head_logits[0, obj_head_pos, :] > obj_threshold).nonzero().flatten()
                            for pre_index in pre_indexes:
                                sub_span = (int(head_pos), int(tail_pos)+1)
                                obj_span = (int(obj_head_pos-len_query), int(obj_tail_pos+1-len_query))
                                res.append((sub_span, int(pre_index), obj_span))
        return res