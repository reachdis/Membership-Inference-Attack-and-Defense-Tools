import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer

class BERT4RecQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type 
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = module.SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = module.SeqPoolingLayer(pooling_type=self.eval_pooling_type)

    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                if self.eval_pooling_type == 'mask':
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return transformer_out

class BERT4Rec(basemodel.BaseRetriever):

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.mask_token = train_data.num_items
        self.query_fields = self.query_fields | set(["mask_token"])

    def _get_dataset_class(self):
        return dataset.SeqDataset

    def _get_query_encoder(self, train_data):
        return BERT4RecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=self.config['head_num'],
            hidden_size=self.config['hidden_size'], dropout=self.config['dropout'],
            activation=self.config['activation'], layer_norm_eps=self.config['layer_norm_eps'],
            n_layer=self.config['layer_num'],
            training_pooling_type=self.config['pooling_type'],
            item_encoder=self.item_encoder,
            bidirectional=True,
        )

    def _get_item_encoder(self, train_data):
        # id num_items is used for mask token
        return torch.nn.Embedding(train_data.num_items+1, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""SoftmaxLoss is used as the loss function."""
        return loss_func.SoftmaxLoss()

    def _get_sampler(self, train_data):
        return None

    def _reconstruct_train_data(self, batch):
        # print(batch['in_'+self.fiid].shape)
        # print(batch[self.fiid].shape)
        item_seq = batch['in_'+self.fiid]
        padding_mask = item_seq == 0
        rand_prob = torch.rand_like(item_seq, dtype=torch.float)
        rand_prob.masked_fill_(padding_mask, 1.0)
        masked_mask = rand_prob < self.config['mask_ratio']
        # print(rand_prob.shape)
        # print(self.config['mask_ratio'])
        # print(masked_mask.shape)
        masked_token = item_seq[masked_mask]
        # print('---')
        # print(item_seq.shape)
        # print(masked_mask.shape)
        # print(masked_token.shape)
        # print('---')

        item_seq[masked_mask] = self.mask_token
        batch['in_'+self.fiid] = item_seq
        batch[self.fiid] = masked_token     # N
        batch['mask_token'] = masked_mask

        # print(batch['in_'+self.fiid].shape)
        # print(batch[self.fiid].shape)
        # print(batch['mask_token'].shape)
        # print('---------')
        return batch

    def training_step(self, batch):
        if not self.config['steal']:
            batch = self._reconstruct_train_data(batch)
        return super().training_step(batch)
