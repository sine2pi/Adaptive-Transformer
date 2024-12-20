@staticmethod
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id) -> torch.Tensor:
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def forward(self, input_features, labels=None, dec_input_ids=None):
    if labels is not None:
        if dec_input_ids is None:
            dec_input_ids = self.shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

    encoded_features = self.encoder(input_features).to(device)
    logits = self.decoder(dec_input_ids, encoded_features)

    loss = None
    if labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) 
        labels = labels.to(logits.device).long()
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

    return {
        "loss": loss,
        "logits": logits,
    }
