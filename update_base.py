
def update_base(self, new_base):
    self.encoder.combined_rotary.update_base(new_base)
    self.decoder.combined_rotary.update_base(new_base)

    for name, module in self.encoder.named_modules():
        if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding)):
            module.update_base(new_base)

    for name, module in self.decoder.named_modules():
        if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding)):
            module.update_base(new_base)

def adjust_base(self, loss, factor=1.05):
    if loss < self.best_loss:
        new_base = self.base * factor
    else:
        new_base = self.base / factor

    self.update_base(new_base)
    self.best_loss = loss
    # print(f"Adjusted base: {new_base}")
