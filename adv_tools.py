import torch
import transformers

def mask_tokens(inputs, tokenizer, mlm_probability):
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.")

    #labels : word id
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # cast to numpy ndarray
    return inputs.detach().numpy(), labels.detach().numpy(), masked_indices.float().detach().numpy()

def mask_tokens_fix(inputs, tokenizer, mlm_probability):
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.")
    labels = inputs.clone()
    probability_matrix = torch.rand(labels.shape)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    word_num = torch.count_nonzero(probability_matrix, dim=1)
    mask_permuted = torch.zeros_like(inputs)
    for i in range(len(inputs)):
        masked_indices = torch.topk(probability_matrix[i], int(word_num[i]*mlm_probability)).indices
        labels[i][~masked_indices] = -100
        inputs[i][masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        mask_permuted[i][masked_indices] = 1
    return inputs.detach().numpy(), labels.detach().numpy(), mask_permuted.float().detach().numpy()