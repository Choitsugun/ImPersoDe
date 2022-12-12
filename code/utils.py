from transformers import GPT2Tokenizer, BertTokenizer
import torch.nn.functional as F
import logging
import torch
import sys

def gaussian_kld(po_mu, po_logvar, pr_mu, pr_logvar):
    kld = -0.5 * torch.mean(1 + (po_logvar - pr_logvar) - torch.div((pr_mu - po_mu).pow(2), pr_logvar.exp())
                            - torch.div(po_logvar.exp(), pr_logvar.exp()))

    return kld


def gaussian_kld_dimkp(n_mu, n_va, l_mu, l_va):
    kld = -0.5 * (1 + (n_va - l_va) - torch.div((l_mu - n_mu).pow(2), l_va.exp())
                  - torch.div(n_va.exp(), l_va.exp()))

    return kld


def sample_gaussian(mu, logvar, device):
    epsilon = torch.randn(mu.shape).to(device)
    std = (0.5 * logvar).exp()
    z = mu + std * epsilon

    return z


def po_di_loss_cal(posterior_mu, posterior_logvar, dis_obj, device):
    batch_size = posterior_mu.size()[0]
    l_mu = posterior_mu.unsqueeze(1).repeat(1, batch_size, 1)
    l_va = posterior_logvar.unsqueeze(1).repeat(1, batch_size, 1)
    n_mu = posterior_mu.unsqueeze(0).repeat(batch_size, 1, 1)
    n_va = posterior_logvar.unsqueeze(0).repeat(batch_size, 1, 1)

    kl = gaussian_kld_dimkp(n_mu, n_va, l_mu, l_va)
    kl = torch.where(kl==0, torch.tensor(dis_obj).to(device), kl)
    kl = kl - dis_obj
    kl = torch.where(kl<0, kl, torch.tensor(0.).to(device))
    po_di_loss = torch.sum(torch.mean(kl**2, dim=-1))

    return po_di_loss


def bow_loss_cal(bow_logits, resp, pos_class, vcab_size, pad_token_id, device):
    targets = F.one_hot(resp["input_ids"], vcab_size)
    targets[:, :, pad_token_id] = 0    # Ignore the targets padding tokens loss
    bow_targets = torch.minimum(torch.sum(targets, dim=1), torch.tensor(1.).to(device))
    pos_weight = pos_class * torch.ones([vcab_size]).to(device)
    bow_loss = F.binary_cross_entropy_with_logits(bow_logits, bow_targets, reduction='mean', pos_weight=pos_weight)

    return bow_loss


def make_role(inputs, target_value, dial_role):
    input_ids = torch.flip(inputs["input_ids"], [1])
    bl = (input_ids == target_value)
    bl = torch.where(bl==False, 0, 1)
    count = torch.cumsum(bl, dim=-1)

    if dial_role:
        role = torch.where(count%2==0, 1, 0)
    else:
        role = torch.where(count%2!=0, 1, 0)

    role = torch.flip(role, [1])

    return role * inputs["attention_mask"]


def make_mask(inputs, target_value, point=2):
    batch_size, max_length = inputs.shape
    bl = (inputs == target_value)
    nonzero = torch.nonzero(bl)
    count = torch.count_nonzero(bl, dim=1)
    cumsum = torch.cumsum(count, dim=0)
    indices = cumsum - point
    l_rep = nonzero[:, 1][indices].view(-1, 1).repeat(1, max_length)
    mask = torch.arange(max_length).repeat(batch_size, 1).to(inputs.device)

    return l_rep < mask


def GPT2_batch_buil(embed_table, dail, latent_z, latent_a, eos_token_id, device):
    posit_ids = torch.arange(0, dail["input_ids"].size()[-1]-2, device=device)
    posit_ids = posit_ids.repeat(dail["input_ids"].size()[0], 1)
    Ge_posit_ids = F.pad(posit_ids, (2,0), "constant", 0)

    Ge_in_embeds = embed_table(dail["input_ids"])
    Ge_in_embeds[:, 0, :] = latent_z
    Ge_in_embeds[:, 1, :] = latent_a

    Ge_atte_mask = dail["attention_mask"]

    cont_mask = make_mask(inputs=dail["input_ids"], target_value=eos_token_id)
    Ge_labels = torch.where(cont_mask & (dail["attention_mask"]==1), dail["input_ids"], -100)

    Ge_role = make_role(dail, eos_token_id, True)

    # The posit embeds of [z] [a] are set as 0 in transformers.models.gpt2.modeling_gpt2
    # The role embeds of [z] [a] [BOS] are set as 0 in transformers.models.gpt2.modeling_gpt2
    return Ge_posit_ids, Ge_in_embeds, Ge_atte_mask, Ge_labels, Ge_role


def GPT2_test_buil(embed_table, cont_input_ids, latent_z, latent_a, device):
    posit_ids = torch.arange(0, cont_input_ids.size()[-1]-2, device=device)
    posit_ids = posit_ids.repeat(cont_input_ids.size()[0], 1)
    Ge_posit_ids = F.pad(posit_ids, (2,0), "constant", 0)

    Ge_in_embeds = embed_table(cont_input_ids)
    Ge_in_embeds[:, 0, :] = latent_z
    Ge_in_embeds[:, 1, :] = latent_a

    return Ge_posit_ids, Ge_in_embeds


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., :1] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generator(args, GPT2, tokenizer, cont_input_ids, latent_z, latent_a, role):
    generate = []

    for _ in range(args.max_resp_len):
        c_pi, c_ie = GPT2_test_buil(GPT2.transformer.wte, cont_input_ids, latent_z, latent_a, args.device)
        outputs = GPT2.forward(position_ids=c_pi, inputs_embeds=c_ie, token_type_ids=role)
        next_token_logits = outputs.logits[0, -1, :]

        for id in set(generate):
            if next_token_logits[id] > 0:
                next_token_logits[id] /= args.repeti_penalty
            else:
                next_token_logits[id] *= args.repeti_penalty

        next_token_logits = next_token_logits / args.temperature
        next_token_logits[args.ignore_tks_ids] = -float('Inf')
        next_token_logits[tokenizer.pad_token_id] = -float('Inf')
        next_token_logits[tokenizer.additional_special_tokens_ids] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

        if next_token == tokenizer.eos_token_id:
            break
        else:
            generate.append(next_token.item())
            cont_input_ids = torch.cat((cont_input_ids, next_token.unsqueeze(0)), dim=1)
            role = torch.cat((role, torch.tensor([0]).to(args.device).unsqueeze(0)), dim=1)

    return tokenizer.decode(generate)


def expansion_embed_init(model, nexpan, logger):
    if nexpan < 0:
        logger.info("The length of tokenizer shorter than the vocab size")
        sys.exit()
    elif nexpan == 0:
        logger.info("We don't resize the embedding tabel since the length of tokenizer equal vocab size")
    else:
        params = model.state_dict()
        embeddings = params['transformer.wte.weight']

        pre_expansion_embeddings = embeddings[:-nexpan, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)

        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(nexpan))), dim=0)
        embeddings[-nexpan:, :] = new_embeddings
        params['transformer.wte.weight'] = embeddings
        model.load_state_dict(params)

        logger.info("Resize the length: {} for the embedding tabel".format(nexpan))


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def buil_GPT2Tokenizer(args):
    tokenizer = \
    GPT2Tokenizer.from_pretrained(args.GPT2_tokenizer, pad_token="[PAD]", additional_special_tokens=["[z]", "[a]"])

    return tokenizer


def buil_BertTokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.Bert_tokenizer)

    return tokenizer


def device_assign(args, logger):
    torch.multiprocessing.set_start_method('spawn')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device:{}'.format(args.device))


def to_device(device, logger, *params):
    if len(params) == 1:
        logger.info("One model is discovered")
        model = params[0]
        model.to(device)
        logger.info("Using {} to train/eval it".format(device))

        return model

    elif len(params) == 2:
        logger.info("Two models are discovered")
        model1, model2 = params
        model1.to(device)
        model2.to(device)
        logger.info("Using {} to train/eval them".format(device))

        return model1, model2

    else:
        logger.info("Invalid number of models, please check the argument")
        sys.exit()