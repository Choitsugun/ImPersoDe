from transformers import GPT2LMHeadModel
from model_bayvar import Variational_Bayes
from load_data import load_test_dataset
from hyperparams import set_args
from utils import*
import codecs
import os

def main():
    device_assign(args, logger)

    GPT2 = GPT2LMHeadModel.from_pretrained(args.GPT2_checkp)
    VarB = Variational_Bayes(args.d_model, args.n_head, args.n_layer, len(tokenizer), args.device)
    VarB.load_state_dict(torch.load(args.VarB_checkp)["VarB"])
    logger.info("Restored the GPT-2 model and the variational Bayes model from the check point")
    GPT2, VarB = to_device(args.device, logger, GPT2, VarB)
    GPT2.eval()
    VarB.eval()

    if not os.path.exists(args.save_resu_path):
        os.makedirs(args.save_resu_path)
    file = codecs.open(os.path.join(args.save_resu_path, "result"), 'w', 'utf8')

    conts, _, _ = load_test_dataset(args.dm_dial_path, args.al_cont_path, args.al_prof_path, args.dm_resp_path,
                                    args.dm_keyw_path, tokenizer, logger)
    logger.info('Starting boost the Za')
    count_len = [0,0,0,0,0,0,0,0,0,0,0]
    count = None

    for i, cont in enumerate(conts):
        count = i + 1
        if count > args.n_cont:
            break

        cont_enco = tokenizer(cont, padding=False, return_tensors="pt").to(args.device)
        role = make_role(cont_enco, tokenizer.eos_token_id, False)
        cont_input_ids = cont_enco["input_ids"]

        prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
        VarB.prior_net(GPT2.transformer.wpe, GPT2.transformer.wte, cont_enco, role)
        latent_z = sample_gaussian(prior_mu, prior_logvar, args.device)

        for n in range(0, 11):
            latent_a = (torch.ones(1, 1024).to(args.device) + torch.randn(1, 1024).to(args.device)*0.25) * n * 0.1
            text = generator(args, GPT2, tokenizer, cont_input_ids, latent_z, latent_a, role)
            file.write("- genera: " + text + "\n")
            count_len[n] = count_len[n] + len(text.split())

    for i in range(11):
        print(count_len[i] / count)

    file.close()
    logger.info('The gradient finished')


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_GPT2Tokenizer(args)
    main()