from transformers import GPT2LMHeadModel
from model_bayvar import Variational_Bayes
from load_data import load_test_dataset
from model_selecto import Reinforce
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

    if args.n_generat > 1:
        logger.info("Using the Selector model for resp choosing")
        Sele = Reinforce(args, buil_BertTokenizer(args), logger, if_train=False)
        Sele = to_device(args.device, logger, Sele)
    elif args.n_generat == 1:
        logger.info("Generating the resp with only onetime sampling")
    else:
        logger.info("Invalid setting for n_generat, that required greater than zero")
        sys.exit()

    if not os.path.exists(args.save_resu_path):
        os.makedirs(args.save_resu_path)
    file = codecs.open(os.path.join(args.save_resu_path, "result"), 'w', 'utf8')

    conts, profs, resps = load_test_dataset(args.test_dial_path, args.test_cont_path, args.test_prof_path,
                                            args.test_resp_path, args.test_keyw_path, tokenizer, logger)
    logger.info('Starting inference')

    for cont, prof, resp in zip(conts, profs, resps):
        cont_enco = tokenizer(cont, padding=False, return_tensors="pt").to(args.device)
        cont_input_ids = cont_enco["input_ids"]

        role = make_role(cont_enco, tokenizer.eos_token_id, False)
        prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
        VarB.prior_net(GPT2.transformer.wpe, GPT2.transformer.wte, cont_enco, role)
        texts =[]

        for _ in range(args.n_generat):
            latent_z = sample_gaussian(prior_mu, prior_logvar, args.device)
            latent_a = \
            VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)
            text = generator(args, GPT2, tokenizer, cont_input_ids, latent_z, latent_a, role)
            texts.append(text)

        if args.n_generat > 1:
            text, _ = Sele.select_action(cont, texts)
        else:
            text = texts[0]

        file.write("- profil: " + prof + "\n")
        file.write("- source: " + cont + "\n")
        file.write("- expect: " + resp + "\n")
        file.write("- genera: " + text + "\n\n")

    file.close()
    logger.info('Inference finished')


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_GPT2Tokenizer(args)
    main()