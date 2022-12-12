from transformers import GPT2LMHeadModel
from model_bayvar import Variational_Bayes
from load_data import load_test_dataset
from hyperparams import set_args
import numpy as np
from utils import*
import os

def var_save(np_var, save_path, symbol):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    var_path = os.path.join(save_path, "%s.npy" %(symbol))
    np.save(var_path, np_var)


def main():
    device_assign(args, logger)

    GPT2 = GPT2LMHeadModel.from_pretrained(args.GPT2_checkp)
    VarB = Variational_Bayes(args.d_model, args.n_head, args.n_layer, len(tokenizer), args.device)
    VarB.load_state_dict(torch.load(args.VarB_checkp)["VarB"])
    logger.info("Restored the GPT-2 model and the variational Bayes model from the check point")
    GPT2, VarB = to_device(args.device, logger, GPT2, VarB)
    GPT2.eval()
    VarB.eval()

    # The dm file needs to be set to have the same number of lines as the al file.
    conts, profs, _ = load_test_dataset(args.dm_dial_path, args.al_cont_path, args.al_prof_path, args.dm_resp_path,
                                        args.dm_keyw_path, tokenizer, logger)

    #=================posterior extra================#
    logger.info('Starting the posterior extra')
    mu     = np.empty(shape=(0, args.d_model), dtype=np.float32)
    logvar = np.empty(shape=(0, args.d_model), dtype=np.float32)
    z      = np.empty(shape=(0, args.d_model), dtype=np.float32)

    for i, prof in enumerate(profs):
        if i+1 > args.n_prof:
            break

        prof = tokenizer(prof, padding=False, return_tensors="pt").to(args.device)
        posterior_mu, posterior_logvar = VarB.posterior_net_z(GPT2.transformer.wpe, GPT2.transformer.wte, prof)

        for i in range(args.n_sample):
            latent_z = sample_gaussian(posterior_mu, posterior_logvar, args.device)
            latent_z = latent_z.data.cpu().numpy()
            z = np.concatenate((z, latent_z), axis=0)

        posterior_mu = posterior_mu.data.cpu().numpy()
        posterior_logvar = posterior_logvar.data.cpu().numpy()
        mu     = np.concatenate((mu, posterior_mu), axis=0)
        logvar = np.concatenate((logvar, posterior_logvar), axis=0)

    var_save(mu, args.var_poste_save, "mu")
    var_save(logvar, args.var_poste_save, "logvar")
    var_save(z, args.var_poste_save, "z")
    logger.info('The posterior extra finished')

    #===================prior extra==================#
    logger.info('Starting the prior extra')
    mu     = np.empty(shape=(0, args.d_model), dtype=np.float32)
    logvar = np.empty(shape=(0, args.d_model), dtype=np.float32)
    z      = np.empty(shape=(0, args.d_model), dtype=np.float32)
    a      = np.empty(shape=(0, args.d_model), dtype=np.float32)

    for i, cont in enumerate(conts):
        if i+1 > args.n_cont:
            break

        cont = tokenizer(cont, padding=False, return_tensors="pt").to(args.device)
        role = make_role(cont, tokenizer.eos_token_id, False)
        prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
        VarB.prior_net(GPT2.transformer.wpe, GPT2.transformer.wte, cont, role)
        latent_z = sample_gaussian(prior_mu, prior_logvar, args.device)
        latent_a = \
        VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)

        prior_mu = prior_mu.data.cpu().numpy()
        prior_logvar = prior_logvar.data.cpu().numpy()
        latent_z = latent_z.data.cpu().numpy()
        latent_a = latent_a.data.cpu().numpy()

        mu = np.concatenate((mu, prior_mu), axis=0)
        logvar = np.concatenate((logvar, prior_logvar), axis=0)
        z = np.concatenate((z, latent_z), axis=0)
        a = np.concatenate((a, latent_a), axis=0)

    var_save(mu, args.var_prior_save, "mu")
    var_save(logvar, args.var_prior_save, "logvar")
    var_save(z, args.var_prior_save, "z")
    var_save(a, args.var_prior_save, "a")
    logger.info('The prior extra finished')


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_GPT2Tokenizer(args)
    main()