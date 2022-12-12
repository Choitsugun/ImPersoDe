from transformers import GPT2LMHeadModel, GPT2Config
from load_data import load_dataset, Collater
from model_bayvar import Variational_Bayes
from torch.utils.data import DataLoader
from hyperparams import set_args
from tqdm import tqdm
import transformers
from utils import*

def save_model(GPT2, VarB, epoch):
    GPT2.save_pretrained(args.save_model_path + "/agent/epoch{}/".format(epoch))
    torch.save({'VarB': VarB.state_dict()}, args.save_model_path + "/agent/epoch{}/VarB.pth".format(epoch))
    logger.info("Saved the agent model of epoch:{}".format(epoch))


def train_epoch(GPT2, VarB, train_dataloader, optimizer, scheduler, epoch):
    GPT2.train()
    VarB.train()

    total_kl_z, total_kl_a, total_ge_l, total_pd_l = 0, 0, 0, 0
    batch_step = len(train_dataloader)

    for batch_idx, (dail, cont, prof, resp, keyw) in enumerate(tqdm(train_dataloader)):
        try:
            posterior_mu, posterior_logvar = VarB.posterior_net_z(GPT2.transformer.wpe, GPT2.transformer.wte, prof)
            role = make_role(cont, tokenizer.eos_token_id, False)
            prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
            VarB.prior_net(GPT2.transformer.wpe, GPT2.transformer.wte, cont, role)
            latent_z = sample_gaussian(posterior_mu, posterior_logvar, args.device)
            latent_a = VarB.posterior_net_a(GPT2.transformer.wte, resp, keyw)
            prior_a = \
            VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)

            d_pi, d_ie, d_am, d_la, d_ro = \
            GPT2_batch_buil(GPT2.transformer.wte, dail, latent_z, latent_a, tokenizer.eos_token_id, args.device)
            outputs = \
            GPT2.forward(position_ids=d_pi, inputs_embeds=d_ie, attention_mask=d_am, labels=d_la, token_type_ids=d_ro)

            mean_kld_z = gaussian_kld(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
            mean_kld_a = F.kl_div(prior_a.softmax(dim=-1).log(), latent_a.softmax(dim=-1), reduction='batchmean')
            gener_loss = outputs.loss
            po_di_loss = po_di_loss_cal(posterior_mu, posterior_logvar, args.dis_obj, args.device)
            loss = gener_loss + mean_kld_z + mean_kld_a + po_di_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_kl_z += float(mean_kld_z)
            total_kl_a += float(mean_kld_a)
            total_ge_l += float(gener_loss)
            total_pd_l += float(po_di_loss)

            del posterior_mu, posterior_logvar, prior_mu, prior_logvar, role, c_ro, c_pe, c_ie, c_am, latent_z, latent_a, \
                outputs, prior_a, d_pi, d_ie, d_am, d_la, d_ro, mean_kld_z, mean_kld_a, gener_loss, po_di_loss, loss

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    kl_z = total_kl_z / batch_step
    kl_a = total_kl_a / batch_step
    ge_l = total_ge_l / batch_step
    pd_l = total_pd_l / batch_step
    logger.info("Training epoch:{} KL_z:{} KL_a:{} Loss:{} Po-di:{}".format(epoch, kl_z, kl_a, ge_l, pd_l))


def valid_epoch(GPT2, VarB, valid_dataloader, epoch):
    GPT2.eval()
    VarB.eval()

    total_kl_z, total_kl_a, total_ge_l = 0, 0, 0
    batch_step = len(valid_dataloader)

    for batch_idx, (dail, cont, prof, resp, keyw) in enumerate(tqdm(valid_dataloader)):
        try:
            posterior_mu, posterior_logvar = VarB.posterior_net_z(GPT2.transformer.wpe, GPT2.transformer.wte, prof)
            role = make_role(cont, tokenizer.eos_token_id, False)
            prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
            VarB.prior_net(GPT2.transformer.wpe, GPT2.transformer.wte, cont, role)
            latent_z = sample_gaussian(posterior_mu, posterior_logvar, args.device)
            latent_a = VarB.posterior_net_a(GPT2.transformer.wte, resp, keyw)
            prior_a = \
            VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)

            d_pi, d_ie, d_am, d_la, d_ro = \
            GPT2_batch_buil(GPT2.transformer.wte, dail, latent_z, latent_a, tokenizer.eos_token_id, args.device)
            outputs = \
            GPT2.forward(position_ids=d_pi, inputs_embeds=d_ie, attention_mask=d_am, labels=d_la, token_type_ids=d_ro)

            mean_kld_z = gaussian_kld(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
            mean_kld_a = F.kl_div(prior_a.softmax(dim=-1).log(), latent_a.softmax(dim=-1), reduction='batchmean')
            gener_loss = outputs.loss

            total_kl_z += float(mean_kld_z)
            total_kl_a += float(mean_kld_a)
            total_ge_l += float(gener_loss)

            del posterior_mu, posterior_logvar, prior_mu, prior_logvar, role, c_ro, c_pe, c_ie, c_am, latent_z, \
                latent_a, outputs, prior_a, d_pi, d_ie, d_am, d_la, d_ro, mean_kld_z, mean_kld_a, gener_loss

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    kl_z = total_kl_z / batch_step
    kl_a = total_kl_a / batch_step
    ge_l = total_ge_l / batch_step
    elbo = kl_z + kl_a + ge_l
    logger.info("Validating epoch:{} KL_z:{} KL_a:{} Loss:{} ELBO:{}".format(epoch, kl_z, kl_a, ge_l, elbo))

    return elbo


def train(GPT2, VarB, train_data, valid_data):
    patience = 0
    best_val_loss = float('Inf')
    collate_fn = Collater(tokenizer, args.device)
    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)
    valid_dataloader = \
    DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    if args.forwa_only:
        # ========== eval ========== #
        logger.info('Starting validating')
        valid_epoch(GPT2, VarB, valid_dataloader, None)
        logger.info('Validating finished')
    else:
        # ========== train ========== #
        t_total = len(train_dataloader) * args.epochs
        optimizer = transformers.AdamW(list(GPT2.parameters()) + list(VarB.parameters()), lr=args.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)
        logger.info('Starting training')

        for epoch in range(1, args.epochs+1):
            train_epoch(GPT2, VarB, train_dataloader, optimizer, scheduler, epoch)
            val_loss = valid_epoch(GPT2, VarB, valid_dataloader, epoch)

            if (val_loss < best_val_loss and args.early_stop) or args.niter_save <= epoch:
                # Save GPT-2 model and variational model
                save_model(GPT2, VarB, epoch)
                best_val_loss = val_loss
                patience = 0
            else:
                # This variable is useless when early_stop is False
                patience = patience + 1

            if args.patience < patience and args.early_stop:
                logger.info("Early stop due to run out of patience")
                break

        logger.info('Training finished')


def main():
    device_assign(args, logger)

    train_data = load_dataset(args.train_dial_path, args.train_cont_path, args.train_prof_path,
                              args.train_resp_path, args.train_keyw_path, tokenizer, logger)
    valid_data = load_dataset(args.valid_dial_path, args.valid_cont_path, args.valid_prof_path,
                              args.valid_resp_path, args.valid_keyw_path, tokenizer, logger)

    if args.forwa_only:
        GPT2 = GPT2LMHeadModel.from_pretrained(args.GPT2_checkp)
        VarB = Variational_Bayes(args.d_model, args.n_head, args.n_layer, len(tokenizer), args.device)
        VarB.load_state_dict(torch.load(args.VarB_checkp)["VarB"])
        logger.info("Restored the GPT-2 model and the variational Bayes model from the check point")
    else:
        if args.GPT2_pretrained:
            GPT2 = GPT2LMHeadModel.from_pretrained(args.GPT2_pretrained)
            GPT2.transformer.wte.weight.requires_grad = False
            GPT2.transformer.wpe.weight.requires_grad = False
            logger.info("Initialized the GPT-2 model from the pretrained weight")
        else:
            GPT2_config = GPT2Config.from_json_file(args.GPT2_config)
            GPT2 = GPT2LMHeadModel(config=GPT2_config)
            logger.info("Initialized the GPT-2 model")

        vocab_size = GPT2.config.vocab_size
        GPT2.resize_token_embeddings(len(tokenizer))
        expansion_embed_init(GPT2, len(tokenizer)-vocab_size, logger)

        VarB = Variational_Bayes(args.d_model, args.n_head, args.n_layer, len(tokenizer), args.device)
        logger.info("Initialized the variational Bayes model")

    GPT2, VarB = to_device(args.device, logger, GPT2, VarB)
    train(GPT2, VarB, train_data, valid_data)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_GPT2Tokenizer(args)
    main()