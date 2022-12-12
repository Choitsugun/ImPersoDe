import torch.nn.functional as F
import torch.nn as nn
import torch

class Variational_Bayes(nn.Module):
    def __init__(self, d_model, n_head, n_layer, vcab_size, device):
        super().__init__()
        self.device = device
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.prior = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.poste = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.pr_fc1 = nn.Linear(in_features=d_model, out_features=d_model*4)
        self.pr_fc2 = nn.Linear(in_features=d_model*4, out_features=d_model*2)
        self.po_fc1 = nn.Linear(in_features=d_model, out_features=d_model*4)
        self.po_fc2 = nn.Linear(in_features=d_model*4, out_features=d_model*2)
        #self.bow_fc = nn.Linear(in_features=d_model*3, out_features=vcab_size)

    def posterior_net_z(self, posit_table, embed_table, prof):
        posit_ids = torch.arange(0, prof["input_ids"].size()[-1]-1, device=self.device)
        posit_ids = posit_ids.repeat(prof["input_ids"].size()[0], 1)
        posit_ids = F.pad(posit_ids, (1,0), "constant", 0)
        posit_embeds = posit_table(posit_ids)
        posit_embeds[:, :1, :] = 0    # The position of [z] is filled by 0
        input_embeds = embed_table(prof["input_ids"])
        attent_masks = torch.where(prof["attention_mask"]==0, True, False)

        output = self.poste(src=(posit_embeds+input_embeds).permute([1,0,2]), src_key_padding_mask=attent_masks)
        output = output.permute([1, 0, 2])    # len N C → N len C
        z_embeds = output[:, 0, :]
        z_mu_var = self.po_fc2(F.leaky_relu(self.po_fc1(z_embeds)))
        posterior_mu, posterior_logvar = torch.split(z_mu_var, self.d_model, dim=1)

        return posterior_mu, posterior_logvar

    def posterior_net_a(self, embed_table, resp, keyw):
        resp_embeds = embed_table(resp["input_ids"])
        keyw_embeds = embed_table(keyw["input_ids"])
        h_n = F.normalize(resp_embeds, dim=-1)
        k_e = F.normalize(keyw_embeds, dim=-1)
        distance = torch.abs(h_n.matmul(k_e.permute([0, 2, 1])))

        k_attent_masks = keyw["attention_mask"].unsqueeze(1).repeat(1, resp["input_ids"].size()[-1], 1)
        r_attent_masks = resp["attention_mask"].unsqueeze(2).repeat(1, 1, keyw["input_ids"].size()[-1])
        attent_masks = k_attent_masks * r_attent_masks
        distance = torch.where(attent_masks==1, distance, torch.tensor(0.).to(self.device))
        distance, _ = torch.max(distance, dim=1)

        max_dis, _ = torch.max(distance, dim=-1, keepdim=True)
        ave_dis = torch.sum(distance, dim=-1, keepdim=True) / torch.sum(keyw["attention_mask"], dim=-1, keepdim=True)
        dis = torch.where(max_dis>=0.4, max_dis, ave_dis)
        #dis = torch.sum(resp["attention_mask"], dim=-1, keepdim=True) / 40

        latent_a = dis.repeat(1, self.d_model)

        return latent_a

    def prior_net(self, posit_table=None, embed_table=None, cont=None, role=None, latent_z=None,
                  posit_embeds=None, input_embeds=None, role_embeds=None, attent_masks=None):
        if posit_table and embed_table and cont and role is not None:
            posit_ids = torch.arange(0, cont["input_ids"].size()[-1]-2, device=self.device)
            posit_ids = posit_ids.repeat(cont["input_ids"].size()[0], 1)
            posit_ids = F.pad(posit_ids, (2,0), "constant", 0)
            posit_embeds = posit_table(posit_ids)
            posit_embeds[:, :2, :] = 0    # The position of [z] and [a] are filled by 0
            input_embeds = embed_table(cont["input_ids"])
            role_embeds = embed_table(role)
            role_mask = role.unsqueeze(2).repeat(1, 1, self.d_model)
            role_embeds = role_mask * role_embeds
            role_embeds[:, :3, :] = 0    # The role embeds of [z] [a] and [BOS] are filled by 0
            attent_masks = torch.where(cont["attention_mask"]==0, True, False)

            src = (posit_embeds+input_embeds+role_embeds).permute([1,0,2]).contiguous()
            output = self.prior(src=src, src_key_padding_mask=attent_masks)
            output = output.permute([1, 0, 2])    # len N C → N len c
            z_embeds = output[:, 0, :]
            z_embeds = self.pr_fc2(F.leaky_relu(self.pr_fc1(z_embeds)))
            prior_mu, prior_logvar = torch.split(z_embeds, self.d_model, dim=1)

            return prior_mu, prior_logvar, posit_embeds, input_embeds, role_embeds, attent_masks
        else:
            assert latent_z     is not None
            assert posit_embeds is not None
            assert input_embeds is not None
            assert role_embeds  is not None
            assert attent_masks is not None

            input_embeds[:, 0, :] = latent_z    # This position of input embeds is filled by the latent_z
            output = \
            self.prior(src=(posit_embeds+input_embeds+role_embeds).permute([1,0,2]), src_key_padding_mask=attent_masks)
            output = output.permute([1, 0, 2])  # len N C → N len c
            prior_a = output[:, 1, :]
            """
            # ========== use for BOW ========== #
            attent_masks = attent_masks.unsqueeze(2).repeat(1, 1, self.d_model)
            c_r = torch.where(attent_masks==False, output, torch.tensor(0.).to(self.device))
            c_r = torch.sum(c_r[:, 2:, :], dim=1)
            attent_masks = torch.where(attent_masks[:, 2:, 0]==False, 1, 0)    # The tensor in the device
            c_r = torch.div(c_r, torch.sum(attent_masks, dim=-1, keepdim=True))    # c_r is not used if BOW is not used
            
            return prior_a, c_r
            """

            return prior_a

    def bow_execut(self, latent_z, latent_a, cont_repre):
        logits = torch.cat((latent_z.unsqueeze(1), latent_a.unsqueeze(1), cont_repre.unsqueeze(1)), dim=1)
        logits = logits.reshape([-1, self.d_model*3])
        bow_logits = self.bow_fc(logits)

        return bow_logits