# author: akshitac8
# tf-TFVAEGAN inductive
from __future__ import print_function

import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# import functions
import networks.TFVAEGAN_model as model
import datasets.image_util as util
import classifiers.classifier_images as classifier
from config_images import opt
from scipy import io as sio


class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'a+')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass


def train_classifier_with_no_syn(data):
    if opt.gzsl:
        # Concatenate real seen/unseen features with synthesized unseen features
        train_X = data.classifier_train_feature
        train_Y = data.classifier_train_label
        gzsl_weight = data.classifier_train_weight
        nclass = data.allclasses.shape[0]
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, gzsl_weight, data, nclass, opt.cuda, opt.classifier_lr, 0.5,
                                         opt.nepoch_classifier, generalized = True, netDec = None,
                                         dec_size = opt.attSize, dec_hidden_size = 4096)
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H), end = " ")

    # Zero-shot learning
    # Train ZSL classifier
    train_X = data.aux_unseen_feature if opt.n_aux_per_class else None
    train_Y = data.aux_unseen_label if opt.n_aux_per_class else None
    zsl_weight = data.aux_unseen_weight if opt.n_aux_per_class else None
    if train_X != None and train_Y != None:
        zsl_cls = classifier.CLASSIFIER(train_X, util.map_label(train_Y, data.unseenclasses), zsl_weight, data,
                                        data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5,
                                        opt.nepoch_classifier, generalized = False, netDec = None,
                                        dec_size = opt.attSize, dec_hidden_size = 4096)
        acc = zsl_cls.acc
        print('ZSL: unseen accuracy=%.4f' % (acc))
    exit(0)


now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
log_dir = f'log/{opt.dataset}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# oc代表外部数据仅用于训练分类器
# 没有oc代表外部数据用于训练vae-gan和分类器
# 其他的不同策略请自行在strategy字段里标明
log_file = f'{opt.filtering_strategy}_{"" if opt.aux_to_vae else "oc_"}a{opt.n_aux_per_class}_s{opt.syn_num}_{now}.log'
logger = Logger(log_dir + log_file)
sys.stdout = logger
sys.stderr = logger

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
opt.attSize = data.attribute.shape[1]
opt.nz = opt.attSize
opt.latent_size = opt.attSize
print(json.dumps(opt.__dict__, indent = 4))

if not opt.syn_num:
    train_classifier_with_no_syn(data)

print("# of training samples: ", data.ntrain_vae)
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxiliary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt, opt.attSize)
print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
input_weight = torch.FloatTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1]).data[0]
mone = one * -1
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    input_weight = input_weight.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()


def loss_fn(recon_x, x, mean, log_var, weight):
    # 这里的bce只表示距离的度量
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction = 'none')
    BCE = BCE.T * weight
    BCE = BCE.sum() / x.size(0)
    # -0.5 * E(1 + log(var) - mean^2 - var)
    KLD = -0.5 * torch.sum((1 + log_var - mean.pow(2) - log_var.exp()).T * weight) / x.size(0)
    return (BCE + KLD)


def sample():
    batch_feature, batch_weight, batch_att = data.next_vae_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_weight.copy_(batch_weight)
    input_att.copy_(batch_att)


def WeightedL1(pred, gt, weight):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    loss = loss.T * weight
    return loss.sum() / loss.size(0)


def generate_syn_feature(generator, classes, attribute, num, netF = None, netDec = None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise, requires_grad = False)
        syn_attv = Variable(syn_att, requires_grad = False)
        fake = generator(syn_noisev, c = syn_attv)
        if netF is not None:
            dec_out = netDec(fake)  # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet()  # no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1 = opt.a2, c = syn_attv, feedback_layers = feedback_out)
        output = fake
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


optimizerE = optim.Adam(netE.parameters(), lr = opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr = opt.feed_lr, betas = (opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr = opt.dec_lr, betas = (opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att, input_weight):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad = True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs = disc_interpolates, inputs = interpolates,
                              grad_outputs = ones,
                              create_graph = True, retain_graph = True, only_inputs = True)[0]
    gradient_penalty = (((gradients.norm(2, dim = 1) - 1) ** 2) * input_weight).mean() * opt.lambda1
    return gradient_penalty


best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0, opt.nepoch):
    for loop in range(0, opt.feedback_loop):
        for i in range(0, data.ntrain_vae, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.parameters():  # unfreeze discrimator
                p.requires_grad = True

            for p in netDec.parameters():  # unfreeze deocder
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0  # lAMBDA VARIABLE
            # critic 就是评判 指的是discrimination
            # 与之相应的是classification和generation
            for iter_d in range(opt.critic_iter):
                # data loading
                sample()
                netD.zero_grad()
                # resv就是resnet提取的visual feats
                # torch.Size([64, 2048])
                input_resv = Variable(input_res)
                input_weightv = Variable(input_weight)
                # torch.Size([64, att_size])
                input_attv = Variable(input_att)

                netDec.zero_grad()
                # recons mean reconstructed attr
                recons = netDec(input_resv)
                R_cost = opt.recons_weight * WeightedL1(recons, input_attv, input_weightv)
                # R loss
                # reconstructed attr 和 attr 的 L1 距离
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real *= input_weightv.resize(input_weightv.shape[0], 1)
                criticD_real = opt.gammaD * criticD_real.mean()
                # lossD_real = lambda0 * E(D(x,a))
                criticD_real.backward(mone)
                # if encoded_noise: z 是由 feat encode 成的
                # 否则是从正态随机采样
                if opt.encoded_noise:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means  # torch.Size([64, 312])
                else:
                    # Fills self tensor with elements samples from the normal distribution parameterized by mean and std
                    noise.normal_(0, 1)
                    z = Variable(noise)

                # loop为1时 进行带feed back的generate
                # 否则generate不带有feedback
                if loop == 1:
                    fake = netG(z, c = input_attv)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(z, a1 = opt.a1, c = input_attv, feedback_layers = feedback_out)
                else:
                    fake = netG(z, c = input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake *= input_weightv.resize(input_weightv.shape[0], 1)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                # lossD_fake=lambda1*E(D(xhat,a))
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_att,
                                                                      input_weight)
                # if opt.lambda_mult == 1.1:
                #
                gp_sum += gradient_penalty.data
                # errD = errD_real + errD_fake + gradient_penalty
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty  # add Y here and #add vae reconstruction loss
                optimizerD.step()

            # 在critic一轮训练结束后用 gradient penalty的和来判断应当增加惩罚力度还是减少
            gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters():  # freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters():  # freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            # E不是直接生成z 而是生成mean和log(var)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means  # torch.Size([64, 312])
            # recon_x = z
            # dec_out = g_feat
            # loop为1时 进行带feed back的generate
            # 否则generate不带有feedback
            if loop == 1:
                recon_x = netG(z, c = input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1 = opt.a1, c = input_attv, feedback_layers = feedback_out)
            else:
                recon_x = netG(z, c = input_attv)

            # minimize E 3 with this setting feedback will update the loss as well
            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var, input_weightv)
            errG = vae_loss_seen

            if opt.encoded_noise:
                criticG_fake = (netD(recon_x, input_attv).T * input_weightv).mean()
                fake = recon_x
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake = netG(noisev, c = input_attv)
                    dec_out = netDec(recon_x)  # Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1 = opt.a1, c = input_attv, feedback_layers = feedback_out)
                else:
                    fake = netG(noisev, c = input_attv)
                criticG_fake = (netD(fake, input_attv).T * input_weightv).mean()

            G_cost = -criticG_fake
            errG += opt.gammaG * G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv, input_weightv)
            errG += opt.recons_weight * R_cost
            # 在encoder and generator训练的过程中也会更新Dec参数
            errG.backward()
            # write a condition here
            optimizerE.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec:  # not train decoder at feedback time
                optimizerDec.step()

    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' % (
        epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), vae_loss_seen.item()), end = " ")
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, netF = netF,
                                                  netDec = netDec)
    # Generalized zero-shot learning
    if opt.gzsl:
        # Concatenate real seen features, web unseen feats and synthesized unseen features
        train_X = torch.cat((data.classifier_train_feature, syn_feature), 0)
        train_Y = torch.cat((data.classifier_train_label, syn_label), 0)
        gzsl_weight = torch.cat((data.classifier_train_weight, torch.ones(syn_feature.shape[0])))
        nclass = data.allclasses.shape[0]
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, gzsl_weight, data, nclass, opt.cuda, opt.classifier_lr, 0.5,
                                         opt.nepoch_classifier, opt.syn_num, generalized = True, netDec = netDec,
                                         dec_size = opt.attSize, dec_hidden_size = 4096)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
            syn_mat = {'feat': np.array(syn_feature.cpu()), 'label': np.array(syn_label.cpu())}
            sio.savemat('syn.mat', syn_mat)

        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))
        if abs(gzsl_cls.acc_seen - gzsl_cls.acc_unseen) >= 0.05:
            data.seen_weight_alpha_classifier = data.seen_weight_alpha_classifier * (
                1.1 if gzsl_cls.acc_seen < gzsl_cls.acc_unseen else 0.9)
            data.classifier_train_weight = torch.cat(
                [data.train_weight * data.seen_weight_alpha_classifier, data.aux_unseen_weight], dim = 0)
            print(f'seen weight alpha of classifier: {data.seen_weight_alpha_classifier}')
    # Zero-shot learning
    # Train ZSL classifier
    train_X = torch.cat((data.aux_unseen_feature, syn_feature), 0) if opt.n_aux_per_class else syn_feature
    train_Y = torch.cat((data.aux_unseen_label, syn_label), 0) if opt.n_aux_per_class else syn_label
    zsl_weight = torch.cat((data.aux_unseen_weight,
                            torch.ones(syn_feature.shape[0]))) if opt.n_aux_per_class else torch.ones(
        syn_feature.shape[0])
    zsl_cls = classifier.CLASSIFIER(train_X, util.map_label(train_Y, data.unseenclasses), zsl_weight, data,
                                    data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier,
                                    opt.syn_num, generalized = False, netDec = netDec, dec_size = opt.attSize,
                                    dec_hidden_size = 4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    # reset G to training mode
    netG.train()
    netDec.train()
    netF.train()

print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
