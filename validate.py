#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn

import onmt

from onmt.inputters.inputter import build_dataset_iter
from onmt.model_builder import build_base_model


def shannon_entropy(p_vector):
    log_p = torch.log(p_vector)
    log_p[log_p == -float("Inf")] = 0
    return -(p_vector * log_p).sum(-1)


class Validator(object):
    def __init__(self, model, tgt_padding_idx):
        self.model = model
        self.tgt_padding_idx = tgt_padding_idx

        # Set model in training mode.
        self.model.eval()

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        """
        # Set model in validating mode.
        stats = {'support': 0, 'tgt_words': 0, 'src_words': 0,
                 'attended': 0, 'attended_possible': 0,
                 'self_attended_possible': 0}
        N_JS_div = 0
        with torch.no_grad():
            for batch in valid_iter:

                src, src_lengths = batch.src
                stats['src_words'] += src_lengths.sum().item()

                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = self.model(src, tgt, src_lengths)
                # outputs is seq x batch x hidden_size
                bottled_out = outputs.view(-1, outputs.size(-1))
                generator_out = self.model.generator(bottled_out)

                tgt_lengths = tgt[1:].squeeze(2).ne(
                    self.tgt_padding_idx).sum(dim=0)
                grid_sizes = src_lengths * tgt_lengths
                stats['attended_possible'] += grid_sizes.sum().item()

                self_grid_sizes = tgt_lengths * tgt_lengths
                stats['self_attended_possible'] += self_grid_sizes.sum().item()

                out_support = generator_out.gt(0).sum(dim=1)
                tgt_non_pad = tgt[1:].ne(self.tgt_padding_idx).view(-1)
                support_non_pad = out_support.masked_select(tgt_non_pad)
                tgt_words = support_non_pad.size(0)
                stats['support'] += support_non_pad.sum().item()
                stats['tgt_words'] += tgt_words

                attn = attns['std']
                attn = attn.view(-1, attn.size(-1))
                attended = attn.gt(0).sum(dim=1)
                attended_non_pad = attended.masked_select(tgt_non_pad)
                stats['attended'] += attended_non_pad.sum().item()

                stats['n_layers'] = attns['self'].size(0)
                self_attn = attns['self']
                for ii, layer in enumerate(self_attn):
                    if 'self_layer_%i' % (ii) not in stats.keys():
                        # number of heads
                        stats['self_layer_%i' % (ii)] = [0] * layer.size(0)
                        stats['head_sum_self_layer_%i' % (ii)] = 0
                        stats['JS_div_self_layer_%i' % (ii)] = 0
                    for jj, head in enumerate(layer):
                        head = head.view(-1, head.size(-1))
                        attended = head.gt(0).sum(dim=1)
                        attended_non_pad = attended.masked_select(tgt_non_pad)
                        stats['self_layer_%i' % (ii)][jj] += \
                            attended_non_pad.sum().item()

                    head_sum = layer.sum(0).view(-1, head.size(-1))
                    attended = head_sum.gt(0).sum(dim=1)
                    attended_non_pad = attended.masked_select(tgt_non_pad)
                    stats['head_sum_self_layer_%i' % (ii)] += \
                        attended_non_pad.sum().item()

                    # n_heads = layer.size(0)
                    # N_lens = tgt_lengths.repeat(layer.size(1))
                    # n_ij_att = layer.gt(0).sum(0).view(-1, layer.size(-1))
                    # n_ij_no_att = layer.eq(0).sum(0).view(-1, layer.size(-1))
                    # p_att = \
                    #     n_ij_att.sum(-1).masked_select(
                    #         tgt_non_pad).float() / \
                    #     (N_lens*n_heads).masked_select(
                    #         tgt_non_pad).float()
                    # p_no_att = 1 - p_att
                    # P_i = (n_ij_att**2 + n_ij_no_att**2 - n_heads).float()/\
                    #     (n_heads*(n_heads-1))

                    # P_bar = P_i.sum(-1)/P_i.size(-1)
                    # P_bar = P_bar.masked_select(tgt_non_pad)
                    # P_bar_e = p_att**2 + p_no_att**2
                    # kappa = (P_bar - P_bar_e)/(1 - P_bar_e)

                    JS_div = \
                        (shannon_entropy(
                            (layer.sum(0) / layer.size(0)).view(
                                -1, layer.size(-1))) -
                         (shannon_entropy(layer) / layer.size(0)).sum(0).view(
                            -1)).masked_select(tgt_non_pad)

                    stats['JS_div_self_layer_%i' % (ii)] += \
                        (JS_div /
                            torch.log(
                                JS_div.new_ones(JS_div.size())*layer.size(0)
                                )).sum().item()

                context_attn = attns['context']
                for ii, layer in enumerate(context_attn):
                    if 'context_layer_%i' % (ii) not in stats.keys():
                        # number of heads
                        stats['context_layer_%i' % (ii)] = [0] * layer.size(0)
                        stats['head_sum_context_layer_%i' % (ii)] = 0
                        stats['JS_div_context_layer_%i' % (ii)] = 0
                    for jj, head in enumerate(layer):
                        head = head.view(-1, head.size(-1))
                        attended = head.gt(0).sum(dim=1)
                        attended_non_pad = attended.masked_select(tgt_non_pad)
                        stats['context_layer_%i' % (ii)][jj] += \
                            attended_non_pad.sum().item()

                    head_sum = layer.sum(0).view(-1, head.size(-1))
                    attended = head_sum.gt(0).sum(dim=1)
                    attended_non_pad = attended.masked_select(tgt_non_pad)
                    stats['head_sum_context_layer_%i' % (ii)] += \
                        attended_non_pad.sum().item()

                    # n_heads = layer.size(0)
                    # N_lens = tgt_lengths.repeat(layer.size(1))
                    # n_ij_att = layer.gt(0).sum(0).view(-1, layer.size(-1))
                    # n_ij_no_att = layer.eq(0).sum(0).view(-1, layer.size(-1))
                    # p_att = \
                    #     n_ij_att.sum(-1).masked_select(
                    #         tgt_non_pad).float() / \
                    #     (N_lens*n_heads).masked_select(
                    #         tgt_non_pad).float()
                    # p_no_att = 1 - p_att
                    # P_i = (n_ij_att**2 + n_ij_no_att**2 - n_heads).float()/\
                    #     (n_heads*(n_heads-1))

                    # P_bar = P_i.sum(-1)/P_i.size(-1)
                    # P_bar = P_bar.masked_select(tgt_non_pad)
                    # P_bar_e = p_att**2 + p_no_att**2
                    # kappa = (P_bar - P_bar_e)/(1 - P_bar_e)

                    JS_div = \
                        (shannon_entropy(
                            (layer.sum(0) / layer.size(0)).view(
                                -1, layer.size(-1))) -
                         (shannon_entropy(layer) / layer.size(0)).sum(0).view(
                            -1)).masked_select(tgt_non_pad)

                    stats['JS_div_context_layer_%i' % (ii)] += \
                        (JS_div /
                            torch.log(
                                JS_div.new_ones(JS_div.size())*layer.size(0)
                                )).sum().item()

                N_JS_div += len(JS_div)

            for ii in range(len(self.model.decoder.transformer_layers)):

                stats['JS_div_self_layer_%i' % (ii)] /= N_JS_div
                stats['JS_div_context_layer_%i' % (ii)] /= N_JS_div

                '''
                print(src.size())
                print(tgt.size())
                foo = attns['std'].squeeze(1)
                print(foo.size())
                print(foo.sum())
                # what's going on here: the tgt is size 10, the src is size 8,
                # the attention is (9 x 8).
                print('attn nonzeros', foo.gt(0).sum().item())
                print('total src words', src_lengths.sum().item())
                print('total tgt words', tgt_lengths.sum().item())
                print('src seq',
                      [self.fields['src'].vocab.itos[i] for i in src])
                print('tgt seq',
                      [self.fields['tgt'].vocab.itos[i] for i in tgt])
                print(foo)
                '''

        return stats


def load_model(checkpoint, fields, k=0, bisect_iter=0, gpu=False):
    model_opt = checkpoint['opt']
    alpha_lookup = {'softmax': 1.0, 'entmax': 1.5, 'sparsemax': 2.0}
    if not hasattr(model_opt, 'loss_alpha'):
        model_opt.loss_alpha = alpha_lookup[model_opt.generator_function]
    gen_alpha = alpha_lookup.get(model_opt.generator_function,
                                 model_opt.loss_alpha)
    if not hasattr(model_opt, 'global_attention_alpha'):
        model_opt.global_attention_alpha = \
            alpha_lookup[model_opt.global_attention_function]
    if not hasattr(model_opt, 'global_attention_bisect_iter'):
        model_opt.global_attention_bisect_iter = 0
    model = build_base_model(model_opt, fields, gpu, checkpoint)

    assert opt.k == 0 or opt.bisect_iter == 0, \
        "Bisection and topk are mutually exclusive ! !"
    if gen_alpha == 1.0:
        gen_func = nn.Softmax(dim=-1)
    elif gen_alpha == 2.0:
        if k > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxTopK(
                dim=-1, k=k)
        elif bisect_iter > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxBisect(
                n_iter=bisect_iter)
        else:
            gen_func = onmt.modules.sparse_activations.Sparsemax(dim=-1)
    elif gen_alpha == 1.5 and bisect_iter == 0:
        if k > 0:
            gen_func = onmt.modules.sparse_activations.Tsallis15TopK(
                dim=-1, k=k)
        else:
            gen_func = onmt.modules.sparse_activations.Tsallis15(dim=-1)
    else:
        # generic tsallis with bisection
        assert bisect_iter > 0, "Must use bisection with alpha != 1,1.5,2"
        gen_func = onmt.modules.sparse_activations.TsallisBisect(
            alpha=gen_alpha, n_iter=bisect_iter)

    gen_weights = model.generator[0] if \
        isinstance(model.generator, nn.Sequential) else model.generator

    generator = nn.Sequential(gen_weights, gen_func)
    model.generator = generator

    model.eval()
    model.generator.eval()

    return model


def main(opt):
    # Build model.
    for path in opt.models:
        checkpoint = torch.load(
            path,
            map_location=lambda storage, loc: storage)
        fields = checkpoint['vocab']
        fields = {'src': fields['src'], 'tgt': fields['tgt']}
        model = load_model(
            checkpoint, fields, k=opt.k,
            bisect_iter=opt.bisect_iter, gpu=opt.gpu)

        tgt_padding_idx = \
            fields['tgt'].base_field.vocab.stoi[
                fields['tgt'].base_field.pad_token]

        validator = Validator(model, tgt_padding_idx)
        if opt.verbose:
            print(model.decoder.attn)
            print(model.generator)

        valid_iter = build_dataset_iter(
            "valid", fields, checkpoint['opt'], is_train=False)

        valid_stats = validator.validate(valid_iter)
        print('avg. attended positions/tgt word: {}'.format(
            valid_stats['attended'] / valid_stats['tgt_words']))
        print('avg. support size: {}'.format(
            valid_stats['support'] / valid_stats['tgt_words']))
        print('attention density: {}'.format(
            valid_stats['attended'] / valid_stats['attended_possible']))

        for ii in range(valid_stats['n_layers']):

            density_per_head = \
                [x / valid_stats['self_attended_possible']
                 for x in valid_stats['self_layer_%i' % (ii)]]

            density_per_head.sort()
            density_per_head = \
                [float('%.2f' % elem) for elem in density_per_head]

            print(
                ('self attention density in layer %i, per head (sorted): {}'
                 % (ii)
                 ).format(density_per_head)
            )

            print(
                ('average self attention density in layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (sum(density_per_head) / len(density_per_head)))
            )

            union_of_heads = \
                valid_stats[
                    'head_sum_self_layer_%i' % (ii)
                    ] / valid_stats['attended_possible']
            print(
                ('sum of heads self attention density in layer %i: {}'
                 % (ii)
                 ).format("%.2f" % union_of_heads)
            )

            print(
                ('JS_div of head agreement in self layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (valid_stats['JS_div_self_layer_%i' % (ii)]))
            )

        print('\n')

        for ii in range(valid_stats['n_layers']):

            density_per_head = \
                [x / valid_stats['attended_possible']
                 for x in valid_stats['context_layer_%i' % (ii)]]

            density_per_head.sort()
            density_per_head = \
                [float('%.2f' % elem) for elem in density_per_head]

            print(
                ('context attention density in layer %i, per head (sorted): {}'
                 % (ii)
                 ).format(density_per_head)
            )

            print(
                ('average context attention density in layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (sum(density_per_head) / len(density_per_head)))
            )

            union_of_heads = \
                valid_stats[
                    'head_sum_context_layer_%i' % (ii)
                    ] / valid_stats['attended_possible']
            print(
                ('sum of heads context attention density in layer %i: {}'
                 % (ii)
                 ).format("%.2f" % union_of_heads)
            )

            print(
                ('JS_div of head agreement in context layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (valid_stats['JS_div_context_layer_%i' % (ii)]))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-gpu', action='store_true')
    parser.add_argument('-models', nargs='+')
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-k', default=0, type=int)
    parser.add_argument('-bisect_iter', default=0, type=int)
    opt = parser.parse_args()
    main(opt)
