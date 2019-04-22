#!/usr/bin/env python

import argparse
import pickle

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
                 'attended': 0, 'context_attended_possible': 0,
                 'self_attended_possible': 0,
                 'enc_self_attended_possible': 0}
        N_JS_div_enc = 0
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
                stats['context_attended_possible'] += grid_sizes.sum().item()

                self_grid_sizes = tgt_lengths * (tgt_lengths + 1) / 2
                stats['self_attended_possible'] += self_grid_sizes.sum().item()

                enc_self_grid_sizes = src_lengths * src_lengths
                stats['enc_self_attended_possible'] += \
                    enc_self_grid_sizes.sum().item()

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

                self_attn = attns['encoder_attn']
                src_non_pad = src.ne(self.tgt_padding_idx).view(-1)
                for ii, layer in enumerate(self_attn):
                    if 'enc_self_layer_%i' % (ii) not in stats.keys():
                        # number of heads
                        stats['enc_self_layer_%i' % (ii)] = [0] * layer.size(0)
                        stats['head_sum_enc_self_layer_%i' % (ii)] = 0
                        stats['JS_div_enc_self_layer_%i' % (ii)] = 0
                    for jj, head in enumerate(layer):
                        head = head.view(-1, head.size(-1))
                        attended = head.gt(0).sum(dim=1)
                        attended_non_pad = attended.masked_select(src_non_pad)
                        stats['enc_self_layer_%i' % (ii)][jj] += \
                            attended_non_pad.sum().item()

                    head_sum = layer.sum(0).view(-1, head.size(-1))
                    attended = head_sum.gt(0).sum(dim=1)
                    attended_non_pad = attended.masked_select(src_non_pad)
                    stats['head_sum_enc_self_layer_%i' % (ii)] += \
                        attended_non_pad.sum().item()

                    JS_div = \
                        (shannon_entropy(
                            (layer.sum(0) / layer.size(0)).view(
                                -1, layer.size(-1))) -
                         (shannon_entropy(layer) / layer.size(0)).sum(0).view(
                            -1)).masked_select(src_non_pad)

                    # JS Div
                    stats['JS_div_enc_self_layer_%i' % (ii)] += \
                        (JS_div /
                            torch.log(
                                JS_div.new_ones(JS_div.size())*layer.size(0)
                                )).sum().item()

                N_JS_div_enc += len(JS_div)

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
                stats['JS_div_enc_self_layer_%i' % (ii)] /= N_JS_div_enc
                stats['JS_div_self_layer_%i' % (ii)] /= N_JS_div
                stats['JS_div_context_layer_%i' % (ii)] /= N_JS_div

            # NORMS
            for ii, layer in enumerate(self_attn):
                # Norm of output attention layer (head choosing)
                w_o = \
                    self.model.encoder.transformer[
                        ii].self_attn.final_linear.weight
                n_heads = \
                    self.model.encoder.transformer[
                        ii].self_attn.head_count
                model_dim = w_o.size(0)
                head_size = model_dim // n_heads
                w_o = \
                    w_o.view(
                        model_dim, n_heads, head_size).transpose(
                            0, 1).contiguous().view(
                                n_heads, -1)
                stats['norm_enc_self_layer_%i' % (ii)] = \
                    w_o.norm(dim=1).tolist()

                # Decoder
                w_o = \
                    self.model.decoder.transformer_layers[
                        ii].self_attn.final_linear.weight
                n_heads = \
                    self.model.decoder.transformer_layers[
                        ii].self_attn.head_count
                model_dim = w_o.size(0)
                head_size = model_dim // n_heads
                w_o = \
                    w_o.view(
                        model_dim, n_heads, head_size).transpose(
                            0, 1).contiguous().view(
                                n_heads, -1)
                stats['norm_self_layer_%i' % (ii)] = \
                    w_o.norm(dim=1).tolist()

                # Context Att
                w_o = \
                    self.model.decoder.transformer_layers[
                        ii].context_attn.final_linear.weight
                n_heads = \
                    self.model.decoder.transformer_layers[
                        ii].context_attn.head_count
                model_dim = w_o.size(0)
                head_size = model_dim // n_heads
                w_o = \
                    w_o.view(
                        model_dim, n_heads, head_size).transpose(
                            0, 1).contiguous().view(
                                n_heads, -1)
                stats['norm_context_layer_%i' % (ii)] = \
                    w_o.norm(dim=1).tolist()

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
    for ith_model, path in enumerate(opt.models):
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
        model_opt = checkpoint['opt']
        model_opt.accum_count = [1]
        valid_iter = build_dataset_iter(
            "valid", fields, checkpoint['opt'], is_train=False)

        valid_stats = validator.validate(valid_iter)

        if opt.stats_file:
            with open(opt.stats_file[ith_model], 'wb') as f:
                pickle.dump(valid_stats, f)
        # print('avg. attended positions/tgt word: {}'.format(
        #     valid_stats['attended'] / valid_stats['tgt_words']))
        print('avg. support size: {}'.format(
            valid_stats['support'] / valid_stats['tgt_words']))
        # print('attention density: {}'.format(
        #     valid_stats['attended'] / valid_stats['attended_possible']))

        # ENC SELF ATT
        for ii in range(valid_stats['n_layers']):

            density_per_head = \
                [x / valid_stats['enc_self_attended_possible']
                 for x in valid_stats['enc_self_layer_%i' % (ii)]]

            density_per_head.sort()
            density_per_head = \
                [float('%.2f' % elem) for elem in density_per_head]

            print(
                ('enc_self att density in layer %i, per head (sorted): {}'
                 % (ii)
                 ).format(density_per_head)
            )

            print(
                ('average enc_self att density in layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (sum(density_per_head) / len(density_per_head)))
            )

            union_of_heads = \
                valid_stats[
                    'head_sum_enc_self_layer_%i' % (ii)
                    ] / valid_stats['enc_self_attended_possible']
            print(
                ('sum of heads enc_self att density in layer %i: {}'
                 % (ii)
                 ).format("%.2f" % union_of_heads)
            )

            print(
                ('JS_div of head disagreement in enc_self layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (valid_stats['JS_div_enc_self_layer_%i' % (ii)]))
            )

        print('\n')

        # SELF ATT
        for ii in range(valid_stats['n_layers']):

            density_per_head = \
                [x / valid_stats['self_attended_possible']
                 for x in valid_stats['self_layer_%i' % (ii)]]

            density_per_head.sort()
            density_per_head = \
                [float('%.2f' % elem) for elem in density_per_head]

            print(
                ('self att density in layer %i, per head (sorted): {}'
                 % (ii)
                 ).format(density_per_head)
            )

            print(
                ('average self att density in layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (sum(density_per_head) / len(density_per_head)))
            )

            union_of_heads = \
                valid_stats[
                    'head_sum_self_layer_%i' % (ii)
                    ] / valid_stats['self_attended_possible']
            print(
                ('sum of heads self att density in layer %i: {}'
                 % (ii)
                 ).format("%.2f" % union_of_heads)
            )

            print(
                ('JS_div of head disagreement in self layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (valid_stats['JS_div_self_layer_%i' % (ii)]))
            )

        print('\n')

        # CONTEXT ATT

        for ii in range(valid_stats['n_layers']):

            density_per_head = \
                [x / valid_stats['context_attended_possible']
                 for x in valid_stats['context_layer_%i' % (ii)]]

            density_per_head.sort()
            density_per_head = \
                [float('%.2f' % elem) for elem in density_per_head]

            print(
                ('context att density in layer %i, per head (sorted): {}'
                 % (ii)
                 ).format(density_per_head)
            )

            print(
                ('average context att density in layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (sum(density_per_head) / len(density_per_head)))
            )

            union_of_heads = \
                valid_stats[
                    'head_sum_context_layer_%i' % (ii)
                    ] / valid_stats['context_attended_possible']
            print(
                ('sum of heads context att density in layer %i: {}'
                 % (ii)
                 ).format("%.2f" % union_of_heads)
            )

            print(
                ('JS_div of head disagreement in context layer %i: {}'
                 % (ii)
                 ).format(
                    "%.2f" % (valid_stats['JS_div_context_layer_%i' % (ii)]))
            )

        print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-gpu', action='store_true')
    parser.add_argument('-models', nargs='+')
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-k', default=0, type=int)
    parser.add_argument('-bisect_iter', default=0, type=int)
    parser.add_argument('-stats_file', nargs='+', default=None, type=str)
    opt = parser.parse_args()
    main(opt)
