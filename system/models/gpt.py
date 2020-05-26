# -*- coding: utf-8 -*- 

import argparse
import json
import os, sys
import numpy as np
import tensorflow as tf
sys.path.append(os.curdir+'/src')
import model
import sample
import refine_punc
import tokenization
import utils


class GPT:
    def __init__(self,
                 checkpoint_path,
                 device,
                 seed=None,
                 nsamples=1,
                 batch_size=1,
                 tok_length=256,
                 sent_length=3,
                 top_k=0,
                 top_p=0.0):

        if device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

        self.conditional = True

        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0
        self.batch_size = batch_size
        self.nsamples = nsamples

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file='./vocab/vocab.txt',
            do_lower_case=False)
        self.en = False
        print('Korean GPT loaded!')

        hparams = model.default_hparams()
        with open(os.path.join('./vocab/hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if tok_length is None:
            tok_length = hparams.n_ctx // 2
        elif tok_length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        start_token = None

        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams, length=tok_length,
            context=self.context,
            start_token=start_token,
            batch_size=batch_size,
            temperature=1, top_k=top_k, top_p=top_p)

        self.sent_length = sent_length

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, ckpt)

    def infer(self, raw_text):
        output = []
        if self.conditional:
            if self.en:
                context_tokens = self.enc.encode(raw_text)
            else:
                context_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(raw_text))
            for sample_id in range(self.nsamples // self.batch_size):
                out = self.sess.run(self.output, feed_dict={
                    self.context: [context_tokens for _ in range(1)]})[:, len(context_tokens):]
                for batch_id in range(self.batch_size):
                    if self.en:
                        text = self.enc.decode(out[batch_id])
                        text = text.split('.')[:self.sent_length]
                        if text[-1] != '':
                            text = text + ['']
                        text = '. '.join([i.strip() for i in text]).strip()
                        output.append(text)

                    else:
                        text = self.tokenizer.convert_ids_to_tokens(out[batch_id])
                        text = refine_punc.refine_punc(text)
                        text = ' '.join(utils.rm_sp(utils.convert_text(text).split('.. '))[:self.sent_length])
                        output.append(text)
        else:
            generated = 0
            while self.nsamples == 0 or generated < self.nsamples:
                out = self.sess.run(self.output)
                for batch_id in range(self.batch_size):
                    generated += self.batch_size
                    if self.en:
                        text = self.enc.decode(out[batch_id])
                    else:
                        text = self.tokenizer.convert_ids_to_tokens(out[batch_id])
                        text = refine_punc.refine_punc(text)
                    text = text.split('.')[:self.sent_length]
                    if text[-1] != '':
                        text = text + ['']
                    text = '. '.join([i.strip() for i in text]).strip()
                    output.append(text)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default='./checkpoint/test/',
                        help='trained checkpoint path')
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nsamples", type=int, default=1) #2
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tok_length", type=int, default=128) #128
    parser.add_argument("--sent_length", type=int, default=5) #3
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=.0)
    parser.add_argument("--context", type=str, default="트럼프")
    args = parser.parse_args()

    model = GPT(args.checkpoint_path,
                args.device,
                args.seed,
                args.nsamples,
                args.batch_size,
                args.tok_length,
                args.sent_length,
                args.top_k,
                args.top_p)

    out = model.infer(args.context)
    print(out)
