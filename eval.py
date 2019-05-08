import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


# sentences = [
#   # From July 8, 2017 New York Times:
#   'Scientists at the CERN laboratory say they have discovered a new particle.',
#   'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#   'President Trump met with other leaders at the Group of 20 conference.',
#   'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
#   # From Google's Tacotron example page:
#   'Generative adversarial network or variational auto-encoder.',
#   'The buses aren\'t the problem, they actually provide a solution.',
#   'Does the quick brown fox jump over the lazy dog?',
#   'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
# ]
sentences = [
  'lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de5 di3 se4 si4 yue4 de5 lin2 luan2 geng4 shi4 lv4 de5 xian1'
  'huo2 xiu4 mei4 shi1 yi4 ang4 ran2 l v4 sh ix4 ii iang2 ch un1 ii ian1 j ing3 d a4 k uai4 uu un2 zh ang1 d e5 d i3 s e4'
  's iy4 vv ve4 d e5 l in2 l uan2 g eng4 sh ix4 l v4 d e5 x ian1 h uo2 x iu4 m ei4 sh ix1 ii i4 aa ang4 r an2'
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
