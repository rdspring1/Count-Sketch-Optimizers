def add_recurrent_args(parser):
    parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
    parser.add_argument('--proj', type=bool, default=True,
                    help='use linear projection layer to map LSTM to word embeddings')
    parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.01,
                    help='dropout applied to layers (0 = no dropout)')
    return parser

def add_transformer_args(parser):
    parser.add_argument('--emsize', type=int, default=1024,
                        help='size of word embeddings')
    parser.add_argument('--dropout', type=float, default=0.01,
                        help='dropout probability -- transformer only')
    parser.add_argument('--attention-dropout', type=float, default=0.0,
                        help='dropout probability for attention weights -- transformer only')
    parser.add_argument('--relu-dropout', type=float, default=0.1,
                        help='dropout probability after ReLU in FFN -- transformer only')
    #ignore the encoder args for transformer. That's meant for seq2seq transformer
    parser.add_argument('--encoder-embed-path', type=str, default=None,
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-embed-dim', type=int, default=1024, # originally 512 but 64 for char level
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', type=int, default=4096, # originally 2048 but scaled for char level
                        help='encoder embedding dimension for FFN')
    parser.add_argument('--encoder-layers', type=int, default=18,
                        help='num encoder layers')
    parser.add_argument('--encoder-attention-heads', type=int, default=8,
                        help='num encoder attention heads')
    parser.add_argument('--encoder-normalize-before', default=True, action='store_true',
                        help='apply layernorm before each encoder block')
    parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                        help='use learned positional embeddings in the encoder')
    parser.add_argument('--decoder-embed-path', type=str, default=None,
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-embed-dim', type=int, default=1024, # originally 512 but 64 for char level
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-ffn-embed-dim', type=int, default=4096, # originally 2048 but scaled for char level
                        help='decoder embedding dimension for FFN')
    parser.add_argument('--decoder-layers', type=int, default=18,
                        help='num decoder layers')
    parser.add_argument('--decoder-attention-heads', type=int, default=8,
                        help='num decoder attention heads')
    parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                        help='use learned positional embeddings in the decoder')
    parser.add_argument('--decoder-normalize-before', default=True, action='store_true',
                        help='apply layernorm before each decoder block')
    parser.add_argument('--chkpt-grad', default=True, action='store_true',
                        help='checkpoint gradients to allow for training with larger models and sequences')
    return parser
