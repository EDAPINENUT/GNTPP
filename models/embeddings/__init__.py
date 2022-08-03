from .time import *
from .type import *
from .position import *


EMBEDDING_DICT = {
    'Trigo': TrigonoTimeEmbedding,
    'Linear': LinearTimeEmbedding
}

def get_embedding(**args):
    time_emb_name = args['time_emb']
    return EMBEDDING_DICT[time_emb_name](embed_size=args['embed_size'] // 2),\
           TypeEmbedding(embed_size = args['embed_size'] // 2, padding_idx=args['event_type_num'], event_type_num=args['event_type_num']),\
           PositionEmbedding(embed_size = args['embed_size'] // 2, max_length=args['max_length'])
